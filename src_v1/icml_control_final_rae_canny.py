import torch
import torch.nn as nn
import math
from time import time
import argparse
import cv2
import numpy as np
from PIL import Image
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

# 确保能找到项目中的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.model_utils import instantiate_from_config
from stage2.transport import create_transport, Sampler
from utils.train_utils import parse_configs
from stage1 import RAE
from controlnet_rae import RAEControlNet 
from torchvision.utils import save_image

# 开启加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =======================================================
# 1. 核心模型包装器 (保持不变)
# =======================================================
class JointRAEWrapperWithStrength(nn.Module):
    def __init__(self, base_model, control_model, control_scale=1.0):
        super().__init__()
        self.base_model = base_model
        self.control_model = control_model
        self.control_scale = control_scale

    def forward(self, x, t, y, canny, **kwargs):
        control_residuals = self.control_model(x, canny, t, y)
        if self.control_scale != 1.0:
            control_residuals = [res * self.control_scale for res in control_residuals]
        return self.base_model(x, t, y, control_residuals=control_residuals)

# =======================================================
# 2. 预处理工具函数 (保持不变)
# =======================================================
def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

# =======================================================
# 3. 新增：批量 Canny 处理函数
# =======================================================
def batch_process_canny(pil_images, image_size=256):
    """
    输入: list of PIL Images
    输出: Tensor [B, 1, 256, 256]
    """
    canny_list = []
    for img in pil_images:
        # 1. Center Crop Resize
        img_cropped = center_crop_arr(img, image_size)
        img_np = np.array(img_cropped)
        
        # 2. Canny Edge
        edge = cv2.Canny(img_np, 100, 200)
        
        # 3. To Tensor
        # edge shape: [256, 256]
        canny_tensor = torch.from_numpy(edge).float() / 255.0
        
        # 增加 Channel 维度 -> [1, 256, 256]
        canny_tensor = canny_tensor.unsqueeze(0) 
        canny_list.append(canny_tensor)
        
    # Stack 之后已经是 [B, 1, 256, 256] 了，不要再 unsqueeze 了
    return torch.stack(canny_list, dim=0)

# =======================================================
# 4. 新增：自定义数据集 (用于返回文件名)
# =======================================================
class ImageNetValDataset(Dataset):
    def __init__(self, root_dir):
        # 使用 ImageFolder 自动处理 nXXXXXX -> class_index 的映射
        # ImageFolder 会按照文件夹名字母顺序排序，这正是 ImageNet 的标准索引顺序
        self.ds = datasets.ImageFolder(root=root_dir)
        self.samples = self.ds.samples # list of (path, class_index)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        filename = os.path.basename(path)
        # 返回: 原图(PIL), 类别索引(int), 文件名(str)
        return img, target, filename

def collate_fn(batch):
    # 自定义 collate 因为 PIL Image 不能直接被 default_collate 变成 tensor
    images = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch])
    filenames = [item[2] for item in batch]
    return images, targets, filenames

# =======================================================
# 5. CFG 包装函数 (保持不变)
# =======================================================
def joint_fwd_with_cfg(x, t, y, canny, cfg_scale, cfg_interval, joint_model):
    x_in = torch.cat([x, x], dim=0)
    t_in = torch.cat([t, t], dim=0) if t.ndim > 0 else t
    model_out = joint_model(x_in, t_in, y, canny)
    cond_out, uncond_out = model_out.chunk(2, dim=0)
    guided_out = uncond_out + cfg_scale * (cond_out - uncond_out)
    return guided_out

# =======================================================
# 6. 主程序
# =======================================================
def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. 输出目录设置 (修改了这里) ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建存放生成图的文件夹
    gen_save_dir = os.path.join(args.output_dir, "gen_vis")
    os.makedirs(gen_save_dir, exist_ok=True)
    
    # 创建存放 Canny 图的文件夹
    canny_save_dir = os.path.join(args.output_dir, "canny_vis")
    os.makedirs(canny_save_dir, exist_ok=True)

    print(f"📂 Output Directories:\n  - Gen: {gen_save_dir}\n  - Canny: {canny_save_dir}")

    # --- 加载配置 ---
    rae_config, model_config, transport_config, sampler_config, guidance_config, misc, _, _ = parse_configs(args.config)
    
    # --- 加载模型 ---
    print("🚀 Loading Models...")
    rae = instantiate_from_config(rae_config).to(device).eval()
    base_model = instantiate_from_config(model_config).to(device)
    if args.base_ckpt:
        ckpt = torch.load(args.base_ckpt, map_location="cpu")
        base_model.load_state_dict(ckpt.get('model', ckpt), strict=False)
    base_model.eval()

    control_model = RAEControlNet(base_model).to(device)
    if args.control_ckpt:
        c_ckpt = torch.load(args.control_ckpt, map_location="cpu")
        control_model.load_state_dict(c_ckpt.get('model', c_ckpt), strict=True)
    control_model.eval()

    joint_model = JointRAEWrapperWithStrength(base_model, control_model, control_scale=args.control_scale).to(device)

    # --- Sampler ---
    latent_size = misc.get("latent_size", (768, 16, 16))
    transport = create_transport(**transport_config['params'])
    sampler = Sampler(transport)
    if sampler_config['mode'] == "ODE":
        sample_fn = sampler.sample_ode(**sampler_config['params'])
    else:
        sample_fn = sampler.sample_sde(**sampler_config['params'])

    # --- 数据集加载器 ---
    print(f"📂 Loading ImageNet Val from: {args.val_dir}")
    dataset = ImageNetValDataset(args.val_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                        num_workers=4, collate_fn=collate_fn)

    print(f"🚀 Start generating {len(dataset)} images...")

    # --- 批量处理循环 ---
    for batch_idx, (pil_imgs, labels, filenames) in enumerate(loader):
        n = len(pil_imgs)
        
        # 准备 Condition
        canny = batch_process_canny(pil_imgs, image_size=256).to(device)
        y = labels.to(device)
        z = torch.randn(n, *latent_size, device=device)

        # CFG 设置
        guidance_scale = args.cfg_scale
        if guidance_scale > 1.0:
            y_null = torch.tensor([1000] * n, device=device)
            y_combined = torch.cat([y, y_null], 0)
            canny_combined = torch.cat([canny, canny], 0)
            t_min = guidance_config.get("t_min", 0.0)
            t_max = guidance_config.get("t_max", 1.0)
            
            model_kwargs = dict(
                y=y_combined, canny=canny_combined, 
                cfg_scale=guidance_scale, cfg_interval=(t_min, t_max),
                joint_model=joint_model
            )
            sample_model_fwd = joint_fwd_with_cfg
        else:
            model_kwargs = dict(y=y, canny=canny)
            sample_model_fwd = joint_model.forward

        # 采样
        print(f"🎨 Batch {batch_idx}/{len(loader)} | Size: {n}")
        samples = sample_fn(z, sample_model_fwd, **model_kwargs)[-1]
        
        # 反归一化
        # if args.stats_path:
        #     stats = torch.load(args.stats_path, map_location=device)
        #     if stats['mean'].ndim == 1:
        #          stats['mean'] = stats['mean'].view(1, -1, 1, 1)
        #          stats['std'] = stats['std'].view(1, -1, 1, 1)
        #     samples = samples * stats['std'] + stats['mean']

        # 解码
        images = rae.decode(samples)

        # --- 2. 保存逻辑 (修改了这里) ---
        for i in range(n):
            fname = filenames[i]
            fname_no_ext = os.path.splitext(fname)[0]
            
            # 保存到 gen_vis 文件夹
            # 这里的路径变了：从 args.output_dir 变成了 gen_save_dir
            save_path = os.path.join(gen_save_dir, f"{fname_no_ext}_gen.png")
            save_image(images[i], save_path, normalize=True, value_range=(0, 1))
            
            # 保存到 canny_vis 文件夹
            canny_save_path = os.path.join(canny_save_dir, f"{fname_no_ext}_canny.png")
            save_image(canny[i], canny_save_path)

        del samples, images, canny, z

    print("✅ All finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径配置
    parser.add_argument("--config", type=str, default="/data3/user/myy2/control_canny/RAE/configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml")
    parser.add_argument("--base_ckpt", type=str, default="/data3/user/myy2/control_canny/RAE/models/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt")
    parser.add_argument("--control_ckpt", type=str, default="/data3/user/myy2/control_canny/RAE/src_v1/results/rae_controlnet_run1/checkpoint_epoch_100.pt")
    parser.add_argument("--stats_path", default="/data2/user/myy/controlnet/RAE_CN/RAE/src/rae_features/rae_latents_train_r0.01/latents_stats.pt")
    
    # 数据集路径
    parser.add_argument("--val_dir", type=str, default="/data2/user/myy/controlnet/imagenet1k/val", help="ImageNet val directory with structure val/nXXXX/image.jpg")
    parser.add_argument("--output_dir", type=str, default="results_imagenet_val_v1", help="Directory to save results")
    
    # 采样参数
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--control_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)