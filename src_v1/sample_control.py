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

# 确保能找到项目中的模块 (如果你在 src 下运行，这会把根目录加入路径)
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
# 1. 核心模型包装器 (支持推理时的强度控制)
# =======================================================
class JointRAEWrapperWithStrength(nn.Module):
    def __init__(self, base_model, control_model, control_scale=1.0):
        super().__init__()
        self.base_model = base_model
        self.control_model = control_model
        self.control_scale = control_scale

    def forward(self, x, t, y, canny, **kwargs):
        # 提取残差
        control_residuals = self.control_model(x, canny, t, y)
        
        # 应用控制强度缩放
        if self.control_scale != 1.0:
            control_residuals = [res * self.control_scale for res in control_residuals]
            
        # 注入到 Base Model
        return self.base_model(x, t, y, control_residuals=control_residuals)

# =======================================================
# 2. 预处理工具函数
# =======================================================
def center_crop_arr(pil_image, image_size):
    # 保持长宽比缩放
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

def get_canny_tensor(image_path, image_size=256):
    """读取参考图，提取 Canny 边缘并返回 [1, 1, 256, 256] Tensor"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Canny reference image not found: {image_path}")
    img = Image.open(image_path).convert('RGB')
    img = center_crop_arr(img, image_size)
    img_np = np.array(img)
    edge = cv2.Canny(img_np, 100, 200)
    # 转为 Tensor 范围 [0, 1]
    canny_tensor = torch.from_numpy(edge).float() / 255.0
    canny_tensor = canny_tensor.unsqueeze(0).unsqueeze(0) # [1, 1, 256, 256]
    return canny_tensor

# =======================================================
# 3. CFG 包装函数 (解决 ODE Solver 的形状对齐问题)
# =======================================================
def joint_fwd_with_cfg(x, t, y, canny, cfg_scale, cfg_interval, joint_model):
    """
    x: [n, 768, 16, 16] (来自 Solver)
    t: [n]
    y: [2n] (包含有条件和空类)
    canny: [2n, 1, 256, 256]
    """
    # 1. 内部临时扩充为 2倍 Batch 进行并行推理
    x_in = torch.cat([x, x], dim=0)
    t_in = torch.cat([t, t], dim=0) if t.ndim > 0 else t
    
    # 2. 调用包装后的模型
    model_out = joint_model(x_in, t_in, y, canny)
    
    # 3. 拆分 cond 和 uncond 结果
    cond_out, uncond_out = model_out.chunk(2, dim=0)
    
    # 4. 执行分类器自由引导 (Classifier-Free Guidance)
    guided_out = uncond_out + cfg_scale * (cond_out - uncond_out)
    
    # 5. 返回 n 个样本给 Solver，确保形状一致
    return guided_out

# =======================================================
# 4. 主程序
# =======================================================
def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 加载配置 ---
    # 建议在项目根目录运行，否则此处 config 内部的路径可能失效
    rae_config, model_config, transport_config, sampler_config, guidance_config, misc, _, _ = parse_configs(args.config)
    
    # --- 实例化并加载模型 ---
    print("🚀 Loading RAE Stage1 (Tokenizer)...")
    rae = instantiate_from_config(rae_config).to(device)
    rae.eval()
    
    print("🚀 Loading Base Model (DDT)...")
    base_model = instantiate_from_config(model_config).to(device)
    if args.base_ckpt:
        ckpt = torch.load(args.base_ckpt, map_location="cpu")
        base_model.load_state_dict(ckpt.get('model', ckpt), strict=False)
    base_model.eval()

    print("🚀 Loading ControlNet & Joint Wrapper...")
    control_model = RAEControlNet(base_model).to(device)
    if args.control_ckpt:
        c_ckpt = torch.load(args.control_ckpt, map_location="cpu")
        control_model.load_state_dict(c_ckpt.get('model', c_ckpt), strict=True)
    control_model.eval()

    # 使用带强度控制的 Wrapper
    joint_model = JointRAEWrapperWithStrength(base_model, control_model, control_scale=args.control_scale).to(device)

    # --- 设置 Sampler & Transport ---
    latent_size = misc.get("latent_size", (768, 16, 16))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    
    transport = create_transport(**transport_config['params'], time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)
    
    if sampler_config['mode'] == "ODE":
        sample_fn = sampler.sample_ode(**sampler_config['params'])
    else:
        sample_fn = sampler.sample_sde(**sampler_config['params'])

    # --- 准备输入数据 ---
    canny = get_canny_tensor(args.canny_path, image_size=256).to(device)
    class_labels = [int(x.strip()) for x in args.classes.split(',')]
    n = len(class_labels)
    
    z = torch.randn(n, *latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # --- 核心：CFG 引导设置 ---
    guidance_scale = args.cfg_scale
    if guidance_scale > 1.0:
        y_null = torch.tensor([1000] * n, device=device) # 默认 1000 为空类
        y_combined = torch.cat([y, y_null], 0)
        # Canny 需要在两路路径（有条件和无条件）中都输入
        canny_combined = torch.cat([canny.repeat(n,1,1,1), canny.repeat(n,1,1,1)], 0)
        
        t_min = guidance_config.get("t_min", 0.0)
        t_max = guidance_config.get("t_max", 1.0)
        
        model_kwargs = dict(
            y=y_combined, 
            canny=canny_combined, 
            cfg_scale=guidance_scale,
            cfg_interval=(t_min, t_max),
            joint_model=joint_model
        )
        sample_model_fwd = joint_fwd_with_cfg
    else:
        # 无 CFG 模式
        canny_in = canny.repeat(n, 1, 1, 1)
        model_kwargs = dict(y=y, canny=canny_in)
        sample_model_fwd = joint_model.forward

    # --- 执行采样推理 ---
    print(f"🎨 Sampling: {n} images | CFG: {guidance_scale} | Control Strength: {args.control_scale}")
    start_time = time()
    
    # Solver 调用返回的是轨迹，取最后一帧 [-1]
    samples = sample_fn(z, sample_model_fwd, **model_kwargs)[-1]
    
    # --- 极其重要：反归一化 (De-normalization) ---
    if args.stats_path:
        print(f"📊 Applying stats from {args.stats_path}")
        stats = torch.load(args.stats_path, map_location=device)
        # stats['mean'] 和 stats['std'] 的形状应为 [1, 768, 1, 1]
        samples = samples * stats['std'] + stats['mean']
        

    # 解码潜空间到图像
    print("📸 RAE Decoding...")
    images = rae.decode(samples)
    
    print(f"✨ Finished in {time() - start_time:.2f}s.")

    # 保存结果
    out_name = f"control_res_s{args.control_scale}_cfg{args.cfg_scale}_{int(time())}.png"
    save_image(images, out_name, nrow=n, normalize=True, value_range=(0, 1))
    save_image(canny, "reference_canny_used.png")
    print(f"✅ Saved to {out_name} (Reference canny saved as reference_canny_used.png)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/data3/user/myy2/control_canny/RAE/configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml")
    parser.add_argument("--base_ckpt", type=str, default="/data3/user/myy2/control_canny/RAE/models/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt")
    parser.add_argument("--control_ckpt", type=str, default="/data3/user/myy2/control_canny/RAE/src_v1/results/rae_controlnet_run1/checkpoint_epoch_100.pt")
    parser.add_argument("--canny_path", type=str, required=True, help="Reference image for Canny edge")
    parser.add_argument("--stats_path", default="/data2/user/myy/controlnet/RAE_CN/RAE/src/rae_features/rae_latents_train_r0.01/latents_stats.pt")
    parser.add_argument("--classes", type=str, required=True, help="ImageNet class IDs, e.g., '207,360'")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--control_scale", type=float, default=1.0, help="ControlNet residual strength (e.g., 1.0-1.5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)