import torch
# 开启 TF32 加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from safetensors.torch import save_file
from datetime import datetime
from omegaconf import OmegaConf
from tqdm import tqdm

# =======================================================
# 1. 导入 RAE 模块
# =======================================================
try:
    from utils.model_utils import instantiate_from_config
except ImportError:
    print("❌ 错误: 找不到 utils.model_utils。请确保在 RAE 项目根目录下运行。")
    exit(1)

# =======================================================
# 2. 预处理函数
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
# 3. RAE 专用 Dataset
# =======================================================
class RAEExtractionDataset(ImageFolder):
    def __init__(self, root, target_size=256, encoder_size=224, force_flip=False, sample_ratio=1.0):
        super().__init__(root)
        self.target_size = target_size
        self.encoder_size = encoder_size
        self.force_flip = force_flip
        
        if 0.0 < sample_ratio < 1.0:
            total_len = len(self.samples)
            step = int(1 / sample_ratio)
            self.samples = self.samples[::step]
            print(f"📉 [采样模式] Rank {dist.get_rank() if dist.is_initialized() else 0} | 数量: {total_len} -> {len(self.samples)}")
        
        self.rae_transform = transforms.Compose([
            transforms.Resize((encoder_size, encoder_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path) 
        except Exception:
            sample = Image.new('RGB', (self.target_size, self.target_size))

        sample = center_crop_arr(sample, self.target_size)
        
        # 强制翻转逻辑
        if self.force_flip:
            sample = sample.transpose(Image.FLIP_LEFT_RIGHT)

        # Canny 分支 (256x256)
        img_np = np.array(sample)
        edge = cv2.Canny(img_np, 100, 200)
        canny_tensor = torch.from_numpy(edge).float() / 255.0
        canny_tensor = canny_tensor.unsqueeze(0) 

        # RAE 分支 (224x224)
        rae_input = self.rae_transform(sample)

        return rae_input, target, canny_tensor

# =======================================================
# 4. 主程序
# =======================================================
def main(args):
    # --- DDP 初始化 ---
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
    except:
        print("DDP Init Failed. Running locally.")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # --- 输出路径 ---
    ratio_str = f"_r{args.sample_ratio}" if args.sample_ratio < 1.0 else ""
    output_dir = os.path.join(args.output_path, f"rae_latents_{args.data_split}{ratio_str}")
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📂 Output Dir: {output_dir}")

    # --- 加载模型 ---
    if rank == 0: print("🚀 Loading RAE Model...")
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.stage_1)
    
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
        elif "model" in ckpt: ckpt = ckpt["model"]
        new_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(new_ckpt, strict=False)
    
    model.to(device).eval()
    model.requires_grad_(False)

    # --- 数据集 (两路：原图 & 翻转) ---
    ds_normal = RAEExtractionDataset(args.data_path, force_flip=False, sample_ratio=args.sample_ratio)
    ds_flipped = RAEExtractionDataset(args.data_path, force_flip=True, sample_ratio=args.sample_ratio)
    
    samp_normal = DistributedSampler(ds_normal, num_replicas=world_size, rank=rank, shuffle=False)
    samp_flipped = DistributedSampler(ds_flipped, num_replicas=world_size, rank=rank, shuffle=False)

    loader_normal = DataLoader(ds_normal, batch_size=args.batch_size, sampler=samp_normal, num_workers=args.num_workers, pin_memory=True)
    loader_flipped = DataLoader(ds_flipped, batch_size=args.batch_size, sampler=samp_flipped, num_workers=args.num_workers, pin_memory=True)

    # --- 提取循环 ---
    latents, cannys, labels = [], [], []
    latents_flip, cannys_flip = [], [] # 修正：这里只初始化两个列表
    saved_files = 0
    
    # 进度条仅在主进程显示
    pbar = tqdm(total=len(loader_normal), desc=f"Rank {rank}", disable=rank != 0)

    for batch_normal, batch_flipped in zip(loader_normal, loader_flipped):
        # 处理原图
        img_n, lbl, edg_n = batch_normal
        img_n = img_n.to(device, non_blocking=True)
        
        # 处理翻转图
        img_f, _, edg_f = batch_flipped
        img_f = img_f.to(device, non_blocking=True)

        with torch.no_grad():
            # 获取原图 Latent
            z_n = model.encode(img_n).detach().cpu()
            # 获取翻转图 Latent
            z_f = model.encode(img_f).detach().cpu()

        latents.append(z_n)
        cannys.append(edg_n)
        labels.append(lbl)
        
        latents_flip.append(z_f)
        cannys_flip.append(edg_f)

        pbar.update(1)

        # 每 5000 条记录保存一次分片 (防止内存溢出)
        if len(latents) * args.batch_size >= 5000:
            out = {
                "latents": torch.cat(latents, dim=0).contiguous(),
                "cannys": torch.cat(cannys, dim=0).contiguous(),
                "labels": torch.cat(labels, dim=0).contiguous(),
                "latents_flip": torch.cat(latents_flip, dim=0).contiguous(),
                "cannys_flip": torch.cat(cannys_flip, dim=0).contiguous(),
            }
            fname = os.path.join(output_dir, f"rae_shard_{rank:02d}_{saved_files:03d}.safetensors")
            save_file(out, fname)
            
            if rank == 0: print(f"\n✅ Saved {fname}")
            
            # 清空缓存
            latents, cannys, labels = [], [], []
            latents_flip, cannys_flip = [], []
            saved_files += 1

    # 保存最后剩余的数据
    if len(latents) > 0:
        out = {
            "latents": torch.cat(latents, dim=0).contiguous(),
            "cannys": torch.cat(cannys, dim=0).contiguous(),
            "labels": torch.cat(labels, dim=0).contiguous(),
            "latents_flip": torch.cat(latents_flip, dim=0).contiguous(),
            "cannys_flip": torch.cat(cannys_flip, dim=0).contiguous(),
        }
        fname = os.path.join(output_dir, f"rae_shard_{rank:02d}_{saved_files:03d}.safetensors")
        save_file(out, fname)
        if rank == 0: print(f"✅ Saved last shard {fname}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="./rae_features")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)