import os
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from safetensors import safe_open

class SafetensorsControlDataset(Dataset):
    def __init__(self, data_dir, latent_norm=True, latent_multiplier=1.0):
        self.data_dir = data_dir
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier

        # 1. 扫描文件
        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        if not self.files:
            raise ValueError(f"No .safetensors found in {data_dir}")

        # 2. 建立索引映射 (官方逻辑)
        print(f"Indexing {len(self.files)} safetensors files...")
        self.img_to_file_map = self.get_img_to_safefile_map()
        
        # 3. 计算或加载统计数据 (官方逻辑)
        # 注意：只对 Latent 做归一化，Canny 不需要
        if latent_norm:
            self._latent_mean, self._latent_std = self.get_latent_stats()

    def get_img_to_safefile_map(self):
        img_to_file = {}
        # 只需要遍历文件头，速度很快
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                # 读取 labels 的 shape 来确定该文件有多少张图
                labels_slice = f.get_slice('labels')
                num_imgs = labels_slice.get_shape()[0]
                
                cur_len = len(img_to_file)
                for i in range(num_imgs):
                    img_to_file[cur_len + i] = {
                        'safe_file': safe_file,
                        'idx_in_file': i
                    }
        return img_to_file

    def get_latent_stats(self):
        """
        计算 Latent 的均值和方差。
        如果有缓存文件 latents_stats.pt 则直接加载。
        """
        latent_stats_cache_file = os.path.join(self.data_dir, "latents_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            print("Computing latent stats (this happens only once)...")
            latent_stats = self.compute_latent_stats()
            torch.save(latent_stats, latent_stats_cache_file)
        else:
            print("Loading latent stats from cache...")
            latent_stats = torch.load(latent_stats_cache_file)
        return latent_stats['mean'], latent_stats['std']
    
    def compute_latent_stats(self):
        # 随机采样 10000 张图来估算均值方差
        num_samples = min(10000, len(self.img_to_file_map))
        random_indices = np.random.choice(len(self.img_to_file_map), num_samples, replace=False)
        latents = []
        
        for idx in tqdm(random_indices, desc="Computing Stats"):
            img_info = self.img_to_file_map[idx]
            safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                # 只读取 Latent
                features = f.get_slice('latents')
                feature = features[img_idx:img_idx+1]
                latents.append(feature)
                
        latents = torch.cat(latents, dim=0)
        # 计算全局 mean/std
        mean = latents.mean(dim=[0, 2, 3], keepdim=True)
        std = latents.std(dim=[0, 2, 3], keepdim=True)
        latent_stats = {'mean': mean, 'std': std}
        print(f"Computed Latent Stats -> Mean: {mean.mean().item():.4f}, Std: {std.mean().item():.4f}")
        return latent_stats

    def __len__(self):
        return len(self.img_to_file_map.keys())

    def __getitem__(self, idx):
        img_info = self.img_to_file_map[idx]
        safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            # 50% 概率翻转
            is_flip = np.random.uniform(0, 1) > 0.5
            
            # [关键修改] 同步选择 key
            if is_flip:
                latent_key = "latents_flip"
                canny_key = "cannys_flip"
            else:
                latent_key = "latents"
                canny_key = "cannys"
            
            # 读取数据 slice
            latent_slice = f.get_slice(latent_key)
            canny_slice = f.get_slice(canny_key)
            label_slice = f.get_slice('labels')
            
            # 获取具体索引的数据
            latent = latent_slice[img_idx:img_idx+1] # [1, C, H, W]
            canny = canny_slice[img_idx:img_idx+1]   # [1, 1, H, W]
            label = label_slice[img_idx:img_idx+1]   # [1]

        # --- Latent 处理 (标准化) ---
        if self.latent_norm:
            latent = (latent - self._latent_mean) / self._latent_std
        latent = latent * self.latent_multiplier
        
        # --- Canny 处理 (无标准化) ---
        # Canny 已经是 [0, 1] 了，不需要减 mean/std
        # 只需要确保是 Float
        
        # 去掉 batch 维度
        latent = latent.squeeze(0) # [C, H, W]
        canny = canny.squeeze(0)   # [1, H, W]
        label = label.squeeze(0)   # Scalar
        
        return latent, label, canny