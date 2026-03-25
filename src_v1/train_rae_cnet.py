import argparse
import os
import torch
import yaml
import math
from copy import deepcopy
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed

# --- Imports ---
from utils.model_utils import instantiate_from_config
from stage2.transport import create_transport, Sampler
from controlnet_rae import RAEControlNet
from joint_wrapper import JointRAEWrapper
from dataset_cnet import SafetensorsControlDataset 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to RAE Stage2 config.yaml")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Pretrained Stage 2 Model .pt")
    parser.add_argument("--data_path", type=str, required=True, help="Path to extracted features (.safetensors)")
    parser.add_argument("--results_dir", type=str, default="results/rae_controlnet")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="bf16",
        project_dir=args.results_dir
    )
    print(f"目前的混合精度设置: {accelerator.mixed_precision}")
    set_seed(args.global_seed)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("rae_controlnet_logs")
        os.makedirs(args.results_dir, exist_ok=True)
        print(f"🚀 Training RAE ControlNet on {accelerator.device}")

    # 1. Load Config
    cfg = OmegaConf.load(args.config)
    
    # 2. Init Base Model
    if accelerator.is_main_process:
        print("Initializing Base Model from Config...")
    base_model = instantiate_from_config(cfg.stage_2)
    
    # Load Weights
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    base_model.load_state_dict(new_state_dict, strict=False)
        
    # Freeze Base Model
    base_model.eval()
    base_model.requires_grad_(False)
    for module in base_model.modules():
        if hasattr(module, 'use_checkpoint'):
            module.use_checkpoint = True

    # 3. Init ControlNet
    control_model = RAEControlNet(base_model)
    control_model.train()
    
    # Wrapper
    joint_model = JointRAEWrapper(base_model, control_model)

    # 4. Dataset
    dataset = SafetensorsControlDataset(
        data_dir=args.data_path,
        latent_norm=True,
        latent_multiplier=1.0 
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # 5. Transport & Optimizer
    transport_cfg = cfg.transport.params
    latent_size = cfg.misc.latent_size 
    shift_dim = cfg.misc.get("time_dist_shift_dim", math.prod(latent_size))
    
    
    shift_base = cfg.misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    print(f"📊 检查 Time Dist Shift: {time_dist_shift:.4f}")
    
    transport = create_transport(
        path_type=transport_cfg.get('path_type', 'Linear'),
        prediction=transport_cfg.get('prediction', 'velocity'),
        loss_weight=transport_cfg.get('loss_weight', None),
        time_dist_shift=time_dist_shift 
    )

    optimizer = torch.optim.AdamW(control_model.parameters(), lr=args.lr)

    # Prepare
    control_model, optimizer, loader = accelerator.prepare(
        control_model, optimizer, loader
    )
    
    # Re-bind joint model components
    joint_model.base_model = base_model.to(accelerator.device)
    joint_model.control_model = control_model

    # 6. Loop
    if accelerator.is_main_process:
        print("Start Training...")
        
    global_step = 0
    for epoch in range(args.epochs):
        control_model.train()
        for batch_idx, (latents, labels, cannys) in enumerate(loader):
            with accelerator.accumulate(control_model):
                model_kwargs = dict(y=labels, canny=cannys)
                loss_dict = transport.training_losses(joint_model, latents, model_kwargs)
                loss = loss_dict["loss"].mean()
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(control_model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 10 == 0 and accelerator.is_main_process:
                    print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")
            
            # --- 原有的按 Step 保存逻辑 ---
            # if global_step > 0 and global_step % 20000 == 0 and accelerator.is_main_process:
            #     save_path = os.path.join(args.results_dir, f"checkpoint_step_{global_step}.pt")
            #     unwrapped = accelerator.unwrap_model(control_model)
            #     torch.save({'model': unwrapped.state_dict()}, save_path)
            #     print(f"💾 Saved Step Checkpoint: {save_path}")

        # --- 新增：按 Epoch 保存逻辑 (每 10 个 epoch) ---
        # 使用 epoch + 1 是为了直观（第10, 20...个epoch结束时保存）
        if (epoch + 1) % 20 == 0 and accelerator.is_main_process:
            save_path = os.path.join(args.results_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            unwrapped = accelerator.unwrap_model(control_model)
            # 建议保存为 dict 格式，方便后续加载
            torch.save({'model': unwrapped.state_dict(), 'epoch': epoch}, save_path)
            print(f"🌟 Saved Epoch Checkpoint: {save_path}")

    accelerator.end_training()

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=2 python train_rae_cnet.py    --config /data2/user/myy/controlnet/RAE_CN/RAE/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B.yaml  --ckpt_path /data2/user/myy/controlnet/RAE_CN/RAE/models/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt  --data_path /data2/user/myy/controlnet/RAE_CN/RAE/src/rae_features/rae_latents_train_r0.01  --results_dir results/rae_controlnet_run1   --batch_size 4  --grad_accum_steps 1  --epochs 50  --lr 1e-4