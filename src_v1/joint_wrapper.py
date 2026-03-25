import torch
import torch.nn as nn

class JointRAEWrapper(nn.Module):
    def __init__(self, base_model, control_model):
        super().__init__()
        self.base_model = base_model
        self.control_model = control_model

    def forward(self, x, t, y, canny):
        """
        x: Noisy RAE Latents [B, 768, 16, 16]
        """
        # [关键] 强制输入需要梯度，以便 Gradient Checkpointing 生效
        if not x.requires_grad:
            x.requires_grad_(True)
            
        # 1. ControlNet
        control_residuals = self.control_model(x, canny, t, y)
        
        # 2. Base Model (Encoder-Decoder)
        # 注意：你需要确保你的 models.py 里的 forward 已经加上了 control_residuals 参数
        pred = self.base_model(x, t, y, control_residuals=control_residuals)
        
        return pred