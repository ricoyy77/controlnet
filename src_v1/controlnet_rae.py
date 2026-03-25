


import torch
import torch.nn as nn
from copy import deepcopy
from stage2.models.DDT import DiTwDDTHead

def zero_module(module):
    """
    ControlNet 标准输出端零初始化
    """
    for p in module.parameters():
        p.detach().zero_() # 使用 detach().zero_() 更加彻底，和代码1保持一致
    return module

class RAEControlNet(nn.Module):
    def __init__(self, base_model: DiTwDDTHead, downsample_ratio=16):
        """
        downsample_ratio: 默认为16 (对应 256 pixel -> 16 latent)
        """
        super().__init__()
        
        encoder_hidden_size = base_model.encoder_hidden_size
        num_encoder_blocks = base_model.num_encoder_blocks
        
        # 1. 复制 Base Model 核心组件 (保持 RAE 的特有组件不变)
        self.t_embedder = deepcopy(base_model.t_embedder)
        self.y_embedder = deepcopy(base_model.y_embedder)
        self.pos_embed = deepcopy(base_model.pos_embed) 
        self.enc_feat_rope = deepcopy(base_model.enc_feat_rope)
        self.s_embedder = deepcopy(base_model.s_embedder) # RAE 特有的 latent embedder
        
        self.encoder_blocks = nn.ModuleList([
            deepcopy(base_model.blocks[i]) for i in range(num_encoder_blocks)
        ])
        
        # 2. 【核心修改】Hint Embedder 改为简单的 Patchify 逻辑
        # 对应代码 1 的 self.hint_embedder
        # 抛弃了之前的 5 层卷积 + SiLU
        self.hint_embedder = nn.Conv2d(
            in_channels=1,                # 假设输入是 Canny (1通道)
            out_channels=encoder_hidden_size,
            kernel_size=downsample_ratio, # 16
            stride=downsample_ratio,      # 16
            bias=True
        )
        
        # 3. 【核心修改】初始化策略对齐代码 1
        # 使用 Xavier Uniform 初始化权重，Bias 设为 0
        # 这意味着训练开始时，Control 信号就已经有数值了，而不是全 0
        # nn.init.xavier_uniform_(self.hint_embedder.weight.view([encoder_hidden_size, -1]))
        # nn.init.constant_(self.hint_embedder.bias, 0)
        nn.init.zeros_(self.hint_embedder.weight)
        nn.init.zeros_(self.hint_embedder.bias)

        
        # (移除了 LayerNorm，因为代码1没有)

        # 4. Zero Linear Layers (输出层)
        self.zero_layers = nn.ModuleList([
            zero_module(nn.Linear(encoder_hidden_size, encoder_hidden_size))
            for _ in range(num_encoder_blocks)
        ])

    def forward(self, x_noisy, x_hint, t, y):
        """
        x_noisy: (B, C, H, W) -> Latent
        x_hint:  (B, 1, H_img, W_img) -> Canny
        """
        # 1. 处理 Latent (Base Model 逻辑)
        x = self.s_embedder(x_noisy)
        
        # 2. 【逻辑对齐】处理 Hint
        # 直接卷积 Patchify: [B, 1, 256, 256] -> [B, D, 16, 16]
        hint_emb = self.hint_embedder(x_hint)
        
        # 调整形状: [B, D, 16, 16] -> [B, 256, D]
        hint_emb = hint_emb.flatten(2).transpose(1, 2)
        
        # 3. 【逻辑对齐】直接融合
        # 简单相加
        CONTROL_INPUT_SCALE = 1
        x = x + CONTROL_INPUT_SCALE * hint_emb
        
        # 4. Pos Embed & Time/Label (保持 RAE 逻辑)
        if self.pos_embed is not None:
            x = x + self.pos_embed
            
        t_emb = self.t_embedder(t)
        # 注意：这里保留 RAE 的 training 标志逻辑，或者你可以根据需要改成 fixed
        y_emb = self.y_embedder(y, self.training) 
        
        # RAE 的条件融合方式可能和 Code 1 略有不同 (SiLU vs Add)，这里保留 RAE 原生的方式比较安全
        # 只要 Hint 的注入方式改了就行
        c = nn.functional.silu(t_emb + y_emb)
        
        residuals = []
        
        # 5. Encoder Loop
        for block, zero_layer in zip(self.encoder_blocks, self.zero_layers):
            x = block(x, c, feat_rope=self.enc_feat_rope)
            
            # 输出残差
            res = zero_layer(x)
            
            # 【逻辑对齐】直接 append，不乘 scale (默认就是 1.0)
            residuals.append(res)
            
        return residuals