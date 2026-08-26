# 专题 01：时空 3D DiT 架构设计 (Spatiotemporal DiT for Video Generation)
> **技术定位**: 现代视频生成大模型的主干基石 (Sora, Wan 2.1, HunyuanVideo, CogVideoX)  
> **核心概念**: 3D Patchification、分因子时空解耦注意力 (Factorized Space-Time Attention)、3D RoPE、Zero-init 时序层初始化。

---

## 模块 1：核心动机与视频 Token 爆炸 (The "Why")

### 1.1 朴素 4D 全注意力的显存灾难
一段 5 秒 24 FPS 的高清视频（120 帧，分辨率 $1024 \times 1024$），经过 $16 \times 16$ 空间 Patchify 后：
$$N = 120 \times (64 \times 64) \approx 491,520 \text{ 个 Token}$$
全时空自注意力的复杂度为 $\mathcal{O}(N^2) \approx 2.4 \times 10^{11}$ 次计算，单张 H100 显存瞬间 OOM 爆炸成千上万倍！

### 1.2 工业级破局方案
1. **3D Patchification**：在时间轴上也做 Patch 压缩（如 $2 \times 16 \times 16$），将时间帧序列减半。
2. **分因子时空解耦注意力 (Factorized Attention)**：
   - 步骤 1：帧内空间自注意力 $\mathcal{O}(T \cdot S^2)$；
   - 步骤 2：跨帧时间自注意力 $\mathcal{O}(S \cdot T^2)$；
   - 计算量从 $\mathcal{O}((T \cdot S)^2)$ 骤降 99% 以上！

---

## 模块 2：严谨数学推导与 3D RoPE (Mathematical Derivations)

### 2.1 解耦时空注意力公式
对于输入潜码张量 $Z \in \mathbb{R}^{B \times T \times S \times D}$（其中 $S = H \times W$）：
1. **空间维度自注意力**：
   $$Z_{\text{spatial}} = \text{Attention}\big(\text{reshape}(Z, (BT, S, D))\big) \in \mathbb{R}^{BT \times S \times D}$$
2. **时间维度自注意力**：
   $$Z_{\text{temporal}} = \text{Attention}\big(\text{transpose}(Z_{\text{spatial}}, (BS, T, D))\big) \in \mathbb{R}^{BS \times T \times D}$$

### 2.2 3D 旋转位置编码 (3D RoPE)
Wan 2.1 与 Sora 采用 3D RoPE 替代传统 1D/2D 绝对编码：
$$\mathbf{R}_{3D}(t, h, w) = \text{diag}\Big( \mathbf{R}_{\text{time}}(t), \ \mathbf{R}_{\text{height}}(h), \ \mathbf{R}_{\text{width}}(w) \Big)$$
将特征通道维度 $D$ 均分为 3 份，分别应用时间、高度、宽度的旋转变换矩阵，天然支持可变分辨率与可变帧率推理。

---

## 模块 3：PyTorch 极简参考代码与张量流转

```python
import torch
import torch.nn as nn
from einops import rearrange

class SpatioTemporalDiTBlock(nn.Module):
    """
    工业级时空解耦 DiT 模块 (Factorized Space-Time DiT Block)
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        # 1. 空间注意力 (Spatial Self-Attention)
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # 2. 时间注意力 (Temporal Self-Attention)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # 3. 前馈网络 (MLP)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, H, W, D] 4D 视频潜码 Token
        """
        B, T, H, W, D = x.shape
        S = H * W  # 空间 Token 总数

        # === 1. 空间自注意力 (每一帧内部独立做 Attention) ===
        x_space = rearrange(x, 'b t h w d -> (b t) (h w) d')
        norm_space = self.spatial_norm(x_space)
        attn_out_space, _ = self.spatial_attn(norm_space, norm_space, norm_space)
        x_space = x_space + attn_out_space

        # === 2. 时间自注意力 (每个空间位置跨时间帧做 Attention) ===
        x_time = rearrange(x_space, '(b t) s d -> (b s) t d', b=B, t=T, s=S)
        norm_time = self.temporal_norm(x_time)
        attn_out_time, _ = self.temporal_attn(norm_time, norm_time, norm_time)
        x_time = x_time + attn_out_time

        # === 3. MLP 前馈与残差恢复 ===
        x_out = rearrange(x_time, '(b h w) t d -> b t h w d', b=B, h=H, w=W)
        x_out = x_out + self.mlp(self.mlp_norm(x_out))
        
        return x_out
```

---

## 模块 4：工业实战避坑指南 (Pitfalls)

1. **时间层 Zero-init 初始化**：
   - 从 2D 图像预训练权重迁移到视频模型时，必须将时间注意力输出层的权重初始化为 **全 0**，防止初始噪声破坏图像单帧先验。
2. **长视频 Context Parallelism (Ring Attention / Ulysses)**：
   - 面对超过 100k 的超长视频序列，单卡无法容纳，必须使用 Ring Attention 或 DeepSpeed Ulysses 将时间/空间 Token 沿显卡集群分布式切分。
