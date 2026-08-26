# 专题 05：DiT 与 MM-DiT 架构精解 (Diffusion Transformer in SD3 / Flux.1)
> **技术定位**: 终结 UNet 的现代生成大模型核心主干  
> **核心概念**: Patchify、adaLN-Zero 零初始化自适应调制、SD3 MM-DiT 双流联合注意力、Flux.1 QK-Norm 防熵坍塌。

---

## 模块 1：核心动机 (The "Why")

1. **严格的 Scaling Law**：DiT 将模型算力与参数规模转化为了持续单调提升的生成 FID 质量。
2. **移除局部归纳偏置**：全局自注意力赋予模型理解复杂物理规律与构图的能力。
3. **共享 LLM 基础设施**：无缝对接 FlashAttention-3、Ring Attention 与 ZeRO 并行系统。

---

## 模块 2：核心机制数学拆解

### 2.1 adaLN-Zero (Adaptive LayerNorm with Zero Initialization)
从时间步 $t$ 与条件 $c$ 预测 6 个调制参数 $(\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2)$：
$$\hat{x} = \text{LayerNorm}(x) \cdot (1 + \gamma_1) + \beta_1$$
$$x_{\text{attn}} = x + \alpha_1 \cdot \text{Attention}(\hat{x})$$
$$x_{\text{out}} = x_{\text{attn}} + \alpha_2 \cdot \text{MLP}\big(\text{LayerNorm}(x_{\text{attn}}) \cdot (1 + \gamma_2) + \beta_2\big)$$

**关键设计**：初始化时将门控参数 $\alpha_1, \alpha_2$ 的投影权重置 0，使初始层严格等价于恒等映射 $x_{\text{out}}=x$。

### 2.2 MM-DiT 双流联合注意力
$$Q = \text{Concat}(Q_{\text{img}}, Q_{\text{text}}), \quad K = \text{Concat}(K_{\text{img}}, K_{\text{text}}), \quad V = \text{Concat}(V_{\text{img}}, V_{\text{text}})$$
经过全量联合自注意力后，拆分图像流与文本流，分别通过各自专用的 MLP 前馈网络。

---

## 模块 3：PyTorch 核心实现

```python
import torch
import torch.nn as nn
from typing import Tuple

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        norm_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        norm_x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x2)
        return x
```
