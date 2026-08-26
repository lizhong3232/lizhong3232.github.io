# 专题 06：可控生成与极速蒸馏 (ControlNet / IP-Adapter / DMD / ADD)
> **技术定位**: 精准空间几何控制与 1~4 步实时生成引擎  
> **核心概念**: ControlNet 零卷积微调、IP-Adapter 解耦 Cross-Attention、一致性模型 (Consistency Models)、分布匹配蒸馏 (DMD / ADD)。

---

## 模块 1：核心动机 (The "Why")

1. **空间可控性需求**：纯文本无法精确指定骨骼姿态、线稿与深度。
2. **极速实时推理需求**：工业级应用（实时交互、游戏渲染）需要将 20~50 步采样压缩到 1~4 步。

---

## 模块 2：核心机制数学拆解

### 2.1 ControlNet 零卷积机制
$$y = \mathcal{F}(x; \Theta) + \mathcal{Z}\Big( \mathcal{F}\big(x + \mathcal{Z}(c; \Theta_{z1}); \ \Theta_c\big); \ \Theta_{z2} \Big)$$
初始时零卷积 $\mathcal{Z}(\cdot)=0$，保证模型初始状态严格等价于原始底模。

### 2.2 IP-Adapter 解耦交叉注意力
$$Z = \text{softmax}\left(\frac{Q K_{\text{txt}}^T}{\sqrt{d}}\right)V_{\text{txt}} + \lambda \cdot \text{softmax}\left(\frac{Q K_{\text{img}}^T}{\sqrt{d}}\right)V_{\text{img}}$$

### 2.3 DMD (Distribution Matching Distillation)
$$\nabla_\theta \mathcal{D}_{\text{KL}}(p_\theta \| p_{\text{real}}) = \mathbb{E}_{x_0 \sim p_\theta}\Big[ \nabla_{x_0} \log p_{\text{real}}(x_0) - \nabla_{x_0} \log p_{\text{fake}}(x_0) \Big]$$

---

## 模块 3：PyTorch 核心实现

```python
import torch
import torch.nn as nn

def make_zero_conv(channels: int) -> nn.Conv2d:
    conv = nn.Conv2d(channels, channels, 1)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv

class ControlNetToyBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.locked_block = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU())
        for p in self.locked_block.parameters():
            p.requires_grad = False
            
        self.trainable_block = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU())
        self.zero_conv_in = make_zero_conv(channels)
        self.zero_conv_out = make_zero_conv(channels)

    def forward(self, x: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        base_out = self.locked_block(x)
        control_in = x + self.zero_conv_in(hint)
        control_out = self.zero_conv_out(self.trainable_block(control_in))
        return base_out + control_out
```
