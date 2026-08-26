# 专题 02：3D Causal VAE 与时空分词 (Causal 3D VAE for Video Generation)
> **技术定位**: 现代视频生成大模型的高效时空压缩编码器  
> **核心概念**: 时间因果卷积 (Causal 3D Conv)、$4 \times 8 \times 8$ 时空压缩、首帧一致性、流式分块解码。

---

## 模块 1：核心动机 (The "Why")

1. **时间单向流动性**：因果卷积只聚合历史帧，杜绝未来信息穿越。
2. **流式与自回归无缝兼容**：支持像音频一样流式生成长视频。
3. **首帧一致性**：当视频仅有 1 帧时，数学上严格等价于 2D 图像 VAE。

---

## 模块 2：数学与非对称 Padding

对于卷积核 $(K_t, K_h, K_w) = (3, 3, 3)$：
- 空间对称填充：$(1, 1)$
- 时间非对称因果填充：$(\text{Front}=K_t - 1=2, \ \text{Back}=0)$

---

## 模块 3：PyTorch 核心实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3, 3), stride=(1, 1, 1), **kwargs):
        super().__init__()
        self.kt, self.kh, self.kw = kernel_size
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(self.kt, self.kh, self.kw),
            stride=stride,
            padding=(0, self.kh // 2, self.kw // 2),
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_pad = (0, 0, 0, 0, self.kt - 1, 0)
        x_padded = F.pad(x, time_pad, mode='replicate')
        return self.conv(x_padded)
```
