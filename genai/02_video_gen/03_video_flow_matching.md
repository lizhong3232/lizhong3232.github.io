# 专题 03：视频流匹配与 I2V 条件控制 (Video Flow Matching & I2V)
> **技术定位**: 现代视频大模型的核心动力学引擎 (Wan 2.1 / HunyuanVideo / MovieGen)  
> **核心概念**: 4D 时空速度场、直线概率路径、I2V 首帧条件拼接 (Concat) vs 交叉注意力 (Cross-Attn)、Camera Motion LoRA。

---

## 模块 1：4D 速度场与直线路径

$$x_\tau = (1 - \tau) x_0 + \tau x_1, \quad u_\tau(x_\tau | x_1) = x_1 - x_0$$
模型预测时空 4D 向量场 $v_\theta(x_\tau, \tau, c)$，使得 ODE 求解器能以极少步数（10~20 步）完成整段高保真视频流生成。

---

## 模块 2：I2V 注入两大范式

1. **通道拼接 (Channel Concatenation)**：首帧以特征通道拼叠在输入端，首帧像素 100% 锁定不漂移。
2. **交叉注意力 (Cross-Attention)**：首帧经编码器注入 Key/Value，动态交互幅度更大。

---

## 模块 3：PyTorch 核心实现

```python
import torch
import torch.nn as nn

class VideoFlowMatchingLoss(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x_1: torch.Tensor, first_frame: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x_1.shape
        tau = torch.rand((B, 1, 1, 1, 1), device=x_1.device)
        x_0 = torch.randn_like(x_1)
        x_tau = (1.0 - tau) * x_0 + tau * x_1
        target_v = x_1 - x_0
        
        cond = torch.zeros_like(x_1)
        cond[:, :, 0, :, :] = first_frame
        
        v_pred = self.model(x_tau, cond, tau.squeeze())
        return torch.mean((v_pred - target_v) ** 2)
```
