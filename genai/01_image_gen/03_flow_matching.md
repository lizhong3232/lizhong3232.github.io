# 专题 03：Flow Matching 与 Rectified Flow 直线流 (SD3 / Flux / Sora 底座)
> **技术定位**: 连续归一化流与直线最优传输生成大一统 (Lipman et al., ICLR 2023 / Liu et al., ICLR 2023)  
> **核心概念**: 连续性方程 (Continuity Equation)、条件向量场回归、欧拉极速采样、Reflow 轨迹直化与 1~2 步极速出图。

---

## 模块 1：核心动机与物理直觉 (The "Why")

### 1.1 弯曲的盘山公路 vs 笔直的高铁隧道
传统扩散（DDPM/DDIM）在方差保持超球面上修建弯弯曲曲的盘山公路。求解器过急弯时如果步子迈得太大，就会产生严重的截断截曲误差，直接飞出悬崖（图像崩溃）。

### 1.2 两点之间线段最短
**Flow Matching 的破局大智慧**：直接从噪声起点 $x_0 \sim \mathcal{N}(0, I)$ 到真实数据终点 $x_1 \sim p_{\text{data}}$ **铺设一条笔直的高铁隧道**！在直线上速度方向恒定不变，一阶欧拉积分的单步截断误差几乎为 0，仅需 10~15 步即可极速出图！

---

## 模块 2：严谨数学推导拆解 (Mathematical Derivations)

### 2.1 连续性方程 (Continuity Equation) 与流场
连续归一化流（CNF）通过含时速度场 $v_t(x)$ 驱动先验概率密度 $p_0$ 演化为数据分布 $p_1$。其连续概率密度演化满足流体力学连续性方程（质量守恒定律）：
$$\frac{\partial p_t(x)}{\partial t} + \nabla \cdot \big( p_t(x) v_t(x) \big) = 0$$

### 2.2 条件流匹配 (Conditional Flow Matching, CFM)
构造直线插值概率路径：
$$x_t = (1 - t) x_0 + t x_1, \quad t \in [0, 1]$$

条件解析速度场就是时间一阶导数：
$$u_t(x_t | x_0, x_1) = \frac{d x_t}{dt} = \frac{d}{dt}\big( (1 - t) x_0 + t x_1 \big) = x_1 - x_0$$

神经网络 $v_\theta(x_t, t)$ 只需要做最简单的无约束 MSE 回归：
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0, 1], x_0 \sim \mathcal{N}(0, I), x_1 \sim p_{\text{data}}} \left[ \Big\| v_\theta(x_t, t) - (x_1 - x_0) \Big\|_2^2 \right]$$

---

### 2.3 轨迹交叉与 Reflow 直化理论
虽然单对样本是直线，但不同随机样本对之间会在高维空间发生轨迹交叉（Trajectory Crossing）。在交差点，边缘速度场 $v(x, t) = \mathbb{E}[u_t | x_t=x]$ 变为多个速度矢量的平均值，导致宏观速度场微弯。

在生成的新确定性数据对 $(x_0, \hat{x}_1)$ 上进行 2-Reflow 微调训练，所有流线完全解耦平行，Euler 欧拉积分单步即可达成极高生成质量！

---

## 模块 3：核心算法代码实现与逐行精讲 (Code Walkthrough)

```python
import torch
import torch.nn as nn

class FlowMatchingPipeline:
    """
    工业级 Flow Matching 与 Rectified Flow 核心管线 (PyTorch)
    """
    def __init__(self, model: nn.Module):
        self.model = model  # 速度预测网络 (如 MM-DiT 或 DiT)

    def training_loss(self, x_1: torch.Tensor) -> torch.Tensor:
        """
        Flow Matching 训练损失计算:
        x_1: 真实目标图像 [Batch, Channels, Height, Width]
        """
        B = x_1.shape[0]
        device = x_1.device

        # 1. 采样源分布标准高斯白噪声 x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)

        # 2. 均匀采样连续时间步 t in [0, 1]
        t = torch.rand(B, 1, 1, 1, device=device)

        # 3. 线性概率路径插值: x_t = (1 - t) * x_0 + t * x_1 (等价于 torch.lerp)
        x_t = torch.lerp(x_0, x_1, t)

        # 4. 解析真实瞬时速度场: u_t = x_1 - x_0
        target_velocity = x_1 - x_0

        # 5. 神经网络前向预测速度向量场
        pred_velocity = self.model(x_t, t.view(B))

        # 6. 极简 MSE 回归损失
        loss = torch.mean((pred_velocity - target_velocity) ** 2)
        return loss

    @torch.no_grad()
    def sample_euler(self, shape: tuple, steps: int = 15, cfg_scale: float = 1.0, null_cond = None, text_cond = None) -> torch.Tensor:
        """
        一阶欧拉 ODE 采样推演: 从 t=0 (纯噪声 x_0) 积分推演至 t=1 (生成图像 x_1)
        """
        device = next(self.model.parameters()).device
        dt = 1.0 / steps
        x = torch.randn(shape, device=device)  # x(0)

        for step in range(steps):
            t_val = step * dt
            t_tensor = torch.full((shape[0],), t_val, device=device)

            if cfg_scale > 1.0 and text_cond is not None:
                # 结合 CFG 引导: v_final = v_uncond + w * (v_cond - v_uncond)
                v_uncond = self.model(x, t_tensor, cond=null_cond)
                v_cond = self.model(x, t_tensor, cond=text_cond)
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = self.model(x, t_tensor)

            # 欧拉单步位移积分: x(t + dt) = x(t) + v * dt
            x = x + v_pred * dt

        return x  # x(1) 得到生成图像
```

### 逐行核心张量解析
- `torch.lerp(x_0, x_1, t)`：底层 CUDA 优化的线性插值算子，计算 $(1-t)x_0 + tx_1$ 精度更高且避免额外内存分配。
- `target_velocity = x_1 - x_0`：常数差值速度向量，无任何复杂的方差调度超参。
- `x = x + v_pred * dt`：由于直线轨迹曲率接近 0，单步截断误差极小，15 步即可无损出图。

---

## 模块 4：工业界落地要点与核心踩坑点 (Engineering Pitfalls)

1. **时间步采样分布加权 (Logit-Normal Timestep Sampling)**：
   在 SD3/Flux.1 中，采用 Logit-Normal 采样分布（集中采样中间阶段），显著提升复杂文本构图的收敛效率。
2. **Guidance Interval 与 Rescale 技巧**：
   全程开启大 CFG 会破坏末端细节；采用 Guidance Interval（仅在 $t \in [0.1, 0.9]$ 开启 CFG）兼具高语义对齐与细腻质感。

---

## 模块 5：自测思考题 (Self-Assessment)

- **Q1: 为什么说 Flow Matching 是比标准扩散模型更通用的生成大一统理论？**
  * *解析*：标准扩散模型被证明是 Flow Matching 在方差保持非线性路径下的特例。Flow Matching 原生支持从任意先验分布 $p_0$（如低分辨率图、草图）到目标分布 $p_1$ 的直接映射。
- **Q2: 在 Rectified Flow 中，为什么 1-Reflow 之后轨迹可以完全平行而不交叉？**
  * *解析*：数据对 $(x_0, \hat{x}_1)$ 来自一阶段确定性 ODE 积分，根据 Picard-Lindelöf 存在唯一性定理，两条不同初值的确定性轨迹永不相交，消除交叉速度平均化带来的弯曲。
