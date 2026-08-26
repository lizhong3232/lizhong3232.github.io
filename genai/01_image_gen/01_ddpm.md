# 专题 01：DDPM 深度精解与数学推导 (Denoising Diffusion Probabilistic Models)
> **技术定位**: 离散时间马尔可夫扩散生成基石 (Ho et al., NeurIPS 2020)  
> **核心概念**: 随机 SDE 模拟、高斯重参数化技巧、贝叶斯后验均值与方差、ELBO 极简 MSE 损失、工业工程落地陷阱。

---

## 模块 1：核心动机与物理直觉 (The "Why")

### 1.1 墨水在清水中的热力学扩散（熵增与信息擦除）
想象一滴纯黑墨水落入静止的清水杯中：由于水分子的无规则热运动碰撞（布朗运动），墨水分子逐渐扩散，图像空间结构被逐步破坏，最终整杯水达到最大熵状态——均匀浑浊的灰色高斯白噪声。

### 1.2 时间倒流录像带（逆向去噪凝聚）
宏观的不可逆破坏，在微观上是由每一步微小的高斯扰动累加而成的。**DDPM 的核心思想**：训练一个神经网络，在每个微小时间步估计水分子对墨水分子的“反向恢复力”。采样时，我们就像**把录像带倒放**一样，从一滩纯高斯噪声中逆向凝聚出一张清晰逼真的图像！

---

## 模块 2：严谨数学推导拆解 (Mathematical Derivations)

### 2.1 正向加噪马尔可夫链与重参数化闭式解
前向加噪过程是一个单向添加高斯噪声的高斯马尔可夫链：
$$q(x_t | x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I\right)$$

令 $\alpha_t = 1 - \beta_t$，以及累乘系数 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。利用独立高斯分布的线性相加方差相加性质：
$$\begin{aligned}
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1} \\
    &= \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}} \epsilon_{t-2}\right) + \sqrt{1 - \alpha_t} \epsilon_{t-1} \\
    &= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t(1 - \alpha_{t-1})} \epsilon_{t-2} + \sqrt{1 - \alpha_t} \epsilon_{t-1} \\
    &= \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \text{其中 } \epsilon \sim \mathcal{N}(0, I)
\end{aligned}$$

**核心结论**：无需循环迭代 $t$ 次，利用闭式解 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ 可在 $\mathcal{O}(1)$ 时间直接从 $x_0$ 采样任意时刻带噪图 $x_t$。

---

### 2.2 核心概念精讲：什么是信噪比 SNR (Signal-to-Noise Ratio)？

#### 1. 通俗直觉：嘈杂酒吧里的交谈
在信号处理中，**信噪比 (SNR)** 衡量的是**“有用信号功率”与“干扰噪声功率”的比值**：
$$\text{SNR} = \frac{P_{\text{signal}}}{P_{\text{noise}}} = \frac{\text{Var}(\text{Signal})}{\text{Var}(\text{Noise})}$$
就像在嘈杂酒吧里聊天：朋友说话声音越洪亮（信号大）、周围背景嘈杂声越小（噪声小），信噪比就越高，话语信息越清晰可辨。

#### 2. 扩散模型中的 SNR 精确数学形式
前向重参数化公式：
$$x_t = \underbrace{\sqrt{\bar{\alpha}_t} x_0}_{\text{保留的真实图像信号}} + \underbrace{\sqrt{1 - \bar{\alpha}_t} \epsilon}_{\text{注入的高斯白噪声}}$$
假设数据已归一化，$\text{Var}(x_0) = 1, \text{Var}(\epsilon) = 1$：
- **信号方差 (Signal Power)**：$\text{Var}(\sqrt{\bar{\alpha}_t} x_0) = (\sqrt{\bar{\alpha}_t})^2 = \bar{\alpha}_t$
- **噪声方差 (Noise Power)**：$\text{Var}(\sqrt{1 - \bar{\alpha}_t} \epsilon) = (\sqrt{1 - \bar{\alpha}_t})^2 = 1 - \bar{\alpha}_t$

两者相除，得到扩散模型中的**瞬时信噪比方程**：
$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}, \qquad \text{Log-SNR: } \lambda_t = \log \text{SNR}(t) = \log \left(\frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}\right)$$

#### 3. 为什么 SNR 是扩散模型的“第一性原理”？（3 大核心作用）
1. **决定不同阶段学什么**：
   - $\text{SNR} \to 0$ ($t \to 1000$)：噪声占绝对统治，模型只能学习宏观构图与大色调。
   - $\text{SNR} \gg 1$ ($t \to 0$)：信号清晰，模型专注精雕细琢高频发丝纹理。
2. **解释为何不能单步出图**：
   - 在 $t=1000$ 时，$\text{SNR} \approx 0$，输入纯噪声的信息量为 0。模型为了最小化 MSE 只能输出所有样本的“平均值”，导致单步生成必然是一团浆糊。
3. **指导现代 Loss 加权**：
   - 现代前沿模型（如 Min-SNR-5 Weighting, VDM）直接以 $\min(\text{SNR}(t), \gamma)$ 重新加权不同时刻的 Loss，大幅提升训练收敛速度与画质稳定性。

---

### 2.3 逆向后验条件分布推导 (Bayes' Rule)
根据贝叶斯定理，当已知初始真实图像 $x_0$ 时，反向单步分布 $q(x_{t-1} | x_t, x_0)$ 解析可求：
$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) q(x_{t-1} | x_0)}{q(x_t | x_0)} = \mathcal{N}\big(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I\big)$$

展开高斯概率密度指数项并配方，得到后验均值与方差：
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t, \quad \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

将 $x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}$ 代入均值消去 $x_0$：
$$\tilde{\mu}_t(x_t, \epsilon) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)$$

---

### 2.3 变分下界 (ELBO) 化简到极简 MSE 损失
最大化负对数似然变分下界 $\mathbb{E}[-\log p_\theta(x_0)] \le L_{\text{VLB}}$，KL 散度展开后简化为：
$$L_{\text{VLB}} = \mathbb{E}_{q}\left[ D_{\text{KL}}\big( q(x_T | x_0) \| p(x_T) \big) + \sum_{t > 1} D_{\text{KL}}\big( q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t) \big) - \log p_\theta(x_0 | x_1) \right]$$

Ho et al. 证明直接优化无加权噪声预测均方误差可获得最佳生成质量：
$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[1, T], x_0, \epsilon \sim \mathcal{N}(0, I)} \left[ \Big\| \epsilon - \epsilon_\theta(x_t, t) \Big\|_2^2 \right]$$

---

## 模块 3：核心算法代码实现与逐行精讲 (Code Walkthrough)

```python
import torch
import torch.nn as nn

class DDPM(nn.Module):
    """
    极简、自包含且工业可运行的 DDPM 核心管线 (PyTorch)
    """
    def __init__(self, model: nn.Module, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.model = model  # 噪声预测网络 (通常为 UNet 或 DiT)
        self.timesteps = timesteps

        # 1. 预先计算加噪常数表 (注册为持久 buffer)
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # 后验方差计算: \tilde{\beta}_t = (1 - \bar{\alpha}_{t-1}) / (1 - \bar{\alpha}_t) * \beta_t
        posterior_variance = (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) * betas
        self.register_buffer('posterior_variance', posterior_variance)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """前向重参数化加噪: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        训练前向: 随机抽取时间步 t，加噪并计算 MSE 损失
        x_0: [Batch, Channels, Height, Width]
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_0.device).long()
        x_t, noise = self.q_sample(x_0, t)
        
        pred_noise = self.model(x_t, t)
        loss = torch.mean((pred_noise - noise) ** 2)
        return loss

    @torch.no_grad()
    def p_sample_loop(self, shape: tuple) -> torch.Tensor:
        """反向采样循环: 从 x_T ~ N(0, I) 逐步推演 1000 步复原图像"""
        device = next(self.model.parameters()).device
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = self.model(x, t_tensor)
            
            alpha = self.alphas[t]
            alpha_bar = self.alphas_cumprod[t]
            beta = self.betas[t]

            mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise)
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.posterior_variance[t])
                x = mean + sigma * noise
            else:
                x = mean
        return x
```

### 逐行核心张量解析
- `alphas_cumprod.view(-1, 1, 1, 1)`：将形状 `[B]` 扩展为 `[B, 1, 1, 1]`，实现与 4D 图像张量的高效广播计算。
- `torch.randint(0, self.timesteps, (B,))`：每个 batch 样本在不同加噪深度独立训练，增强泛化性。
- `sigma = torch.sqrt(self.posterior_variance[t])`：后验条件方差 $\tilde{\beta}_t$ 在 $t=0$ 时自然为 0，保证图像最终输出干净。

---

## 模块 4：工业界落地要点与核心踩坑点 (Engineering Pitfalls)

1. **Latent Space (VAE) 缩放因子**：
   在 Latent Diffusion (如 SD1.5/SD2.1) 中，VAE 编码的潜变量方差不等于 1。必须乘以缩放系数（SD 中为 `0.18215`），使方差归一化到 $\approx 1.0$，否则高斯先验失效导致生成崩塌。
2. **加噪调度衰减陷阱 (Cosine vs Linear)**：
   DDPM 原始 Linear Schedule 在高分辨率图像上过早破坏低频结构，必须切换为 Improved Cosine Schedule 或 Scaled Linear Schedule。

---

## 模块 5：自测思考题 (Self-Assessment)

- **Q1: 为什么在 $t=0$ 的最后一步逆向采样不能添加噪声？**
  * *解析*：$t=0$ 时 $\bar{\alpha}_{-1} = 1.0$，理论后验方差 $\tilde{\beta}_0 = 0$。注入噪声会污染生成的清晰像素。
- **Q2: 为什么 DDPM 采样必须走完 1000 步？**
  * *解析*：正向与逆向过程严格依赖相邻时间步 $x_t \to x_{t-1}$ 的高斯马尔可夫链假设，中途跳步会导致高斯近似失效。这直接催生了 DDIM 的非马尔可夫突破。
