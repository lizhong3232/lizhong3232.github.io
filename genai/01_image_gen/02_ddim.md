## 模块 1：核心解惑 —— DDIM 与 DDPM 的真正区别 & 什么是 ODE？

### 1.1 直击痛点：为什么它俩看起来一模一样？
初学者最容易产生的困惑：
- **前向加噪公式一样**：$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$
- **网络预测目标一样**：还是拟合注入的高斯噪声 $\epsilon_\theta(x_t, t)$
- **训练损失一样**：$\mathcal{L}_{\text{simple}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$

> **震撼事实**：DDIM **根本不需要重新训练模型**！任何已经用标准 DDPM 训练好的权重（如 Stable Diffusion 1.5 的 UNet），可以直接无缝切换为 DDIM 采样器！**它们在训练端 100% 相同，核心分水岭完全发生在「推理采样阶段的概率建模方式」！**

---

### 1.2 三大本质区别对照表 (The 3 Real Differences)

| 对比维度 | DDPM (随机马尔可夫 SDE 视角) | DDIM (确定性非马尔可夫 ODE 视角) |
| :--- | :--- | :--- |
| **概率图模型假设** | **严格马尔可夫链**：$x_t$ 只依赖上一时刻 $x_{t-1}$，与 $x_0$ 无直接条件约束。 | **非马尔可夫联合分布**：$x_t$ 显式以 $x_0$ 和 $x_{t-1}$ 为条件，释放了后验方差的自由度。 |
| **反向采样过程** | **随机扩散 (SDE)**：每一步都显式抽取并注入新的高斯随机噪声 $z \sim \mathcal{N}(0, I)$。 | **确定性积分 (ODE)**：令随机项 $\sigma_t = 0$，逆向推演过程**零随机噪声注入**！ |
| **为什么可以跨步？** | **不可跳步 (必须1000步)**：步长大时，局部无穷小的高斯微扰假设破裂，后验推导全盘失效。 | **自由跳步 (20~50步)**：$\sigma_t=0$ 时退化为几何投影递推式，在任意离散步长下数学形式均严格自洽。 |
| **潜空间可逆性** | **不可逆 (单向)**：同一张初始噪声每次生成不同图片；真实图片无法反演回唯一确定潜码。 | **严格双向可逆 (Inversion)**：$x_0 \leftrightarrow x_T$ 形成确定性一一映射，支持精准图像编辑。 |

---

### 1.3 灵魂追问：DDPM 代码里到底哪里写了必须 1000 步？不也是能一步算出 $x_0$ 吗？

#### 1. 厘清两大阶段的致命混淆（前向加噪 1 步 vs 逆向生成 1000 步）
- **训练时 (前向加噪)**：确实只需 **1 步**！利用重参数化公式 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$，给真实图像 $x_0$ 和随机时间 $t$，直接闭式解计算，零循环耗时。
- **生成时 (逆向采样)**：代码里明确写死了 **1000 步死循环**！
  ```python
  for t in reversed(range(self.timesteps)):  # self.timesteps = 1000 !
      eps = model(x, t)  # 跑 1000 遍 UNet 前向推理!
      x = (1 / sqrt(alpha_t)) * (x - (beta_t / sqrt(1 - a_bar_t)) * eps) + sigma_t * torch.randn_like(x)
  ```
  生成时必须从纯噪声 $x_{1000}$ 开始，**逐一循环调用 1000 次模型推理**！

#### 2. 既然每步都能推算 $\hat{x}_0$，为什么在 $t=1000$ 时不能直接“一步到位”？
如果我们强行把代码改成单步推理：在纯白噪声 $x_{1000}$ 下让模型预测一次 $\epsilon$，然后直接代入公式计算 $\hat{x}_0 = \frac{x_{1000} - \sqrt{1-\bar{\alpha}_{1000}}\epsilon}{\sqrt{\bar{\alpha}_{1000}}}$ 会发生什么？
- **结果：完全是一团灰蒙蒙、没有任何结构的平庸平均色块（如同打翻的调色盘）！**
- **数学原因（条件期望均值模糊）**：在 $t=1000$ 纯噪声下，信噪比 $\text{SNR}=0$。对于输入的一团白噪声，它可能对应训练集中 1 亿张图片里的任何一张（猫、狗、人脸、跑车）。在均方误差 MSE 目标下，无法确定具体类别的神经网络唯一的数学最优解就是**输出所有可能图片的“像素平均值”**，导致所有锐利细节被彻底抹杀！

#### 3. 形象比喻：米开朗基罗雕刻大理石
- **一步到位的妄想**：面对一整块未经雕琢的粗糙大理石（$x_{1000}$），期望抡起大铁锤**一锤子**砸下去直接成型完美雕像。这在物理上必会导致大理石直接粉碎。
- **DDPM 的 1000 步**：由于每步有随机微扰假设，必须小心翼翼敲 **1000 小锤**（第 1~200 锤凿大体轮廓，第 201~700 锤刻五官肌肉，第 701~1000 锤用细砂纸打磨毛孔细节）。
- **DDIM 的突破**：发现确定性流场规律，把弯曲的微小跳跃重构为平滑 ODE，大步流星只需 **20~50 锤**！
- **Flow Matching (SD3/Flux)**：拉出直线最优传输，结合 Reflow 直化，甚至只要 **1~2 锤**即可真正实现单步高保真出图！

---

### 1.4 深入浅出：到底什么是常微分方程 (ODE)？

#### 1. 生活化比喻：河水流速与纸船轨迹
想象一条平缓流动的大河，河面上每一个坐标点 $(x, y)$ 在任意时刻 $t$ 都有一个固定的水流速度和流向 $\vec{v}(x, y, t)$（这就是**速度向量场**）。  
如果你在河面坐标 $x(0)$ 放下一只纸船，纸船在接下来每一秒的运动轨迹，被水流速度**唯一且完全确定**：
$$\frac{d x(t)}{dt} = v\big(x(t), t\big)$$
这个描述“瞬时位置变化率与当前状态关系”的方程，就叫做**常微分方程 (Ordinary Differential Equation, ODE)**！

#### 2. ODE vs SDE 的核心区别
- **确定性 ODE** ($\frac{dx}{dt} = f(x, t)$)：就像平缓水流中的纸船。只要起点 $x(0)$ 固定，未来的轨迹**唯一确定、无分叉、可完美沿原路倒流**（时间反演可逆）。
- **随机 SDE** ($dx = f(x, t)dt + g(t)dw_t$)：就像狂风暴雨中的纸船。除了水流，每一步都被随机狂风（布朗运动 $dw_t$）踢上一脚，**轨迹充满随机抖动且不可逆**。

#### 3. DDIM 怎么就变成 ODE 了？
当 DDIM 令采样噪声项 $\sigma_t = 0$ 时，令连续时间步长 $\Delta t \to 0$，它的单步更新公式严格收敛为一个连续概率流 ODE：
$$\frac{dx}{dt} = -\frac{1}{2} \beta(t) \left[ x + \frac{\epsilon_\theta(x, t)}{\sqrt{1 - \bar{\alpha}_t}} \right]$$
在 ODE 视角下，**神经网络预测的噪声 $\epsilon_\theta$，本质上就是在计算潜空间在时刻 $t$ 这一处的确定性水流流速！**

---

## 模块 2：严谨数学推导拆解 (Mathematical Derivations)

### 2.1 非马尔可夫联合分布构造
Song et al. 证明：扩散模型的训练目标 $\mathcal{L}_{\text{simple}}$ 仅由边缘分布 $q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$ 决定，而联合分布 $q(x_{1:T} | x_0)$ 不必是马尔可夫链！

构造由参数 $\sigma_t$ 自由控制的一族非马尔可夫推断后验：
$$q_\sigma(x_{t-1} | x_t, x_0) = \mathcal{N}\left( x_{t-1}; \ \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1 - \bar{\alpha}_t}}, \ \sigma_t^2 I \right)$$

---

### 2.2 $\sigma_t=0$ 退化为确定性概率流 ODE
当令随机项方差 $\sigma_t = 0$ 时，采样公式完全消除了随机正态噪声：
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{预测的初始图像 } \hat{x}_0} + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

在连续时间极限下，该差分方程严格收敛为连续常微分方程（Probability Flow ODE）：
$$\frac{dx}{dt} = f(x, t) = -\frac{1}{2} \beta(t) \left[ x + \frac{\epsilon_\theta(x, t)}{\sqrt{1 - \bar{\alpha}_t}} \right]$$

---

### 2.3 精确潜空间反演 (Exact Latent Inversion)
由于 ODE 满足时间可逆性：
$$\text{正向反演}: x_{t+1} = \sqrt{\bar{\alpha}_{t+1}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t+1}} \epsilon_\theta(x_t, t)$$
$$\text{反向采样}: x_t = \sqrt{\bar{\alpha}_t} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_{t+1}, t+1)$$

---

## 模块 3：核心算法代码实现与 DDPM vs DDIM 核心差异精讲 (Code Walkthrough)

### 3.1 代码级显微镜：DDPM vs DDIM 采样核心循环对比

```python
# -----------------------------
# 1. DDPM 采样核心循环 (必须 1000 步, 随机 SDE)
# -----------------------------
for t in reversed(range(1000)):
    eps = model(x, t)  # 预测噪声
    
    # 计算后验均值
    mean = (1.0 / sqrt(alpha[t])) * (x - (beta[t] / sqrt(1 - a_bar[t])) * eps)
    
    # ⚠️ 核心区别: 每一步必须注入高斯随机噪声 z !
    noise = torch.randn_like(x) if t > 0 else 0.0
    x = mean + sqrt(beta[t]) * noise


# -----------------------------
# 2. DDIM 采样核心循环 (自由跨步例如 20 步, 确定性 ODE)
# -----------------------------
for t_cur, t_prev in zip(timesteps[:-1], timesteps[1:]):  # 如 [999, 949, ..., 0] 仅 20 步!
    eps = model(x, t_cur)  # 预测噪声 (与 DDPM 完全相同)
    
    # 显式估计当前底图 x_0
    pred_x0 = (x - sqrt(1 - a_cur) * eps) / sqrt(a_cur)
    
    # ✨ 核心区别: eta=0 时零噪声注入, 纯几何投影直接指向 t_prev!
    dir_xt = sqrt(1 - a_prev) * eps
    x = sqrt(a_prev) * pred_x0 + dir_xt  # 纯确定性更新
```

#### 代码层面 4 大核心关键差异点解析：
1. **时间循环索引解耦**：DDPM 硬编码 `reversed(range(1000))`；DDIM 引入 `timesteps = linspace(999, 0, 20)`，可任意配置 10、20、50 步无需重新训练。
2. **彻底抛弃随机噪声注入**：DDPM 有 `+ sqrt(beta) * randn_like(x)`；DDIM 在确定性模式下令 `sigma_t = 0`，随机噪声直接消失为 0！
3. **预测量与状态中介**：DDPM 计算局部的后验均值 `mean`；DDIM 先显式解析出底图 `pred_x0`，再投影到目标时刻 `t_prev`，数学上几何意义更清晰。
4. **权重与前向完全复用**：`model(x, t)` 的输入输出完全一致，因此不需要额外训练新的 Checkpoint，直接替换采样器即可。

---

### 3.2 完整自包含 DDIM 采样器与 Inversion 类实现
    """
    工业级 DDIM 确定性采样与跨步加速管线 (PyTorch)
    """
    def __init__(self, model: nn.Module, alphas_cumprod: torch.Tensor):
        self.model = model
        self.alphas_cumprod = alphas_cumprod  # [1000]

    @torch.no_grad()
    def sample(self, shape: tuple, num_steps: int = 20, eta: float = 0.0) -> torch.Tensor:
        device = next(self.model.parameters()).device
        total_timesteps = len(self.alphas_cumprod)
        
        # 1. 均匀采样时间子序列 (如 [999, 949, 899, ..., 0])
        timesteps = torch.linspace(total_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)
        x = torch.randn(shape, device=device)

        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_prev = timesteps[i + 1]

            a_bar_cur = self.alphas_cumprod[t_cur]
            a_bar_prev = self.alphas_cumprod[t_prev]

            # 2. 模型预测噪声 epsilon
            t_batch = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            eps_pred = self.model(x, t_batch)

            # 3. 估计初始图像 x_0
            pred_x0 = (x - torch.sqrt(1.0 - a_bar_cur) * eps_pred) / torch.sqrt(a_bar_cur)

            # 4. 计算随机方差系数 sigma_t (eta=0 时为 0)
            sigma_t = eta * torch.sqrt((1.0 - a_bar_prev) / (1.0 - a_bar_cur) * (1.0 - a_bar_cur / a_bar_prev))

            # 5. 确定性方向更新
            dir_xt = torch.sqrt(torch.clamp(1.0 - a_bar_prev - sigma_t ** 2, min=0.0)) * eps_pred
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(a_bar_prev) * pred_x0 + dir_xt + sigma_t * noise

        return x

    @torch.no_grad()
    def invert(self, x_0: torch.Tensor, num_steps: int = 20) -> torch.Tensor:
        """DDIM 正向反演 (Exact Inversion): x_0 -> x_T"""
        device = x_0.device
        total_timesteps = len(self.alphas_cumprod)
        timesteps = torch.linspace(0, total_timesteps - 1, num_steps, dtype=torch.long, device=device)
        
        x = x_0.clone()
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]

            a_bar_cur = self.alphas_cumprod[t_cur]
            a_bar_next = self.alphas_cumprod[t_next]

            t_batch = torch.full((x.shape[0],), t_cur, device=device, dtype=torch.long)
            eps_pred = self.model(x, t_batch)

            pred_x0 = (x - torch.sqrt(1.0 - a_bar_cur) * eps_pred) / torch.sqrt(a_bar_cur)
            x = torch.sqrt(a_bar_next) * pred_x0 + torch.sqrt(1.0 - a_bar_next) * eps_pred
        return x
```

---

## 模块 4：工业界落地要点与核心踩坑点 (Engineering Pitfalls)

1. **CFG 下 Inversion 累积漂移**：
   当使用大 CFG 进行 DDIM Inversion 时，局部切线斜率不重合，正反积分会出现累积漂移。必须使用 **Null-text Inversion** 优化无条件文本嵌入向量以消除漂移。
2. **反演时切勿截断 (No Clipping in Inversion)**：
   在 Inversion 过程中强行 clamp 潜变量会破坏 ODE 光滑性，导致反演图像严重失真。

---

## 模块 5：自测思考题 (Self-Assessment)

- **Q1: 为什么使用相同的权重，DDIM 仅需 20 步而 DDPM 却需要 1000 步？**
  * *解析*：DDIM 构造的非马尔可夫概率流 ODE 在任意离散时间子序列下形式自洽，不需要每步满足无穷小高斯扰动假设。
- **Q2: 为什么 DDIM 依然难以压缩到 5 步以内？**
  * *解析*：受限于方差保持调度，潜空间采样轨迹在超球面上弯曲。大步长欧拉法在曲率大的地方截断误差急剧增加。
