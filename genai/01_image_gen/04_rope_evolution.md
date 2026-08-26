# 专题 04：位置编码全景大一统 (1D RoPE ➔ 2D/3D RoPE ➔ Qwen M-RoPE)
> **技术定位**: 视觉大模型与生成 DiT 的空间罗盘  
> **核心概念**: 1D RoPE (RoFormer/LLaMA)、2D Axial RoPE (FLUX.1/SD3)、3D 时空 RoPE (Wan 2.1/Sora)、Qwen M-RoPE (Qwen2-VL/Qwen-Image 多模态三元组)。

---

## 模块 1：核心动机与物理直觉 (The "Why")

### 1.1 为什么绝对位置编码在生成模型中受限？
1. **无法外推可变分辨率**：训练分辨率 $512 \times 512$，推理 $1024 \times 1024$ 时绝对序号越界。
2. **破坏相对平移不变性**：物体在画面中平移后，绝对坐标值改变导致注意力内积失真。
3. **图文多模态无法自然对齐**：1D 文本序列与 2D/3D 视觉网格在拼接处产生位置冲突。

### 1.2 RoPE 的几何本质
通过 2D 块对角旋转矩阵 $\mathbf{R}_m$ 旋转 Query 与 Key，内积 $\langle \mathbf{R}_m \mathbf{q}, \mathbf{R}_n \mathbf{k} \rangle$ 严格取决于**相对位置差 $(m - n)$**。

---

## 模块 2：四大 RoPE 演进推导与独立代码实现 (带详细逐行注释)

### 2.1 1D RoPE (标准文本序列 · RoFormer / LLaMA)
$$\mathbf{R}_{\Theta, m}^{d} = \text{diag}\big( \mathbf{R}(\theta_1 m), \dots, \mathbf{R}(\theta_{d/2} m) \big), \quad \theta_i = 10000^{-2(i-1)/d}$$

```python
import torch
import torch.nn as nn

def apply_1d_rope(x: torch.Tensor, pos_ids: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    1D 旋转位置编码 (RoPE)
    - x: [Batch, Heads, Seq_Len, Dim] 输入的 Q 或 K 向量
    - pos_ids: [Batch, Seq_Len] 每个 Token 的 1D 标量位置 (例如 [0, 1, 2, ...])
    """
    B, H, S, D = x.shape
    assert D % 2 == 0, "特征维度 Dim 必须为偶数"

    # 1. 计算每个复数平面的旋转角频率 theta_i = 1.0 / (base ** (2i / D))
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=x.device).float() / D)) # [D // 2]
    
    # 2. 计算每个位置的实际旋转弧度 m * theta_i: [B, S] 外积 [D // 2] -> [B, S, D // 2]
    angles = torch.einsum('bs, d -> bsd', pos_ids.float(), inv_freq)
    
    # 3. 复制两份构成完整的 [B, 1, S, D] 旋转系数矩阵
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).unsqueeze(1) # [B, 1, S, D]
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).unsqueeze(1) # [B, 1, S, D]

    # 4. 执行复数旋转变换: [x1, x2] 变为 [-x2, x1]
    x1, x2 = x[..., :D//2], x[..., D//2:]
    rotate_half = torch.cat([-x2, x1], dim=-1)
    
    # 5. 旋转公式: x_rot = x * cos + rotate_half * sin
    return x * cos + rotate_half * sin
```

---

### 2.2 2D Axial RoPE (图像 DiT · FLUX.1 / SD3)
特征通道 $D$ 对半拆分：前 $D/2$ 编码高度 $h$，后 $D/2$ 编码宽度 $w$：
$$\mathbf{R}_{2D}(h, w) = \begin{pmatrix} \mathbf{R}_{\Theta_h, h}^{D/2} & \mathbf{0} \\ \mathbf{0} & \mathbf{R}_{\Theta_w, w}^{D/2} \end{pmatrix}$$

```python
def apply_2d_axial_rope(x: torch.Tensor, H_patches: int, W_patches: int, base: float = 10000.0) -> torch.Tensor:
    """
    2D 轴向旋转位置编码 (图像 DiT 专用)
    - x: [Batch, Heads, Num_Patches, Dim] 其中 Num_Patches = H_patches * W_patches
    """
    B, Heads, S, D = x.shape
    dim_h, dim_w = D // 2, D // 2

    # 1. 生成 2D 真实空间网格坐标
    grid_h, grid_w = torch.meshgrid(
        torch.arange(H_patches, device=x.device),
        torch.arange(W_patches, device=x.device),
        indexing='ij'
    )
    pos_h = grid_h.flatten().unsqueeze(0).repeat(B, 1) # [B, S]
    pos_w = grid_w.flatten().unsqueeze(0).repeat(B, 1) # [B, S]

    # 2. 将 x 沿通道维度切分为两半: x_h 负责编码高度，x_w 负责编码宽度
    x_h = x[..., :dim_h]
    x_w = x[..., dim_h:]

    # 3. 分别应用 1D RoPE 旋转
    x_h_rot = apply_1d_rope(x_h, pos_h, base=base)
    x_w_rot = apply_1d_rope(x_w, pos_w, base=base)

    # 4. 重新沿通道拼接回完整的 Dim 向量
    return torch.cat([x_h_rot, x_w_rot], dim=-1)
```

---

### 2.3 3D Spatiotemporal RoPE (视频 DiT · Wan 2.1 / Sora)
特征通道三等分：分别编码时间帧 $t$、高度 $h$、宽度 $w$：
$$\mathbf{R}_{3D}(t, h, w) = \text{diag}\left( \mathbf{R}_{\Theta_t, t}^{D_t}, \ \mathbf{R}_{\Theta_h, h}^{D_h}, \ \mathbf{R}_{\Theta_w, w}^{D_w} \right)$$

```python
def apply_3d_spatiotemporal_rope(x: torch.Tensor, T_frames: int, H_patches: int, W_patches: int, base: float = 10000.0) -> torch.Tensor:
    """
    3D 时空旋转位置编码 (视频 DiT 专用)
    - x: [Batch, Heads, Num_Tokens, Dim] 其中 Num_Tokens = T_frames * H_patches * W_patches
    """
    B, Heads, S, D = x.shape
    dim_t, dim_h, dim_w = 16, 24, 24 # 假定 D=64
    assert dim_t + dim_h + dim_w == D

    # 1. 生成 3D 时空立方体网格坐标 (T, H, W)
    grid_t, grid_h, grid_w = torch.meshgrid(
        torch.arange(T_frames, device=x.device),
        torch.arange(H_patches, device=x.device),
        torch.arange(W_patches, device=x.device),
        indexing='ij'
    )
    pos_t = grid_t.flatten().unsqueeze(0).repeat(B, 1) # [B, S]
    pos_h = grid_h.flatten().unsqueeze(0).repeat(B, 1) # [B, S]
    pos_w = grid_w.flatten().unsqueeze(0).repeat(B, 1) # [B, S]

    # 2. 沿通道切分为三段
    x_t, x_h, x_w = torch.split(x, [dim_t, dim_h, dim_w], dim=-1)

    # 3. 分别施加 RoPE 并拼接
    return torch.cat([
        apply_1d_rope(x_t, pos_t, base=base),
        apply_1d_rope(x_h, pos_h, base=base),
        apply_1d_rope(x_w, pos_w, base=base)
    ], dim=-1)
```

---

## 模块 3：Qwen 独创 M-RoPE (Multimodal RoPE) 机制彻底拆解

### 3.1 统一三元组位置分配算法 $[pos_{\text{temporal}}, pos_{\text{height}}, pos_{\text{width}}]$
1. **纯文本 Token**: $[i, i, i]$（三轴共享相同序号 $i$，等价于标准 1D RoPE）。
2. **图像视觉 Token**: $[t_{\text{img}}, h, w]$（时间固定，空间展开为二维真实网格）。
3. **视频视觉 Token**: $[t_k, h, w]$（时间递增为帧序号 $t_k$，空间展开为二维网格）。

```python
class QwenMRoPE(nn.Module):
    """
    Qwen2-VL / Qwen-Image 官方级 M-RoPE 核心实现
    """
    def __init__(self, head_dim: int, mrope_section: tuple = (16, 24, 24), base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.mrope_section = mrope_section
        assert sum(mrope_section) == head_dim

        self.freqs = nn.ParameterList([
            nn.Parameter(1.0 / (base ** (torch.arange(0, s, 2).float() / s)), requires_grad=False)
            for s in mrope_section
        ])

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor) -> tuple:
        """
        - pos_ids: [3, Batch, Total_Seq_Len] 包含 (Time_ID, Height_ID, Width_ID)
        """
        q_splits = torch.split(q, self.mrope_section, dim=-1)
        k_splits = torch.split(k, self.mrope_section, dim=-1)
        q_rot_list, k_rot_list = [], []
        
        for i in range(3):
            axis_pos = pos_ids[i] # [Batch, Seq_Len]
            inv_freq = self.freqs[i]
            
            angles = torch.einsum('bs, d -> bsd', axis_pos.float(), inv_freq)
            sin = torch.cat([angles.sin(), angles.sin()], dim=-1).unsqueeze(1)
            cos = torch.cat([angles.cos(), angles.cos()], dim=-1).unsqueeze(1)

            q_i, k_i = q_splits[i], k_splits[i]
            d_i = q_i.shape[-1]
            q_rot_half = torch.cat([-q_i[..., d_i//2:], q_i[..., :d_i//2]], dim=-1)
            k_rot_half = torch.cat([-k_i[..., d_i//2:], k_i[..., :d_i//2]], dim=-1)

            q_rot_list.append(q_i * cos + q_rot_half * sin)
            k_rot_list.append(k_i * cos + k_rot_half * sin)

        return torch.cat(q_rot_list, dim=-1), torch.cat(k_rot_list, dim=-1)
```

---

## 模块 4：四大 RoPE 横向对比矩阵

| 编码类型 | 坐标形式 | 通道切分方式 | 代表模型 | 核心优势 |
| :--- | :--- | :--- | :--- | :--- |
| **1D RoPE** | $[m]$ (标量) | 全量旋转 | LLaMA, Mistral | 相对位置线性衰减，外推性强 |
| **2D Axial RoPE** | $[h, w]$ (二维) | 对半切分 $(D/2, D/2)$ | FLUX.1, SD3 | 解耦高宽，天然支持任意长宽比生成 |
| **3D Spatiotemporal** | $[t, h, w]$ (三维) | 三等分 $(D_t, D_h, D_w)$ | Wan 2.1, Sora | 兼顾帧间运动轨迹与单帧内部构图 |
| **Qwen M-RoPE** | $[t, h, w]$ (三元组) | 多模态自适应分配 | Qwen2-VL, Qwen-Image | 彻底统一图文交错位置表，原生动态分辨率 |
