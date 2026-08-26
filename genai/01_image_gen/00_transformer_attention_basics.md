# 专题 00：Transformer 与注意力机制基石 (Attention & Multi-Head Attention)
> **技术定位**: 现代大语言模型与视觉生成 DiT 的算力心脏  
> **核心概念**: 缩放点积注意力、软字典检索、多头注意力 (MHA)、交叉注意力 (Cross-Attention)、FlashAttention 硬件 IO 原理。

---

## 模块 1：核心动机与注意力直觉 (The "Why")

### 1.1 软字典检索 (Soft Key-Value Lookup)
- **Query ($Q$)**: “我当前位置需要检索什么特征？”
- **Key ($K$)**: “我是其他位置，我具有什么特征索引标签？”
- **Value ($V$)**: “若匹配成功，我能提供什么高维语义内容？”

### 1.2 对比 CNN 与 RNN
- **CNN**: 局部感受野受限，需堆叠深层网络才能获得全局视野。
- **RNN**: 严格时间序列串行，长程梯度消失，无法并行利用现代 GPU 算力。
- **Attention**: 任意两个 Token 之间的直接通信距离为 $\mathcal{O}(1)$，具有天然的超强并行性。

---

## 模块 2：严谨数学公式拆解 (KaTeX Math)

### 2.1 缩放点积注意力 (Scaled Dot-Product Attention)
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

#### 面试高频必问：为什么必须除以 $\sqrt{d_k}$？
若 $q$ 和 $k$ 的各分量独立同分布，均值为 0，方差为 1：
$$\text{Var}\left( \sum_{i=1}^{d_k} q_i k_i \right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$
当 $d_k$ 很大时，点积数值绝对值极大，导致 Softmax 进入两端导数几乎为 0 的平坦饱和区（梯度消失）。除以 $\sqrt{d_k}$ 将方差归一化为 1，确保梯度健康流动。

### 2.2 多头注意力 (Multi-Head Attention, MHA)
$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{where} \quad \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

### 2.3 跨模态交叉注意力 (Cross-Attention)
在文生图模型中：
- $Q = x_{\text{img}} W^Q$（来自待生成的图像或潜码 Token）
- $K = c_{\text{text}} W^K, \ V = c_{\text{text}} W^V$（来自文本提示词 CLIP/T5 嵌入）

---

## 模块 3：PyTorch 极简参考代码

```python
import math
import torch
import torch.nn as nn
from typing import Optional

class MultiHeadCrossAttention(nn.Module):
    """
    自包含的工业级多头自注意力 / 交叉注意力模块 (PyTorch)
    """
    def __init__(self, d_model: int, num_heads: int, d_context: Optional[int] = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        d_ctx = d_context if d_context is not None else d_model
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_ctx, d_model, bias=False)
        self.w_v = nn.Linear(d_ctx, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Seq_Q, _ = x.shape
        ctx = context if context is not None else x
        Seq_KV = ctx.shape[1]

        # 投影并分头 [B, Seq, d_model] -> [B, Num_Heads, Seq, d_k]
        Q = self.w_q(x).view(B, Seq_Q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(ctx).view(B, Seq_KV, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(ctx).view(B, Seq_KV, self.num_heads, self.d_k).transpose(1, 2)

        # 缩放点积
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, Seq_Q, self.d_model)
        return self.w_o(out)
```

---

## 模块 4：FlashAttention 硬件 IO 加速原理

1. **Tiling（分块计算）**：将 $Q, K, V$ 按照 GPU 高速片上 SRAM 大小切成小块（如 $128 \times 128$），在 SRAM 内部完成局部矩阵乘法。
2. **Online Softmax（在线 Softmax 统计量递推）**：动态维护局部最大值 $m(x)$ 和指数和 $l(x)$，**完全无需在显存 HBM 中保存 $N \times N$ 的完整注意力矩阵**，显存降为 $\mathcal{O}(N)$，速度提升 2~4 倍。
