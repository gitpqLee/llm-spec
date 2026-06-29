# MHA / GQA / MQA 详解及 KV-Cache 差异

> 从 Multi-Head Attention 到 Multi-Query Attention，再到 Grouped-Query Attention —— 理解它们的演进动机、结构差异，以及对 KV-Cache 显存/带宽的深刻影响。

---

## 一、前置概念回顾

在 Self-Attention 中，输入 $X \in \mathbb{R}^{B \times T \times D}$ 经过三个线性投影得到：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

然后切分为多个 head 并行计算：

$$\text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_h}}\right) V_i$$

**关键问题**：推理时，为了避免重复计算历史 token 的 K/V，我们把它们缓存下来 —— 这就是 **KV-Cache**。KV-Cache 的大小直接正比于 K/V head 的数量。

---

## 二、三种注意力机制对比

### 2.1 MHA — Multi-Head Attention

> 经典做法，每个 Query head 都有自己专属的 K head 和 V head。

```
         Q heads          K heads          V heads
        ┌──────┐         ┌──────┐         ┌──────┐
        │ Q_0  │ ──────▶ │ K_0  │         │ V_0  │
        │ Q_1  │ ──────▶ │ K_1  │         │ V_1  │
        │ Q_2  │ ──────▶ │ K_2  │         │ V_2  │
        │ ...  │         │ ...  │         │ ...  │
        │ Q_h  │ ──────▶ │ K_h  │         │ V_h  │
        └──────┘         └──────┘         └──────┘
           h 个              h 个             h 个
```

**参数**：
- $W_Q \in \mathbb{R}^{D \times D}$，$W_K \in \mathbb{R}^{D \times D}$，$W_V \in \mathbb{R}^{D \times D}$
- 每个 head 维度 $d_h = D / h$

**KV-Cache 大小**（per layer, per token）：
$$\text{KV-Cache} = 2 \times h \times d_h = 2D$$

**典型模型**：GPT-2, GPT-3, BERT, 早期 LLaMA

---

### 2.2 MQA — Multi-Query Attention

> **所有 Q head 共享同一组 K/V**。极端压缩 KV-Cache。

```
         Q heads          K heads          V heads
        ┌──────┐         ┌──────┐         ┌──────┐
        │ Q_0  │ ──┐     │      │         │      │
        │ Q_1  │ ──┤     │ K_0  │         │ V_0  │  ← 只有 1 个!
        │ Q_2  │ ──┤     │      │         │      │
        │ ...  │   │     └──────┘         └──────┘
        │ Q_h  │ ──┘       1 个              1 个
        └──────┘
           h 个
```

**参数**：
- $W_Q \in \mathbb{R}^{D \times D}$（不变）
- $W_K \in \mathbb{R}^{D \times d_h}$，$W_V \in \mathbb{R}^{D \times d_h}$（大幅缩小）

**KV-Cache 大小**（per layer, per token）：
$$\text{KV-Cache} = 2 \times 1 \times d_h = 2d_h = \frac{2D}{h}$$

相比 MHA，KV-Cache **减少了 $h$ 倍**！

**典型模型**：PaLM, Falcon-40B, StarCoder

**优点**：显存大幅降低，推理速度快（尤其是长序列 / 大 batch）

**缺点**：所有 Q head 看到的 K/V 完全相同，表达力下降，训练质量可能受损

---

### 2.3 GQA — Grouped-Query Attention

> MHA 和 MQA 的折中方案。将 Q head 分组，每组共享一个 K/V head。

```
         Q heads          K heads          V heads
        ┌──────┐         ┌──────┐         ┌──────┐
        │ Q_0  │ ──┐     │      │         │      │
        │ Q_1  │ ──┤───▶ │ K_0  │         │ V_0  │
        ├──────┤   │     ├──────┤         ├──────┤
        │ Q_2  │ ──┘     │      │         │      │
        │ Q_3  │ ──┐     │ K_1  │         │ V_1  │
        ├──────┤   │     ├──────┤         ├──────┤
        │ Q_4  │ ──┘     │      │         │      │
        │ Q_5  │ ──┐     │ K_2  │         │ V_2  │
        ├──────┤   │     ├──────┤         ├──────┤
        │ Q_6  │ ──┘     │      │         │      │
        │ Q_7  │ ──┐     │ K_g  │         │ V_g  │
        └──────┘   │     └──────┘         └──────┘
          h=8 个   └───▶    g=4 个           g=4 个
                          (每组2个Q共享)
```

**参数**：
- $W_Q \in \mathbb{R}^{D \times D}$（不变）
- $W_K \in \mathbb{R}^{D \times (g \cdot d_h)}$，$W_V \in \mathbb{R}^{D \times (g \cdot d_h)}$
- 其中 $g$ = KV head 数量，每 $h/g$ 个 Q head 共享一个 KV head

**KV-Cache 大小**（per layer, per token）：
$$\text{KV-Cache} = 2 \times g \times d_h = \frac{2gD}{h}$$

**典型模型**：LLaMA-2 70B (g=8, h=64), LLaMA-3, Gemma-2, Qwen-2, Mistral

---

## 三、一张表看全貌

| 属性 | MHA | GQA | MQA |
|------|-----|-----|-----|
| Q head 数 | $h$ | $h$ | $h$ |
| KV head 数 | $h$ | $g$（$1 < g < h$）| $1$ |
| 每组共享比 | 1:1 | $h/g$ : 1 | $h$ : 1 |
| KV-Cache / layer / token | $2D$ | $2gd_h$ | $2d_h$ |
| KV-Cache 相对 MHA | 1× | $g/h$ × | $1/h$ × |
| 模型质量 | 最佳 | 接近 MHA | 略有下降 |
| 推理吞吐 | 基线 | 显著提升 | 最大提升 |

---

## 四、数值示例

以 **LLaMA-2 70B** 为例：
- $D = 8192$, $h = 64$（Q head 数）, $g = 8$（KV head 数）, $d_h = 128$
- Layers = 80

**每 token 的 KV-Cache（FP16）：**

| 方案 | 公式 | 每层 | 80 层总计 |
|------|------|------|-----------|
| 若是 MHA | $2 \times 64 \times 128 \times 2\text{B}$ | 32 KB | 2.5 MB |
| 实际 GQA | $2 \times 8 \times 128 \times 2\text{B}$ | 4 KB | 320 KB |
| 若是 MQA | $2 \times 1 \times 128 \times 2\text{B}$ | 512 B | 40 KB |

对于 **4096 token 序列**：
- MHA: 4096 × 2.5MB = **10 GB**
- GQA: 4096 × 320KB = **1.25 GB** （节省 **8×**）
- MQA: 4096 × 40KB = **160 MB** （节省 **64×**）

---

## 五、KV-Cache 处理的工程差异

### 5.1 MHA 的 KV-Cache

```python
# 每层维护完整的 K/V cache
# cache shape: [batch, num_heads, seq_len, head_dim]
#              [B,     h,          T,       d_h]

k_cache = torch.zeros(B, h, max_seq, d_h)   # h 个 head, 每个都独立
v_cache = torch.zeros(B, h, max_seq, d_h)

# decode 阶段: 追加新 token 的 kv
k_cache[:, :, pos, :] = k_new    # shape [B, h, 1, d_h] → 写入 pos 位置
v_cache[:, :, pos, :] = v_new

# attention: 每个 Q head 和对应的 K/V head 计算
for i in range(h):
    attn_i = softmax(q[:, i] @ k_cache[:, i].T / sqrt(d_h)) @ v_cache[:, i]
```

### 5.2 MQA 的 KV-Cache

```python
# 只缓存 1 个 KV head
# cache shape: [batch, 1, seq_len, head_dim]

k_cache = torch.zeros(B, 1, max_seq, d_h)   # 只有 1 份!
v_cache = torch.zeros(B, 1, max_seq, d_h)

# decode 阶段
k_cache[:, 0, pos, :] = k_new
v_cache[:, 0, pos, :] = v_new

# attention: 所有 Q head 共享同一份 K/V
# 实现上通过 broadcast 或 expand
k_expanded = k_cache.expand(B, h, max_seq, d_h)  # 逻辑复制, 不占显存
v_expanded = v_cache.expand(B, h, max_seq, d_h)

for i in range(h):
    attn_i = softmax(q[:, i] @ k_expanded[:, i].T / sqrt(d_h)) @ v_expanded[:, i]
```

### 5.3 GQA 的 KV-Cache

```python
# 缓存 g 个 KV head
# cache shape: [batch, num_kv_heads, seq_len, head_dim]

k_cache = torch.zeros(B, g, max_seq, d_h)   # g 份
v_cache = torch.zeros(B, g, max_seq, d_h)

# decode 阶段
k_cache[:, :, pos, :] = k_new    # shape [B, g, 1, d_h]
v_cache[:, :, pos, :] = v_new

# attention: 每 (h/g) 个 Q head 共享一个 KV head
# 方法1: repeat_interleave
num_repeats = h // g   # e.g. 64/8 = 8
k_expanded = k_cache.repeat_interleave(num_repeats, dim=1)  # [B, h, T, d_h]
v_expanded = v_cache.repeat_interleave(num_repeats, dim=1)

# 方法2: reshape + expand (更高效, 避免实际拷贝)
k_expanded = k_cache.unsqueeze(2).expand(B, g, num_repeats, T, d_h).reshape(B, h, T, d_h)
```

---

## 六、在编译器 / IR 层面的体现

在 OpenVINO IR 或 ONNX 中，三种方案的核心区别体现在 **KV projection 的输出维度** 和 **SDPA 前的 reshape/broadcast**：

### 6.1 MHA（无 broadcast）

```
Q_proj: [B, T, D] → MatMul(W_q) → [B, T, D] → Reshape → [B, h, T, d_h]
K_proj: [B, T, D] → MatMul(W_k) → [B, T, D] → Reshape → [B, h, T, d_h]
V_proj: [B, T, D] → MatMul(W_v) → [B, T, D] → Reshape → [B, h, T, d_h]

SDPA(Q, K, V)  ← 形状完全匹配, 无需 broadcast
```

### 6.2 GQA（需要 broadcast / repeat）

```
Q_proj: [B, T, D]        → MatMul(W_q) → [B, T, h*d_h]   → Reshape → [B, h, T, d_h]
K_proj: [B, T, D]        → MatMul(W_k) → [B, T, g*d_h]   → Reshape → [B, g, T, d_h]
V_proj: [B, T, D]        → MatMul(W_v) → [B, T, g*d_h]   → Reshape → [B, g, T, d_h]

# 需要将 K/V 从 g heads 扩展到 h heads:
K: [B, g, T, d_h] → Unsqueeze → [B, g, 1, T, d_h]
                   → Expand    → [B, g, h/g, T, d_h]
                   → Reshape   → [B, h, T, d_h]
(V 同理)

SDPA(Q, K_expanded, V_expanded)
```

> **编译器优化点**：这个 Unsqueeze → Expand → Reshape 链在 NPU 编译器中可以 fuse 成一个虚拟的 broadcast，或直接在 SDPA kernel 内部通过 stride 实现，避免真实的数据搬移。

### 6.3 MQA（退化为 g=1 的 GQA）

与 GQA 形式相同，只是 $g = 1$，broadcast 比例为 $h$：1。

---

## 七、为什么 GQA 是当前主流选择？

```
                质量 ←————————————————————————→ 效率
                  |                               |
                 MHA                             MQA
                  |           GQA                 |
                  |         (sweet spot)          |
                  |            ↕                  |
                  |   接近 MHA 质量               |
                  |   接近 MQA 效率               |
```

1. **训练质量**：GQA 在大模型上的 loss 几乎与 MHA 持平（Google 论文实验证明）
2. **推理效率**：KV-Cache 减少 $h/g$ 倍，显著降低显存和 memory bandwidth 压力
3. **灵活性**：$g$ 可以根据部署场景调整（$g=1$ 退化为 MQA，$g=h$ 退化为 MHA）
4. **已训练模型可转换**：Google 展示了从已训练好的 MHA 模型 "uptrain" 到 GQA 的方法

---

## 八、Decode 阶段的瓶颈分析

在 LLM 推理的 decode 阶段（逐 token 生成），瓶颈是 **memory bandwidth**（不是计算）：

```
Decode 每步工作:
  1. 读取模型权重 (大)     → 受限于 memory bandwidth
  2. 读取 KV-Cache (大)    → 受限于 memory bandwidth  ← GQA 在此处优化
  3. 矩阵计算 (小)         → 算力利用率很低 (arithmetic intensity 低)
```

**KV-Cache 读取量**：
- MHA: 需要读 $2 \times h \times T \times d_h$ 元素 — 随序列长度线性增长
- GQA: 只需读 $2 \times g \times T \times d_h$ — **减少 $h/g$ 倍读取**
- MQA: 只需读 $2 \times 1 \times T \times d_h$ — 最小化

这直接转化为：
- **更高的 tokens/second**（decode 速度）
- **更大的 batch size**（同样显存可以服务更多用户）
- **更长的上下文**（KV-Cache 占比降低）

---

## 九、总结

| | MHA | GQA | MQA |
|---|---|---|---|
| 设计思想 | 每个 Q 有专属 KV | Q 分组共享 KV | 所有 Q 共享 KV |
| KV-Cache 节省 | — | $h/g$ × | $h$ × |
| 质量影响 | 基线 | 极小 | 可接受 |
| 工程实现 | 无 broadcast | 需要 repeat/broadcast | 需要 repeat/broadcast |
| 编译器优化机会 | — | fuse broadcast into SDPA stride | 同 GQA |
| 适用场景 | 小模型/研究 | **当前主流大模型** | 追求极致推理效率 |

---

## 参考文献

1. Vaswani et al., "Attention Is All You Need" (2017) — MHA 原始论文
2. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019) — MQA 提出
3. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023) — GQA 提出
4. LLaMA-2 Technical Report (2023) — 70B 使用 GQA 的工程实践
