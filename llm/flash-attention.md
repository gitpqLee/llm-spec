# Flash Attention 完整解析

## 1. SDPA 回顾

标准 Scaled Dot-Product Attention：

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d + M) · V
```

四步：
1. `S = Q·Kᵀ`：算分数矩阵 `[Lq, Lk]`
2. `S / √d`：缩放，防止点积过大导致 softmax 退化成 one-hot
3. softmax（对 Score 矩阵的每一行独立做）：把每个 query 对所有 key 的分数变成概率分布
4. `P · V`：用概率加权混合 Value

### softmax 的三步

对每一行：

```
softmax(xⱼ) = exp(xⱼ - m) / Σₖ exp(xₖ - m),   其中 m = max(x)
```

| 步骤 | 操作 | 为什么 |
|------|------|--------|
| ReduceMax | 找每行最大值 m | 数值稳定，防 exp(x) 溢出 |
| SubExp | 算 exp(xⱼ - m) | 得到非负权重 |
| ReduceSum + Divide | 除以行和 | 归一化成概率 |

### Lq 和 Lk 的关系

| 场景 | Lq | Lk | 关系 |
|------|--------|--------|------|
| Self-attention prefill | prompt 长度 | prompt 长度 | Lq = Lk |
| Self-attention decode | 1 | 累积序列长度 | Lq ≪ Lk |
| Cross-attention | 解码器长度 | 编码器长度 | 无固定关系 |

---

## 2. 为什么需要 Flash Attention

普通 Attention 的问题：中间矩阵 `S = Q·Kᵀ` 大小是 `Lq × Lk`，序列长时内存炸。

```
普通 Attention：一次算完整张表
        ┌───    Key 序列 (Lk=8)      ───┐
        k0  k1  k2  k3  k4  k5  k6  k7
   q0 │ s00 s01 s02 s03 s04 s05 s06 s07 │  → softmax → Σ(p·V)
   q1 │ s10 s11 s12 s13 s14 s15 s16 s17 │  → softmax → Σ(p·V)
        └───────────────────────────────┘
                   必须全部算完才能做 softmax
```

Flash Attention 快的本质是**减少高成本内存搬运**：中间值尽量留在片上缓存，边算边归并，不落地完整分数矩阵。

---

## 3. Flash Attention 核心思想

不改变数学结果，只改变计算顺序：
1. 把 K 和 V 按列方向切成小块
2. 每次拿一小块和 Q 算局部分数
3. 不保存整张分数表，只更新每一行的在线统计量
4. 全部块处理完，再一次性得到最终输出

### 分块图示

```
        ┌── KV Block 0 ──┐  ┌── KV Block 1 ──┐
        k0  k1  k2  k3      k4  k5  k6  k7
   q0 │ s00 s01 s02 s03 │  │ s04 s05 s06 s07 │
   q1 │ s10 s11 s12 s13 │  │ s14 s15 s16 s17 │
        └────────────────┘  └────────────────┘
              先算这块               再算这块
          只需 Lq×B 的空间       复用同一块空间
```

---

## 4. Running State（在线统计量）

每行维护三个量：

| 名称 | 形状 | 含义 |
|------|------|------|
| running_max `m` | [Lq]，每行一个标量 | 到目前为止见过的最大分数 |
| running_sum `l` | [Lq]，每行一个标量 | 到目前为止的指数和（softmax 分母） |
| running_output `o` | [Lq, dv]，每行一个向量 | 到目前为止的未归一化加权输出 |

### 为什么是这三个量

softmax 定义需要：
- **m**：减最大值防溢出，分块时 max 跨块变化，要维护
- **l**：分母 `Σ exp(xₖ - m)`，最后归一化用
- **o**：分子 `Σ exp(xₖ - m) · vₖ`，故意不除 l，这样分块时可以自由累加和校正

最终：`output = o / l`

### 为什么 o 不提前除 l

因为分块时 l 还没算完。如果提前除了，等新块来了 l 变了，之前除的就全错了。不除就没这个问题——o 是线性累加的，可以校正后直接加，最后一次性除。

---

## 5. 在线合并公式（核心）

处理新 KV 块时，先算该块的局部统计 `m̂, l̂, ô`，再和旧状态合并：

```
m'  = max(m, m̂)                    ← 新的全局 max

α   = exp(m - m')                   ← 旧状态的校正因子
β   = exp(m̂ - m')                  ← 新块的校正因子

l'  = l · α + l̂ · β               ← 合并指数和
o'  = o · α + ô · β               ← 合并未归一化输出

最终: output = o' / l'              ← 只在最后除一次
```

### 校正因子的直觉

新旧两块的 exp 是在不同 max 基准下算的，不能直接加。要先乘校正因子对齐到同一个全局 max 基准：

```
例1: m_old = 4, m̂ = 3 → m' = 4

    旧块：α = exp(4-4) = 1        （不变）
    新块：β = exp(3-4) = 0.368    （缩小）

例2: m_old = 3, m̂ = 5 → m' = 5

    旧块：α = exp(3-5) = 0.135    （缩小）
    新块：β = exp(5-5) = 1        （不变）
```

无论谁大谁小，公式都能正确对齐。α ≤ 1, β ≤ 1，不会溢出。

---

## 6. 手算数值验证

### 设定

分数矩阵 S（缩放后）：

```
S = | 2  4  1  3 |
    | 1  3  2  5 |

V = | 1.0  0.0 |
    | 0.0  1.0 |
    | 1.0  1.0 |
    | 0.5  0.5 |
```

块大小 = 2，分成 Block 0 (k0,k1) 和 Block 1 (k2,k3)。

### 普通 softmax 标准答案

```
Row 0: [2,4,1,3], m=4, l=1.5530
  → output = [0.2377, 0.7944]

Row 1: [1,3,2,5], m=5, l=1.2034
  → output = [0.4721, 0.5693]
```

### Flash 在线算法（Row 0）

**Block 0**：分数 [2, 4]

```
m̂ = 4
exp: exp(2-4)=0.1353, exp(4-4)=1
l̂ = 0.1353 + 1 = 1.1353
ô = 0.1353×[1,0] + 1×[0,1] = [0.1353, 1]

初始化: m=4, l=1.1353, o=[0.1353, 1]
```

**Block 1**：分数 [1, 3]

```
m̂ = 3
exp: exp(1-3)=0.1353, exp(3-3)=1
l̂ = 1.1353
ô = 0.1353×[1,1] + 1×[0.5,0.5] = [0.6353, 0.6353]
```

合并：

```
m' = max(4, 3) = 4
α  = exp(4-4) = 1
β  = exp(3-4) = 0.3679

l' = 1.1353×1 + 1.1353×0.3679 = 1.5530  ✓
o' = [0.1353,1]×1 + [0.6353,0.6353]×0.3679 = [0.3691, 1.2338]

output = o'/l' = [0.3691/1.5530, 1.2338/1.5530] = [0.2377, 0.7944]  ✓
```

**和标准答案完全一致。这不是近似，是恒等变换。**

---

## 7. Causal Mask 优化

分块后，每个 KV 块和对角线的关系只有三种：

```
情况 1: 全有效块（对角线左下方）→ 正常算，跳过 mask 处理
情况 2: 对角线块（和对角线相交）→ 把右上位置填 -inf
情况 3: 全 mask 块（对角线右上方）→ 直接跳过！不算 MatMul，不加载 K/V
```

以 Lq = Lk = 1024，block_size = 256 为例：
- 没有 causal 优化：4×4 = 16 块
- 有 causal 优化：1+2+3+4 = 10 块，省了 37.5%

序列越长，省的越多（趋近 50%）。

---

## 8. 内存对比

```
普通 Attention:
  Score Matrix: Lq × Lk        ← 完整物化

Flash Attention:
  Score: Lq × B                 ← 只要一块的大小，复用空间
  m: [Lq]                       ← 每行一个 running max
  l: [Lq]                       ← 每行一个 running sum
  o: [Lq, dv]                   ← 每行一个 running output
```

当 Lk = 8192，B = 256 时，score 空间节省 32 倍。

---

## 9. Python 实现

```python
import numpy as np

def flash_attention(Q, K, V, block_size):
    """
    Q: [Lq, d], K: [Lk, d], V: [Lk, dv]
    """
    Lq, d = Q.shape
    Lk, dv = V.shape

    # 初始化 running state
    m = np.full(Lq, -np.inf)
    l = np.zeros(Lq)
    o = np.zeros((Lq, dv))

    num_blocks = (Lk + block_size - 1) // block_size

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, Lk)
        K_block = K[start:end]
        V_block = V[start:end]

        # 局部分数
        S_block = Q @ K_block.T / np.sqrt(d)

        # 局部统计量
        m_block = S_block.max(axis=1)
        exp_block = np.exp(S_block - m_block[:, None])
        l_block = exp_block.sum(axis=1)
        o_block = exp_block @ V_block

        # 在线合并
        m_new = np.maximum(m, m_block)
        alpha = np.exp(m - m_new)
        beta = np.exp(m_block - m_new)
        l = l * alpha + l_block * beta
        o = o * alpha[:, None] + o_block * beta[:, None]
        m = m_new

    # 最终归一化
    return o / l[:, None]
```

### 验证

```python
def standard_attention(Q, K, V):
    S = Q @ K.T / np.sqrt(Q.shape[1])
    S_max = S.max(axis=1, keepdims=True)
    exp_S = np.exp(S - S_max)
    P = exp_S / exp_S.sum(axis=1, keepdims=True)
    return P @ V

np.random.seed(42)
Q = np.random.randn(4, 64)
K = np.random.randn(128, 64)
V = np.random.randn(128, 32)

print(np.allclose(standard_attention(Q, K, V),
                  flash_attention(Q, K, V, block_size=32)))  # True
```

### 加 causal mask

```python
# 在算完 S_block 之后加：
for i in range(Lq):
    for j in range(end - start):
        if start + j > i:
            S_block[i, j] = -np.inf
```

---

## 10. 计算流程图

```
Load Q
  │
  ▼
┌─────────────── KV Block 循环 ───────────────┐
│                                              │
│  Load K_block, V_block                       │
│       │                                      │
│       ▼                                      │
│  MatMul: Q × K_block^T → S_block            │
│       │                                      │
│       ▼                                      │
│  Scale (÷√d) + Mask (optional)               │
│       │                                      │
│       ▼                                      │
│  ReduceMax → m̂ (每行局部最大值)              │
│       │                                      │
│       ▼                                      │
│  SubExp: exp(S - m̂) → exp_block             │
│       │                                      │
│       ├──→ ReduceSum → l̂                    │
│       │                                      │
│       └──→ MatMul: exp_block × V → ô        │
│                    │                         │
│                    ▼                         │
│  Online Merge: 用 α,β 合并 m,l,o            │
│                                              │
└──────────────────────────────────────────────┘
  │
  ▼
Finalize: output = o / l
```

---

## 11. 面试要点总结

| 问题 | 答案 |
|------|------|
| FA 改了什么？ | 不改公式，改计算顺序（分块 + 在线 softmax） |
| 为什么快？ | 减少内存读写，score 不落地到主存 |
| 正确性？ | 恒等变换，不是近似 |
| 三个 running state？ | max（对齐基准）、sum（分母）、output（未归一化分子） |
| α 和 β 是什么？ | 把新旧统计量对齐到同一 max 基准的校正因子 |
| 第一块要特殊处理吗？ | 不用，m 初始化为 -∞ 所以 α=0，旧状态自动清零 |
| causal 怎么优化？ | 对角线右上方的整块直接跳过，省接近 50% 计算 |
| 和 FA2 的区别？ | FA2 还沿 Q 方向切外层循环，提高 GPU 并行度 |
| 计算量变了吗？ | 基本不变（略多校正运算），加速来自减少 IO |
