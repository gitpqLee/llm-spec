# Flash Attention 完整解析

## 1. SDPA 回顾

标准 Scaled Dot-Product Attention：

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d + M) · V
```

四步：
1. `S = Q·Kᵀ`：算分数矩阵 `[Lq, Lk]`
2. `S / √d`：缩放 QK 点积的方差，减轻 softmax 饱和；数值溢出主要由减去行最大值解决
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

**在实数精确运算下与标准答案等价，不是算法近似。** 实际 kernel 因为分块改变了浮点累加顺序，
再加上 BF16/FP16、近似指数和并行归约，结果通常不会与普通实现逐 bit 相同。

---

## 7. 为什么切 KV 需要在线 softmax，切 Q 不需要

FA 的核心是沿 KV 序列方向（列方向）切块。Q 也可以切，但原理完全不同：

```
切 KV（沿列切）— 一行被切断：
  q0: [ s00 s01 | s02 s03 ]     ← softmax 需要看完整行，被切断了
                                   → 必须用在线合并（running max/sum/output）

切 Q（沿行切）— 每行仍然完整：
  Block A: q0 [ s00 s01 s02 s03 ]  ← 完整一行，softmax 正常做
           q1 [ s10 s11 s12 s13 ]

  Block B: q2 [ s20 s21 s22 s23 ]  ← 完整一行，softmax 正常做
           q3 [ s30 s31 s32 s33 ]

  → 两块互不干扰，最后直接拼起来，不需要校正因子
```

切 Q 不需要在线 softmax 的原因：softmax 是逐行独立的，切 Q 只是把不同行分到不同块，
每一行仍然能看到完整的 KV 序列，softmax 的输入没有被截断。

上面的图和 Python 代码为了方便讲解，让整个 Q 参与每个 KV block 的计算。生产级 FA1 和 FA2
都会把 Q 切成大小为 `Br` 的 tile，并让一个 Q tile 依次处理多个大小为 `Bc` 的 KV tile。

FA2 的关键改进不是“FA1 不切 Q、FA2 才切 Q”，而是进一步改进 Q tile 的并行调度和
warp 间工作划分，减少重复访存与非 MatMul 运算，并在 batch/head 数较少时提供更多并行任务。

---

## 8. Causal Mask 优化

分块后，每个 KV 块和对角线的关系只有三种：

```
情况 1: 全有效块（对角线左下方）→ 正常算，跳过 mask 处理
情况 2: 对角线块（和对角线相交）→ 把右上位置填 -inf
情况 3: 全 mask 块（对角线右上方）→ 直接跳过！不算 MatMul，不加载 K/V
```

以对齐的方形 causal self-attention，`Lq = Lk = 1024`、`block_size = 256` 为例：
- 没有 causal 优化：4×4 = 16 块
- 有 causal 优化：1+2+3+4 = 10 块，省了 37.5%

对于足够长、Q/K 位置对齐的方形 causal self-attention，可跳过的块比例趋近 50%。这个结论不适用于
普通 cross-attention；decode 还需要考虑当前 query 在 KV cache 中的位置偏移，不能简单使用块内行号判断。

---

## 9. 内存对比

```
普通 Attention:
  Score Matrix: Lq × Lk        ← 完整物化

Flash Attention:
  Score tile: Br × Bc           ← Q/KV tile 的局部分数，片上复用
  m: [Br]                        ← 当前 Q tile 每行一个 running max
  l: [Br]                        ← 当前 Q tile 每行一个 running sum
  o: [Br, dv]                    ← 当前 Q tile 每行一个 running output
```

标准 Attention 需要物化大小为 `Lq × Lk` 的分数/概率中间张量。FlashAttention 的主要片上
工作空间约为 `Br × Bc + Br × dv`，并对每个 tile 复用。它主要降低中间张量的 HBM 流量，
并没有改变 Attention 的渐近计算复杂度。

---

## 10. Python 实现

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

    这段代码只用于说明位置级 mask，并没有实现“跳过整块”的性能优化。生产 kernel 会先根据 Q/K tile
    的全局位置判断：完全有效块不做 mask，完全无效块不加载也不计算，仅对角线相交块执行逐元素 mask。
    如果一个块对某行全部无效，还必须直接跳过或特殊处理，避免 `-inf - (-inf)` 产生 NaN。

---

## 11. 计算流程图

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

## 12. FA1、FA2、FA3 的区别

三代 FlashAttention 共享同一个根基：IO-aware tiling 和在线 softmax。主要差异在硬件调度，而不是
Attention 数学公式改变。

| 版本 | 主要改进 | 典型硬件重点 |
|------|----------|--------------|
| FA1 | 分块计算、在线 softmax，不物化完整 score/probability | 降低 HBM 读写 |
| FA2 | 改进 Q tile 并行调度和 warp 工作划分，减少非 MatMul 开销 | 提高 occupancy 和 Tensor Core 利用率 |
| FA3 | 异步执行、warp specialization、TMA、ping-pong pipeline，并加强低精度支持 | 面向 Hopper 等新架构 |

FA3 中的 TMA、warp specialization 等不是所有 FlashAttention 实现都天然具备的能力，而是依赖具体
GPU 架构和 kernel 实现。

---

## 13. Backward：用重计算换显存

普通训练实现可能保存完整概率矩阵 `P`，其大小为 `Lq × Lk`。FlashAttention forward 通常只保存：

- 输出 `O`；
- 每个 query 行的 log-sum-exp，或等价的 `m/l` 统计量；
- dropout 所需的可重现随机状态（启用 dropout 时）。

Backward 再按 tile 重算：

```text
Q tile, K tile
  ↓
重新计算 score 和 probability
  ↓
结合 dO、V 计算 dQ、dK、dV
```

因此 backward 不是简单把 forward 的在线 recurrence 倒序执行，而是使用保存的行统计量恢复每个 tile
的概率，再累计梯度。它增加了一部分重复计算，但避免保存和读取巨大的概率矩阵；在现代 GPU 上，
多做一些 GEMM 往往比额外搬运 `O(Lq × Lk)` 数据更划算。

---

## 14. Prefill 与 Decode

### Prefill

Prefill 中 `Lq` 和 `Lk` 都较大，存在大量 Q/KV tile，可获得规则的矩阵并行和较高 Tensor Core 利用率。
这是经典 FlashAttention kernel 最擅长的场景。

### Decode

逐 token decode 通常有：

```text
Lq = 1
Lk = 已有 KV cache 长度
```

此时 Q 方向几乎没有并行度，性能更多受 KV cache 读取带宽限制。工程实现通常需要：

- 将 KV 序列切给多个 CTA，再归并局部 softmax 状态（split-KV / FlashDecoding）；
- 支持 paged KV cache；
- 针对 MQA/GQA 避免重复读取共享 K/V；
- 处理 query 的绝对位置偏移和 causal 边界。

因此“支持 FlashAttention”不代表 prefill kernel 原样用于 decode 仍然高效。

---

## 15. MHA、GQA、MQA

记：

```text
Q: [B, Hq, Lq, d]
K: [B, Hkv, Lk, d]
V: [B, Hkv, Lk, dv]
```

- MHA：`Hq = Hkv`；
- GQA：多个 Q head 共享一个 KV head；
- MQA：所有 Q head 共享同一个 KV head。

高效 FlashAttention kernel 应在 head 映射中共享 K/V tile，而不是先把 K/V 物理复制到 `Hq` 份。
这对 decode 尤其重要，因为此时主要成本就是读取 KV cache。详细结构参见
[attention-mha-gqa-mqa.md](attention-mha-gqa-mqa.md)。

---

## 16. 复杂度与性能边界

FlashAttention 的主要价值是降低 HBM 流量和中间存储，而不是降低 dense Attention 的渐近 FLOPs：

```text
计算复杂度：O(Lq × Lk × (d + dv))
普通中间矩阵：O(Lq × Lk)
片上 tile 工作区：O(Br × Bc + Br × dv)
```

实际速度还取决于：

- tile 大小和 head dimension；
- causal、variable-length 和 dropout；
- batch/head 数是否足够填满 GPU；
- 数据类型与硬件 Tensor Core 能力；
- KV cache 是否连续、分页或量化；
- kernel 是否针对 prefill 或 decode 调度。

---

## 17. 面试要点总结

| 问题 | 答案 |
|------|------|
| FA 改了什么？ | 不改公式，改计算顺序（分块 + 在线 softmax） |
| 为什么快？ | 减少内存读写，score 不落地到主存 |
| 正确性？ | 实数数学下等价，不是算法近似；浮点结果不保证逐 bit 相同 |
| 三个 running state？ | max（对齐基准）、sum（分母）、output（未归一化分子） |
| α 和 β 是什么？ | 把新旧统计量对齐到同一 max 基准的校正因子 |
| 第一块要特殊处理吗？ | 不用，m 初始化为 -∞ 所以 α=0，旧状态自动清零 |
| causal 怎么优化？ | 全有效块免 mask，全无效块跳过，对角块局部 mask；方形长序列最多接近跳过一半块 |
| 和 FA2 的区别？ | FA1/FA2 都切 Q；FA2 改善 Q tile 调度、warp 分工和硬件利用率 |
| FA3 的重点？ | Hopper 上的异步执行、TMA、warp specialization 和流水重叠 |
| 计算量变了吗？ | 渐近 FLOPs 基本不变，主要加速来自减少 HBM IO；backward 会重计算部分 score |
| Decode 为什么特殊？ | `Lq=1` 缺少 Q 并行度，通常需要 split-KV/FlashDecoding 类调度 |
