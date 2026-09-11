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

## 17. OpenVINO NPUW HFA 实现

### 17.1 HFA 是什么

NPUW 中的 HFA 是 **Host Flash Attention**。它沿 KV 序列维切块，逐块计算 Attention，并使用
online softmax 状态保证结果等价于一次处理完整 KV 序列。

这里的 `Host` 容易引起误解：

- Host/NPUW 负责识别 Attention、选择 tile、准备 tensor view、绑定输入输出和顺序启动 NPU request；
- NPU 负责 QK、mask、指数、归约、PV、running state 合并和最终归一化；
- Host 不会把每块的数值结果拿回 CPU 做 softmax 合并。

因此 HFA 可以理解为：

```text
Host 控制循环 + NPU 执行每个 FlashAttention tile
```

它与“一个 kernel 内部完成全部 KV tile 循环”的设备端 FlashAttention 不完全相同。HFA 的 KV tile
循环跨越多个 NPU infer request，tile 之间存在 Host 调度；但每个 tile 的数值计算仍在 NPU 上。

### 17.2 从原始 Attention 到 HFA

原始模型中的 Attention 通常近似为：

```text
Q ───────────────┐
     ├─ QKᵀ → Add(mask) → Softmax → PV → output
past_K + new_K ──┤
past_V + new_V ──┘
```

NPUW 启用 HFA 后，不再一次物化完整 `[Lq, Lk]` score，而是构造两类可编译子模型：

```text
Regular Tile Model
    输入：Q, K_tile, V_tile, past_acc, past_max, past_sum
    输出：new_acc, new_max, new_sum

Final Tile Model
    输入：Q, final_K_tile, final_V_tile, past_acc, past_max, past_sum, mask
    输出：最终 Attention output
```

普通 tile 可以复用同一个 compiled model 和 infer request。最后一个 tile 单独使用 final model，原因是它
需要处理尾部 mask，并把 running output 归一化、转置和 reshape 成原 Attention 的输出格式。

### 17.3 一个 Attention 层的运行流程

假设 KV 长度被切成四块：

```text
KV tile 0 | KV tile 1 | KV tile 2 | final KV tile 3
```

执行过程如下：

```text
Host 初始化状态：
    acc = 0
    max = -∞
    sum = 0
  │
  ▼
NPU Regular Tile 0(Q, K0, V0, state0)
  │ state1 = (acc1, max1, sum1)
  ▼
NPU Regular Tile 1(Q, K1, V1, state1)
  │ state2 = (acc2, max2, sum2)
  ▼
NPU Regular Tile 2(Q, K2, V2, state2)
  │ state3 = (acc3, max3, sum3)
  ▼
NPU Final Tile 3(Q, K3, V3, mask3, state3)
  │
  ▼
归一化后的 Attention output
```

NPUW 将 regular tile 的输出 tensor 与下一次 regular tile 的输入 tensor 绑定到同一组 state buffer，
也把这组 buffer 绑定为 final tile 的输入。这样状态留在设备可访问内存中，无需每轮由 CPU 读取、计算
再写回。

对每个 tile，Host 负责：

1. 根据当前 KV offset 取得 `K_tile`、`V_tile` 和需要的 mask tile；
2. 条件允许时直接复用完整 tensor，或者创建 tensor view；
3. 无法 view 时才把对应范围复制到 tile buffer；
4. 绑定 Q、KV tile 和 running state；
5. 启动 regular/final NPU infer request；
6. 等待依赖完成后处理下一 tile。

这里的 tile request 必须顺序执行，因为 tile $i+1$ 依赖 tile $i$ 产生的 `acc/max/sum`。

### 17.4 HFA 中的 online softmax

设前面所有 tile 的状态为：

```text
past_max = m
past_sum = l
past_acc = o
```

当前 tile 首先计算：

$$
S_i = QK_i^T + M_i
$$

然后直接选取旧状态与当前 tile 的共同最大值：

$$
m' = \max(m, \mathrm{rowmax}(S_i))
$$

当前 tile 的指数权重为：

$$
P_i = \exp(S_i-m')
$$

旧状态需要从旧的最大值基准修正到新基准：

$$
\alpha = \exp(m-m')
$$

更新 running sum 和 running accumulator：

$$
l' = \alpha l + \mathrm{rowsum}(P_i)
$$

$$
o' = \alpha o + P_iV_i
$$

注意，这个实现没有显式计算前文通用公式中的 $\beta$。因为当前 tile 的 $P_i$ 已经直接使用新的
全局最大值 $m'$ 作为基准，$\exp(\hat m-m')$ 已经隐含在 $\exp(S_i-m')$ 中。

最后一个 tile 完成后：

$$
O = \frac{o'}{l'}
$$

这不是把多个局部 Softmax 输出相加，而是每处理一块就把未归一化分子、分母和最大值递推到统一基准。

### 17.5 Non-fused HFA 路径

当：

```json
"NPUW_ATTN_HFA_FUSED": "NO"
```

每个 HFA tile 被展开为一个普通 OpenVINO 子图：

```text
MatMul(Q, Kᵀ)
  → Add(mask)
  → ReduceMax + Maximum(past_max)
  → Subtract + Exp
  → ReduceSum
  → Multiply/Add 更新 past_sum
  → MatMul(P, V)
  → Multiply/Add 更新 past_acc
```

final tile 子图再执行：

```text
Divide(acc, sum)
  → Transpose
  → Reshape
  → Attention output
```

“Non-fused”表示这些数学操作在图中仍是独立算子，不表示它们在 CPU 上运行。整个 tile 子图仍然被编译
并交给 NPU 执行，只是会产生更多中间 tensor、设备内存流量和算子调度。

### 17.6 Fused HFA 路径

当：

```json
"NPUW_ATTN_HFA_FUSED": "YES"
```

tile 子图中的 Attention 核心被替换成 NPU internal op：

```text
FlashAttentionTile(
    query,
    key_tile,
    value_tile,
    running_output,
    running_max,
    running_sum,
    optional_mask,
    config = {is_head, is_tail}
)
```

它输出新的：

```text
running_output, running_max, running_sum
```

其中：

- regular tile 通常不带 mask，以减少不必要的 mask 处理；
- final tile 带 mask，用于处理最后 KV 块中的有效范围；
- final tile 设置 `is_tail=true`，kernel 内部直接完成最终 `acc/sum`；
- 外层 final tile 子图检测到 fused 路径后，不再额外插入 `Divide`。

所以 fused 路径的职责划分是：

```text
Host/NPUW：
    KV 切块、tensor/view 绑定、mask tile 准备、request 顺序调度

FlashAttentionTile fused kernel：
    QK、mask、online softmax、PV、running state 合并
    final tile 中再完成最终归一化
```

需要特别区分两个层次：

```text
NPUW_ATTN_HFA_FUSED=YES
    融合的是“单个 HFA tile 内部”的数学操作

HFA 的整个 KV tile 循环
    当前仍由 Host 逐个启动 NPU request，不是一次 fused kernel launch
```

因此 fused HFA 减少了 tile 内部的中间数据和算子开销，但没有消除 tile 之间的 Host 调度边界。

### 17.7 Prefill 与 Decode 中的 HFA

#### Prefill

Prefill 通常有 `Lq > 1`。Q 和 KV 都可能很长，HFA 沿 KV 方向切块，让每次只产生
`[B, Hq, Lq, tile_size]` 的局部 score，而不产生完整 `[B, Hq, Lq, Lk]` score。

主要收益是：

- 降低峰值中间内存；
- 降低完整 score/probability 的 DDR 写回和读回；
- 长 prompt 可以配合 chunk prefill 限制单次模型输入长度。

#### Decode

逐 token decode 通常有：

```text
Lq = 1
Lk = prompt tokens + 已生成 tokens
```

此时 HFA 仍沿越来越长的 KV cache 切块：

```text
Q_new × KV tile 0 → state1
Q_new × KV tile 1 → state2
...
Q_new × final KV tile → output
```

Decode 不需要保存巨大的 `Lq × Lk` score，因为 `Lq=1` 时 score 本身只是一行；HFA 在这个阶段的价值
更偏向于以固定 tile 消费长 KV cache，并为 split-KV/FlashDecoding 风格的执行提供 online softmax 合并。
但 tile 越多，Host 顺序调度的固定开销也越明显，因此短 context 或并行度很低时不保证一定更快。

要让 generate/decode 阶段进入 fused HFA，关键配置是：

```json
{
  "NPUW_LLM_GENERATE_ATTENTION_HINT": "HFA",
  "NPUW_ATTN_HFA_FUSED": "YES"
}
```

仅设置 `NPUW_ATTN_HFA_FUSED=YES` 不够。generate attention 默认是 `STATIC`，必须先用
`NPUW_LLM_GENERATE_ATTENTION_HINT=HFA` 选择 HFA 路径。模型图还必须匹配 NPUW 可识别的 Attention
模式，并满足目标 NPU/compiler 对 `FlashAttentionTile` 的支持条件。

### 17.8 HFA 与 block-based KV cache

HFA 和 block-based KV cache 是两个正交层面的优化：

| 优化 | 解决的问题 |
|------|------------|
| HFA | 如何分块计算 Attention，并在线合并 softmax 状态 |
| Block-based KV cache | 历史 K/V 如何分块存储、绑定、增长和复用 |

没有 block cache 时，HFA 可以从一块逻辑连续的 K/V tensor 中按 offset 创建 tile view：

```text
Continuous KV cache
└── view(KV[0:B]), view(KV[B:2B]), ...
```

启用 block cache 后，历史 K/V 已经由多个独立 block 表示。HFA 遍历 block，并在 block 内继续按自己的
`tile_size` 取 tile：

```text
KV block 0
  ├── HFA tile 0
  └── HFA tile 1
KV block 1
  ├── HFA tile 2
  └── HFA tile 3
```

因此要求 block size 是 HFA tile size 的整数倍。Block cache 让 prefill 建立的 KV blocks 能被 decoding
持续复用和追加，减少不断扩展连续 KV tensor 所需的重分配与历史数据复制；HFA 则负责读取这些 blocks
并完成 Attention。开启 block cache 不会自动开启 fused HFA，反过来也一样。

完整组合通常类似：

```json
{
  "NPUW_LLM_PREFILL_HINT": "DYNAMIC",
  "NPUW_LLM_PREFILL_CHUNK_SIZE": "1024",
  "NPUW_LLM_PREFILL_ATTENTION_HINT": "HFA",
  "NPUW_LLM_GENERATE_ATTENTION_HINT": "HFA",
  "NPUW_ATTN_HFA_FUSED": "YES",
  "NPUW_LLM_ENABLE_BLOCK_BASED_KV_CACHE": "YES",
  "NPUW_LLM_ENABLE_PREFIX_CACHING": "NO"
}
```

其中 chunk size 必须是正的 2 的幂、小于最大 prompt 长度；block cache 当前要求 chunk prefill，且不能
与 prefix caching 同时启用。

### 17.9 HFA 数据流总结

```text
原始 Attention 子图
  │
  │ NPUW 识别并选择 HFA
  ▼
构造并编译：Regular Tile Model + Final Tile Model
  │
  ▼
Host 准备 Q、KV blocks/views、mask 和初始 running state
  │
  ▼
┌────────────── past KV tile 循环 ──────────────┐
│ Host 绑定当前 K/V tile                         │
│        ↓                                      │
│ NPU tile 计算 QK、online softmax、PV           │
│        ↓                                      │
│ NPU 更新 acc/max/sum                           │
└───────────────────────────────────────────────┘
  │
  ▼
Host 启动 final tile request
  │
  ▼
NPU 处理 final K/V + mask，合并状态并完成归一化
  │
  ▼
Attention output
```

最核心的结论是：

> HFA 的“Host”负责切块和控制流；Attention 数学计算、跨 tile 的 online softmax 状态更新以及最终归一化
> 都由编译后的 NPU tile 子模型执行。开启 fused 后，这些数值操作进一步落入 `FlashAttentionTile` kernel，
> 但整个 KV tile 循环仍由 NPUW 在 Host 侧依次调度。

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
