# RoPE 旋转位置编码 — 从原理到 Gemma 4 Dual RoPE

## 1. 为什么需要位置编码？

Transformer 的 self-attention 本质上是一个**集合操作**——它对输入 token 的顺序是无感知的。把 "猫吃鱼" 和 "鱼吃猫" 输入同一个没有位置编码的 Transformer，attention 的计算结果完全相同。

因此必须额外注入位置信息，让模型知道：**这个 token 在第几个位置**。

---

## 2. 位置编码的演进

| 方法 | 代表模型 | 特点 |
|---|---|---|
| 正弦位置编码 | 原始 Transformer | 固定的、与模型无关 |
| 可学习绝对位置 | GPT-2, BERT | 学出来的，但长度固定 |
| 相对位置编码 | T5, ALiBi | 编码 token 之间的距离，而不是绝对位置 |
| **RoPE** | Llama, Gemma, Qwen | 通过旋转实现相对位置，兼具优雅和外推能力 |

---

## 3. RoPE 的核心思想

### 一句话总结

> **把每个 token 的 Q/K 向量"旋转"一个与其位置成正比的角度，使得两个 token 的点积只取决于它们的相对位置差。**

### 直觉类比

想象一个时钟：
- 位置 0 的 token 在 12 点方向
- 位置 1 的 token 旋转了 θ 角度
- 位置 2 的 token 旋转了 2θ 角度
- 位置 m 的 token 旋转了 mθ 角度

当计算 Q_m 和 K_n 的点积时（attention score），由于旋转的性质：

$$Q_m \cdot K_n = f(x_m, x_n, m-n)$$

点积的结果**只取决于相对距离 m-n**，而不是绝对位置 m 和 n。

---

## 4. RoPE 的数学定义

### 4.1 二维情况（最简单的 case）

对一个二维向量 $[x_0, x_1]$，在位置 $m$ 处旋转角度 $m\theta$：

$$\text{RoPE}([x_0, x_1], m) = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \end{bmatrix}$$

展开：
$$= \begin{bmatrix} x_0 \cos(m\theta) - x_1 \sin(m\theta) \\ x_0 \sin(m\theta) + x_1 \cos(m\theta) \end{bmatrix}$$

### 4.2 高维情况（head_dim = d）

把 d 维向量拆成 d/2 对，每对独立旋转，但用**不同频率**的角度：

$$\theta_i = \text{base}^{-2i/d}, \quad i = 0, 1, ..., d/2-1$$

其中 `base` 通常取 10000。

对位置 m，第 i 对的旋转角度为 $m \cdot \theta_i$：

$$\begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix} \rightarrow \begin{bmatrix} x_{2i} \cos(m\theta_i) - x_{2i+1} \sin(m\theta_i) \\ x_{2i} \sin(m\theta_i) + x_{2i+1} \cos(m\theta_i) \end{bmatrix}$$

### 4.3 频率的含义

不同维度对有不同的旋转频率：

| 维度对 i | 频率 θ_i | 旋转速度 | 感知范围 |
|---|---|---|---|
| i=0 | 1.0 (最快) | 每步转很大角度 | 感知近距离位置差 |
| i=1 | 0.01 | 较慢 | 感知中等距离 |
| ... | ... | ... | ... |
| i=d/2-1 | 极小 (最慢) | 几乎不转 | 感知超远距离位置差 |

**类比**：就像时钟的秒针（快频、感知秒级变化）、分针（中频）、时针（慢频、感知小时级变化）。

---

## 5. RoPE 的实际计算方式

### 5.1 等价的高效实现（不用矩阵乘法）

原始旋转矩阵乘法太慢。实际实现用元素级运算替代：

```python
# x shape: [batch, seq_len, num_heads, head_dim]
# 将 x 分成前后两半
x_first  = x[..., :head_dim//2]        # 前半
x_second = x[..., head_dim//2:]        # 后半

# 构造 "rotated" 版本：[-后半, 前半]
x_rotated = concat(-x_second, x_first)  # 旋转90度

# 应用 RoPE
output = x * cos + x_rotated * sin
```

### 5.2 对应 Gemma 4 子图中的计算

```
id=52  Multiply:  Q * cos              ← x * cos 部分
id=68  Slice:     Q[..., :128]         ← 取前半 (head_dim/2=128)
id=59  Slice:     Q[..., 128:]         ← 取后半
id=69  Concat:    [-后半, 前半]          ← 构造旋转版本
id=70  Multiply:  旋转版本 * sin         ← x_rotated * sin 部分
id=71  Add:       Q*cos + rotated*sin   ← 最终结果
```

---

## 6. RoPE 为什么好？

### 6.1 相对位置特性（证明）

位置 m 的 Q 和位置 n 的 K 做点积：

$$Q_m \cdot K_n = \text{RoPE}(q, m) \cdot \text{RoPE}(k, n)$$

由于旋转矩阵的性质 $R_m^T R_n = R_{n-m}$：

$$= q^T R_{m-n} k$$

结果只依赖 **m-n**（相对距离），不依赖绝对位置！

### 6.2 长度外推

训练时只见过位置 0~2048，推理时可以直接用位置 5000，因为：
- RoPE 对任何位置 m 都有定义（不像学习的 embedding 那样固定长度）
- 相对位置特性意味着只要 m-n 不变，attention pattern 不变

### 6.3 无额外参数

RoPE 的 cos/sin 由公式确定，**不需要学习任何参数**。

---

## 7. Gemma 4 的 Dual RoPE

### 7.1 为什么需要两套 RoPE？

Gemma 4 有两种 attention 层：

```
47层的排列: L L L L L G L L L L L G L L L L L G ...
            ←— 5个Local —→ ↑Global
```

**Local 层**（滑动窗口）：只关注附近的 token（窗口内）
**Global 层**（全序列）：关注所有 token（整个上下文）

两种层对位置信息的需求不同：

| | Local 层 | Global 层 |
|---|---|---|
| 需要区分的距离 | 近距离（窗口内，几百~几千 token） | 远距离（整个上下文，数万 token） |
| 频率需求 | 高频为主（细粒度区分近邻） | 低频为主（区分远距离） |
| head_dim | 256 | 512 |
| RoPE 维度 | 256 (128对) | 512 (256对) |

### 7.2 直觉解释

**Local RoPE (256d)** 就像一把短尺子：
- 刻度密（高频），能精确测量 1mm、2mm、3mm 的差别
- 但尺子短（维度小），测不了几十米的距离
- 适合：区分 "前一个词" 和 "前两个词" 的微妙差别

**Global RoPE (512d)** 就像一把长卷尺：
- 刻度范围广（更多低频分量），能表示 0~100m 的距离
- 维度更大，编码容量更高
- 适合：区分 "100个token前" 和 "5000个token前" 的内容

### 7.3 维度差异的数学意义

Local RoPE (d=256, 128对频率):
$$\theta_i = 10000^{-2i/256}, \quad i = 0, ..., 127$$

Global RoPE (d=512, 256对频率):
$$\theta_i = \text{base}^{-2i/512}, \quad i = 0, ..., 255$$

Global 的 256 对频率中：
- 前 128 对覆盖 Local 的频率范围
- **额外的 128 对提供更低的频率**（更慢的旋转 → 编码更远的距离）

### 7.4 在 Gemma 4 模型中的体现

```
模型输入:
  pos_emb_local_cos  [1, seq_len, 1, 256]  ← 给 Local 层的 Q/K 用
  pos_emb_local_sin  [1, seq_len, 1, 256]
  pos_emb_cos        [1, seq_len, 1, 512]  ← 给 Global 层的 Q/K 用
  pos_emb_sin        [1, seq_len, 1, 512]
```

注意这些是**模型输入**而非内部计算的，意味着位置编码在模型外部预计算好再传入。这样做的好处：
- 可以在推理时动态选择不同的 RoPE 缩放策略
- 支持 NTK-aware scaling、YaRN 等长度外推方法，无需改模型结构

### 7.5 Local vs Global 层的 Attention 区别

**Local 层** (layer 0~4, 6~10, ...):
```
Q: [1, 128, 16, 256]  →  16 heads, head_dim=256
K: [1, 128,  8, 256]  →   8 KV heads (GQA ratio=2)
V: [1, 128,  8, 256]

RoPE 用 local_cos/sin (256d) 分别应用到 Q 和 K
Attention 范围: 滑动窗口 (KV cache 8191 + new 128 = 8319)
```

**Global 层** (layer 5, 11, 17, ...):
```
Q: [1, 128, 16, 512]  →  16 heads, head_dim=512
K: [1, 128,  1, 512]  →   1 KV head (GQA ratio=16!)  
V: [1, 128,  1, 512]

RoPE 用 global cos/sin (512d) 分别应用到 Q 和 K
Attention 范围: 全序列 (所有历史 token)
```

---

## 8. 总结对比

```
┌────────────────────────────────────────────────────────────┐
│              标准 Transformer (如 Llama 3)                   │
├────────────────────────────────────────────────────────────┤
│  所有层:  同一套 RoPE, 同样的 head_dim                        │
│  Layer 0: Q[16h×128d] @ K[16h×128d]  + RoPE(128d)         │
│  Layer 1: Q[16h×128d] @ K[16h×128d]  + RoPE(128d)         │
│  ...                                                       │
│  Layer N: Q[16h×128d] @ K[16h×128d]  + RoPE(128d)         │
│  所有层都看全序列, 位置编码一模一样                              │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│              Gemma 4 (Dual RoPE + Hybrid Attention)         │
├────────────────────────────────────────────────────────────┤
│  Local层: Q[16h×256d] @ K[8h×256d]  + Local_RoPE(256d)    │
│           滑动窗口, 高频位置编码, 只看附近                      │
│                                                            │
│  Global层: Q[16h×512d] @ K[1h×512d] + Global_RoPE(512d)   │
│            全序列, 低频位置编码, 看所有历史                      │
│                                                            │
│  排列: L L L L L G  (每6层1个Global)                        │
│                                                            │
│  效果: Local层高效处理局部依赖                                 │
│        Global层偶尔"抬头看远方", 捕获长程依赖                   │
│        整体上比全部用 full attention 节省大量 KV cache          │
└────────────────────────────────────────────────────────────┘
```

---

## 9. KV Cache 节省分析

这个设计极大地减少了 KV cache 占用：

| | 标准 (全 full attention) | Gemma 4 Hybrid |
|---|---|---|
| Local 层 KV | — | 8 heads × 256d × seq | 
| Global 层 KV | — | 1 head × 512d × seq |
| 如果全用 Full attn | 16h × 256d × seq × 47层 | — |
| 实际 Gemma 4 | — | (8×256×40 + 1×512×7) × seq |
| KV cache 比例 | 100% | **~54%** |

通过让 Global 层用极端的 GQA (16:1)、加上只有 7/47 的层是 Global，Gemma 4 在保持长程依赖能力的同时大幅压缩了 KV cache。
