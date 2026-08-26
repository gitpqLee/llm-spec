# FlashQLA 完整解析

> 从 Gated Delta Rule 的逐 token 递推，到 chunk 内矩阵化、GPU warpgroup 流水、双缓冲和长序列 Context Parallelism。

---

## 一、先说结论

FlashQLA（Flash Qwen Linear Attention）是面向 Qwen Gated DeltaNet（GDN）的高性能 kernel 库，主要优化：

- 长序列 prefill；
- 训练 forward；
- 训练 backward。

它的核心思想可以分成三个层次：

```text
算法层：逐 token 递推
            ↓ 分块与下三角求解
        chunk 内矩阵并行、chunk 间状态递推

Kernel 层：多个独立小算子
            ↓ 融合、warpgroup 专业化、双缓冲
        数据搬运、Tensor Core、CUDA Core、SFU 重叠

调度层：少数很长的状态链
            ↓ Context Parallelism
        segment 摘要、边界状态校正、多个短状态链并行
```

最重要的判断是：

> FlashQLA 不是让整条 token 序列毫无依赖地同时运行。它先消除 chunk 内的串行依赖，再通过流水隐藏 chunk 之间的部分搬运和计算延迟；在特定长序列场景下，才进一步使用 segment 级并行。

它也不是用线性注意力近似 Softmax Attention。GDN 本身就是模型定义的线性递推算子，FlashQLA 优化的是这个算子的等价实现。

---

## 二、Gated Delta Rule 在做什么

### 2.1 固定大小的递推状态

对每个 value head，GDN 维护一个状态矩阵：

$$
S_t\in\mathbb{R}^{D_K\times D_V}
$$

当前 FlashQLA 主要针对：

$$
D_K=D_V=128
$$

可以把 $S_t$ 理解成一块固定大小的关联记忆：key 决定从哪里读取，value 是期望读取到的内容。

简化的 token 级递推为：

$$
\widetilde S_{t-1}=\alpha_t S_{t-1},\qquad \alpha_t=e^{g_t}
$$

$$
e_t=v_t-k_t\widetilde S_{t-1}
$$

$$
S_t=\widetilde S_{t-1}+\beta_t k_t^\top e_t
$$

$$
o_t=sq_tS_t
$$

其中：

| 符号 | 含义 |
|------|------|
| $g_t$ | log-space 遗忘门，通常为非正数 |
| $\alpha_t=e^{g_t}$ | 历史状态保留比例 |
| $\beta_t$ | 当前 token 的写入强度 |
| $k_t\widetilde S_{t-1}$ | 从旧记忆中预测出的 value |
| $e_t$ | 预测误差 |
| $k_t^\top e_t$ | 将误差沿 key 方向写回状态 |

Delta Rule 的关键不是把整个 $v_t$ 直接加进状态，而是只写入“旧状态没有预测对的部分”。

### 2.2 为什么天然存在串行依赖

每个 token 都需要上一个 token 的状态：

```text
S0 → token 0 → S1 → token 1 → S2 → token 2 → ...
```

即使总复杂度随序列长度线性增长，这条依赖链仍会限制 GPU 并行度。GPU 擅长的是大量规则矩阵运算，不擅长执行几万次短小且相互依赖的状态更新。

---

## 三、第一步：把序列切成 Chunk

FlashQLA 将序列切成固定大小的 chunk：

- SM90、SM100、SM103：chunk size 64；
- SM120、SM121：chunk size 32。

例如 256 个 token 被切成：

```text
[C0: token 0...63]
        ↓ S1
[C1: token 64...127]
        ↓ S2
[C2: token 128...191]
        ↓ S3
[C3: token 192...255]
```

chunk 之间仍然有状态依赖：

$$
S_{c+1}=F_c(S_c)
$$

所以分块本身并没有让所有 chunk 同时执行。真正关键的是下一步：将一个 chunk 内的 64 次 token 递推改写成矩阵运算。

---

## 四、第二步：用下三角系统消除 Chunk 内串行

### 4.1 Gate 的局部前缀和

对于一个长度为 $C$ 的 chunk，先计算：

$$
\gamma_i=\sum_{r=0}^{i}g_r
$$

从 token $j$ 到 token $i$ 的累计衰减为：

$$
G_{ij}=
\begin{cases}
e^{\gamma_i-\gamma_j},&j\le i\\
0,&j>i
\end{cases}
$$

$G$ 是一个下三角矩阵，因为 causal 计算中，token $i$ 只能依赖自己和更早的 token。

### 4.2 Chunk 内依赖写成线性方程

当前 token 的 delta 会受前面 token 已写入状态的影响。把这些依赖全部展开后，可以得到单位下三角系统：

$$
L V_d=R
$$

其中：

- $L$ 由 $K K^\top$、$\beta$ 和 causal 依赖组成；
- $R$ 是当前 chunk 的原始 value 减去输入状态产生的预测；
- $V_d$ 是消除了 chunk 内相互依赖后的 corrected value。

于是：

$$
A=L^{-1}
$$

$$
V_d=AR
$$

这一步把：

```text
token 0 → token 1 → ... → token 63
```

变成一组适合 Tensor Core 的矩阵计算：

```text
K Kᵀ
下三角求解
A @ R
```

这是算法层真正增加并行度的地方，不只是隐藏执行时间。

### 4.3 不是通用矩阵求逆

FlashQLA 知道矩阵满足以下条件：

- 固定大小；
- 单位下三角；
- causal 结构明确。

Hopper 上的专用求解大致分层为：

```text
4 个 16×16 对角块
          ↓
2 个 32×32 块
          ↓
1 个 64×64 结果
```

相比调用通用矩阵逆，这种结构更容易展开，也更适合寄存器、shared memory 和 Tensor Core。

---

## 五、一个 Chunk 的完整计算

设 $S_c$ 是进入当前 chunk 的状态，$\gamma$ 是 chunk 内 gate 前缀和。

### 5.1 先扣除旧状态的预测

$$
W=V-\mathrm{diag}(e^\gamma)KS_c
$$

代码中可以直观理解为：

```text
读取历史状态产生的预测 K @ S
                 ↓
V - prediction
                 ↓
得到还需要写入的新信息
```

### 5.2 解出 corrected value

构造带 gate 和 beta 的下三角变换：

$$
A_g=G\odot A\odot\beta
$$

然后：

$$
V_d=A_gW
$$

### 5.3 计算输出

输出拆成历史贡献和 chunk 内贡献：

$$
O=O_{\text{history}}+O_{\text{local}}
$$

历史贡献为：

$$
O_{\text{history}}
=s\mathrm{diag}(e^\gamma)QS_c
$$

chunk 内贡献为：

$$
O_{\text{local}}
=s\left[G\odot(QK^\top)\right]V_d
$$

### 5.4 更新 Chunk 末尾状态

$$
S_{c+1}
=e^{\gamma_{C-1}}S_c
+K^\top\mathrm{diag}
\left(e^{\gamma_{C-1}-\gamma}\right)V_d
$$

因此一个 chunk 完成后，只需要把固定大小的 $S_{c+1}$ 传给下一个 chunk，不需要保存整个历史 token 的 K/V。

---

## 六、复杂度和 Softmax Attention 的区别

标准 causal Softmax Attention 的主要计算量为：

$$
O(T^2D)
$$

并且 decode 时 KV cache 随序列长度增长。

GDN 使用固定状态，主要复杂度为：

$$
O(TD_KD_V)
$$

它随 token 数线性增长，状态大小为：

$$
O(D_KD_V)
$$

需要注意：“固定”指的是每个 token/chunk 的工作量和状态大小固定，不是整个序列的总计算量固定。

---

## 七、第三步：合理的 Kernel Fusion

### 7.1 未融合实现的问题

如果严格按照公式逐步启动 kernel：

```text
K @ state
→ 写回显存
→ gate 和 beta 缩放
→ 写回显存
→ A @ value
→ 写回显存
→ QKᵀ
→ 写回显存
→ 输出计算
→ 状态更新
```

会产生大量：

- kernel launch；
- 中间张量的 HBM 写回和重读；
- kernel 之间的同步等待；
- 短小算子造成的硬件利用率不足。

FlashQLA 将主要 forward 流程融合到 `fused_gdr_fwd`，让大量中间值停留在寄存器和 shared memory 中。

### 7.2 为什么没有全部塞进一个 Kernel

FlashQLA 仍保留几个主要阶段：

```text
chunk_local_cumsum
        ↓
kkt_solve
        ↓
fused_gdr_fwd
```

完全融合可能导致：

- 寄存器使用过高；
- shared memory 不够；
- occupancy 降低；
- backward 和 Context Parallelism 难以复用中间结果。

因此它追求的是“合理融合”，而不是 kernel 数量绝对最少。

---

## 八、第四步：Warpgroup 专业化

Hopper forward kernel 使用 512 个线程，即 4 个 warpgroup。它们承担不同职责：

| Warpgroup | 主要职责 |
|-----------|----------|
| 0，线程 0–127 | 保存并更新状态 $S$ |
| 1，线程 128–255 | 计算 value 修正和 $V_d$ |
| 2，线程 256–383 | 计算 $QK^\top$、gate mask 和输出 |
| 3，线程 384–511 | TMA 输入加载、输出与状态写回 |

这叫 warp specialization：不是每组线程都执行相同程序，而是让不同线程组长期负责不同类型工作。

其目的在于同时利用：

- Tensor Core：矩阵乘；
- CUDA Core：逐元素缩放、mask 和状态操作；
- SFU：指数函数；
- TMA：全局内存到 shared memory 的异步搬运。

---

## 九、第五步：双缓冲与硬件流水

### 9.1 单缓冲的问题

假设 shared memory 只有一套输入缓冲：

```text
Chunk 0: [加载][计算]
Chunk 1:             [加载][计算]
Chunk 2:                         [加载][计算]
```

计算期间不能覆盖缓冲区，因此加载单元需要等待；加载期间计算单元也可能空闲。

### 9.2 两套 Shared-memory 工作区

FlashQLA 为主要输入分配两套缓冲：

```python
q_shared[2, C, 128]
k_shared[2, C, 128]
v_shared[2, C, block_DV]
a_shared[2, C, C]
```

可以将它们称为 A、B 两张工作台：

```text
计算组使用 A 处理 Chunk 0
加载组同时把 Chunk 1 放入 B

计算组切换到 B 处理 Chunk 1
加载组同时把 Chunk 2 放入 A
```

缓冲区按下式交替：

$$
\text{buffer}=\text{chunk\_id}\bmod 2
$$

### 9.3 时间线

```text
时间 ────────────────────────────────────────────────>

加载组： [load C0][load C1]       [load C2]       [load C3]
缓冲区：     A        B               A               B

计算组：          [compute C0][compute C1][compute C2][compute C3]
缓冲区：               A           B           A           B
```

搬运没有消失，而是尽量被当前 chunk 的计算覆盖。

### 9.4 Barrier 防止读写冲突

两套缓冲仍需要同步：

```text
data_is_ready[i]：buffer i 已加载完成，可以读取
data_is_free[i]： buffer i 已使用完成，可以覆盖
```

简化逻辑为：

```python
# 加载线程组
wait(data_is_free[buffer])
load_next_chunk(buffer)
signal(data_is_ready[buffer])

# 计算线程组
wait(data_is_ready[buffer])
compute_chunk(buffer)
signal(data_is_free[buffer])
```

### 9.5 双缓冲理论上能节省多少

设每个 chunk：

- 加载时间为 $L$；
- 计算时间为 $C$；
- 总 chunk 数为 $N$。

单缓冲理想时间：

$$
T_{\text{single}}=N(L+C)
$$

双缓冲理想时间：

$$
T_{\text{double}}=L+C+(N-1)\max(L,C)
$$

当 $N$ 很大：

$$
\text{Speedup}\approx\frac{L+C}{\max(L,C)}
$$

$$
\text{节省比例}\approx
\frac{\min(L,C)}{L+C}
$$

例如 100 个 chunk：

| $L$ | $C$ | 单缓冲 | 理想双缓冲 | 节省 |
|----:|----:|-------:|-----------:|-----:|
| 5 us | 5 us | 1000 us | 505 us | 49.5% |
| 3 us | 7 us | 1000 us | 703 us | 29.7% |
| 2 us | 8 us | 1000 us | 802 us | 19.8% |

所以双缓冲只有在加载和计算时间接近时才可能接近减半。实际收益还会受到 barrier、状态依赖、shared memory 和寄存器压力影响。

### 9.6 双缓冲没有消除什么

它没有消除 chunk 状态依赖：

$$
S_{c+1}=F_c(S_c)
$$

因此不能把同一条状态链上的所有 chunk 随意同时更新。它主要重叠的是：

```text
当前 chunk 的计算
        +
下一个 chunk 的数据加载
```

---

## 十、FlashQLA 是否是 Memory Bound

不能笼统地说整个 kernel 都是 memory bound，它是混合型 workload：

| 阶段 | 可能的主要瓶颈 |
|------|----------------|
| $QK^\top$、$KS$、$AV$、$K^\top V$ | Tensor Core / compute |
| gate、mask、缩放 | CUDA Core |
| 指数计算 | SFU |
| 输入和中间数据搬运 | memory bandwidth / latency |
| chunk 状态传递 | dependency / synchronization |
| batch 和 head 太少 | occupancy |

但原始分解实现确实容易受到访存影响，因为多个短小算子反复把中间结果写回显存。

FlashQLA 的应对顺序是：

1. 先通过融合减少不必要的全局内存流量；
2. 再通过双缓冲隐藏仍然不可避免的输入加载；
3. 用 warpgroup 专业化让搬运、GEMM、逐元素操作和指数计算尽量重叠。

---

## 十一、第六步：长序列 Context Parallelism

### 11.1 为什么还需要 Segment 并行

普通模式的并行任务通常来自 batch 和 value head：

$$
\text{parallelism}\approx B\times H_V
$$

在 TP 后 head 很少、batch 又小时，GPU 可能只有几条很长的状态链：

```text
Head 0: C0 → C1 → C2 → ... → C511
Head 1: C0 → C1 → C2 → ... → C511
...
Head 7: C0 → C1 → C2 → ... → C511
```

这些任务不足以占满大量 SM。

### 11.2 将长序列进一步切成 Segment

```text
Segment 0: C0   ... C127
Segment 1: C128 ... C255
Segment 2: C256 ... C383
Segment 3: C384 ... C511
```

每个 segment 对输入状态的作用可以表示为仿射变换：

$$
S_{\text{out}}=H_i+M_iS_{\text{in}}
$$

其中：

- $H_i$：输入状态为零时，该 segment 自己产生的状态；
- $M_i$：输入状态通过该 segment 后的传播矩阵。

每个 segment 的 $H_i,M_i$ 只依赖本段 token，因此可以并行计算：

```text
Segment 0 → H0, M0
Segment 1 → H1, M1
Segment 2 → H2, M2
Segment 3 → H3, M3
```

### 11.3 校正 Segment 边界状态

若整条序列初态为 $S_0$：

$$
S_1=H_0+M_0S_0
$$

$$
S_2=H_1+M_1S_1
$$

$$
S_3=H_2+M_2S_2
$$

这一步仍有依赖，但依赖只存在于少数 segment 边界，不再需要先串行完成每个 chunk 的完整输出。

得到正确的 $S_0,S_1,S_2,S_3$ 后，各段可以并行执行主计算：

```text
Segment 0 使用 S0 ─┐
Segment 1 使用 S1 ─┼─ 并行计算 token 输出
Segment 2 使用 S2 ─┼─
Segment 3 使用 S3 ─┘
```

每个 segment 内部仍然按 chunk 顺序更新状态。

### 11.4 Gate-driven Warmup

当一段累计 gate 小于默认阈值 $-10$：

$$
e^{-10}\approx4.54\times10^{-5}
$$

说明很早的输入状态对后续影响已经很弱。FlashQLA 利用这一性质决定每个 segment 需要多少 warmup chunk；不能安全缩短时则通过完整状态传播矩阵进行校正。

CP 不会无条件开启。序列较短或 $B\times H_V$ 已能提供足够并行度时，额外的摘要和边界校正反而不划算。

---

## 十二、三种并行不要混淆

### 12.1 Chunk 内 Token 并行

通过下三角代数求解，把多个 token 的串行递推变成矩阵计算。

```text
性质：真正减少串行依赖
粒度：一个 chunk 内的 32/64 个 token
```

### 12.2 Warpgroup 流水并行

当前 chunk 的状态、value、输出计算与下一个 chunk 的加载重叠。

```text
性质：主要隐藏搬运和不同计算阶段的延迟
粒度：一个 CTA 内的线程组
```

### 12.3 Segment Context Parallelism

为每段计算状态变换摘要，再校正边界初态，让多个 segment 并行执行。

```text
性质：增加长序列、低 head 场景下的 CTA 数量
粒度：多个 chunk 组成的 segment
```

---

## 十三、Backward 优化

Backward 按相反方向传播 chunk 状态梯度：

```text
最后一个 chunk → ... → 第一个 chunk
```

主要输出包括：

$$
dQ,\ dK,\ dV,\ d\beta,\ dg,\ dS_0
$$

FlashQLA 的 backward 同样采用融合 kernel，并复用 forward 保存的：

- 下三角逆 $A$；
- CP 边界状态；
- 状态传播信息。

gate 输入在 forward 中做 chunk-local cumsum，因此 backward 最后需要 reverse cumsum 将累计 gate 的梯度恢复到原始逐 token gate 梯度。

当前 SM120/SM121 路径仅实现 forward，未实现 backward。

---

## 十四、Prefill 与 Decoding

### 14.1 Prefill

Prefill 一次有大量 token，因此可以：

- 组成 32/64-token chunk；
- 将 chunk 内计算转换为大矩阵运算；
- 使用双缓冲和 warpgroup 流水；
- 长序列时启用 segment CP。

这是 FlashQLA 的主要目标。

### 14.2 Decoding

逐 token decoding 每步只有一个新 token：

$$
S_t=F_t(S_{t-1}),\qquad o_t=q_tS_t
$$

此时没有足够 token 构成 chunk，因此不能利用：

- chunk 内下三角矩阵化；
- $64\times64$ 专用求解；
- segment context parallelism；
- 多 chunk 主流水。

GDN 算法本身仍然适合 decoding，因为状态大小不随上下文长度增长；但 FlashQLA 这个仓库重点不是单 token recurrent decoding kernel。

---

## 十五、GVA：Q/K Head 与 Value Head 可以不同

FlashQLA 输入形状为：

```text
Q:    [B, T, Hq, 128]
K:    [B, T, Hq, 128]
V:    [B, T, Hv, 128]
g:    [B, T, Hv]
beta: [B, T, Hv]
```

当 $H_V>H_Q$ 时使用 Grouped Value Attention（GVA）：多个 value head 共享一个 Q/K head。

这在 TP 后尤其重要，因为每张 GPU 上的 Q/K head 数可能很少，也正是 Context Parallelism 更容易产生收益的场景。

---

## 十六、数值精度策略

仓库测试采用：

- Q/K/V 和主要中间矩阵：BF16；
- gate、beta：FP32；
- 状态和累加：FP32；
- reference：FP64；
- 整体向量相对误差阈值：2%。

测试覆盖：

- 固定长度和 variable-length；
- 不同 Q/K 与 value head 比例；
- 初始状态存在或不存在；
- $[K,V]$ 与 $[V,K]$ 两种状态布局；
- 长序列；
- 多次运行确定性。

Fast math 和 `exp2(x * log2(e))` 用于提高指数计算效率，但核心算法仍是相同 GDN 算子的数值实现，不是替换成另一种注意力近似。

---

## 十七、性能特征与适用边界

H200 benchmark 中，Qwen 397B/122B TP8 的 forward 数据为：

| 序列长度 | FlashQLA | FLA | 相对 FLA |
|---------:|---------:|----:|---------:|
| 2048 | 0.061 ms | 0.081 ms | 1.33x |
| 8192 | 0.119 ms | 0.234 ms | 1.96x |
| 16384 | 0.184 ms | 0.455 ms | 2.47x |
| 32768 | 0.320 ms | 0.902 ms | 2.81x |

可以看出：

- 序列越长，固定开销越容易摊薄；
- head 越少，CP 对 occupancy 的改善越明显；
- 相对 FLA 的收益通常明显；
- 相对 FlashInfer 并不是所有 shape 都领先。

因此“2–3x”应理解为特定硬件和 workload 下相对指定基线的结果，而不是所有输入和所有实现上的固定加速比。

---

## 十八、硬件和实现限制

- 支持 SM90、SM100、SM103、SM120、SM121；
- CUDA 12.8 或更高；
- PyTorch 2.8 或更高；
- 依赖 TileLang；
- 当前主要要求 $D_K=D_V=128$；
- chunk size 是架构相关固定值；
- SM120/SM121 暂无 backward；
- CP 具有额外准备和边界校正成本，不适用于所有 shape。

---

## 十九、与 NPU Chunk Gated Delta 实现的关系

NPU 编译器中的 `ChunkGatedDeltaRuleLoop` 与 FlashQLA 的大方向相同：

```text
逐 token 状态递推
        ↓
chunk 内矩阵化
        ↓
chunk 间传递状态
```

共同点包括：

- gate 前缀和；
- causal 衰减矩阵；
- $K K^\top$ 依赖矩阵；
- 下三角求解；
- corrected value；
- chunk 输出和末尾状态更新。

但执行设计不同：

| 方面 | NPU IE Pass / GatedDeltaNet | FlashQLA |
|------|-----------------------------|----------|
| 层级 | 编译器图变换或 SW kernel | GPU kernel 库 |
| 矩阵计算 | IE MatMul lowering 或 DPU | Tensor Core |
| 控制工作 | Loop / SHAVE | CUDA warpgroup |
| Chunk 间状态 | 串行 | 默认串行 |
| 双缓冲流水 | 依赖 NPU 调度实现 | kernel 内显式 TMA/barrier |
| Segment CP | 当前没有 | 有 |
| Backward | 当前路径不覆盖 | 专用 fused backward |

还需要注意，旧 `ChunkGatedDeltaRuleLoop` pass 注释和匹配的 recurrence 与 FlashQLA 在 decay 应用于 delta 预测状态的位置上存在差异；新的融合 `GatedDeltaNet` SW kernel 数学语义更接近 FlashQLA。

所以应当总结为：

> 两者共享“chunk 内并行、chunk 间递推”的算法方向，但硬件映射、融合方式、跨 segment 并行能力以及部分递推语义并不完全相同。

---

## 二十、源码阅读路线

建议按以下顺序阅读 FlashQLA：

1. `README.md`：项目目标、硬件范围和 benchmark；
2. `tests/ref_gdr.py`：最容易理解的 PyTorch 数学参考实现；
3. `flash_qla/ops/gated_delta_rule/chunk/__init__.py`：forward/backward 调用链；
4. `hopper/kkt_solve.py`：固定下三角求解；
5. `hopper/fused_fwd.py`：warpgroup 分工、双缓冲和 barrier；
6. `cp_context.py`：自动 CP 决策与 segment 预处理；
7. `hopper/cp_fwd.py`：边界初态校正；
8. `tests/test_gdr_unit.py`：shape、精度与确定性测试；
9. `benchmark/benchmark_results_H200.txt`：性能适用范围。

---

## 二十一、最终总结

FlashQLA 的优化不是单一技巧，而是一条完整链路：

```text
1. GDN 原始逐 token 状态递推
                 ↓
2. 固定大小 chunk
                 ↓
3. 下三角代数消除 chunk 内串行依赖
                 ↓
4. 将主要计算转换为 Tensor Core GEMM
                 ↓
5. 融合状态、value 修正和输出计算
                 ↓
6. Warpgroup 专业化利用不同硬件单元
                 ↓
7. Shared-memory 双缓冲隐藏输入加载
                 ↓
8. 长序列低 head 时使用 segment CP
                 ↓
9. Forward/Backward 共同优化
```

最精炼的描述是：

> FlashQLA 先通过数学重排获得 chunk 内并行，再通过融合和双缓冲提高单个 CTA 的硬件利用率，最后通过状态仿射摘要与边界校正增加长序列场景下的 CTA 级并行度。
