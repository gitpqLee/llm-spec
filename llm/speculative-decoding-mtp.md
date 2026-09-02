# Speculative Decoding 与 MTP 技术详解

> 从自回归解码瓶颈出发，理解 Speculative Decoding 的候选生成与并行验证机制，以及 MTP 模型的结构、训练方法和工程落地。

---

## 目录

1. [背景：自回归解码为什么慢](#一背景自回归解码为什么慢)
2. [Speculative Decoding 的核心思想](#二speculative-decoding-的核心思想)
3. [Target Model 如何并行验证候选](#三target-model-如何并行验证候选)
4. [候选接受算法](#四候选接受算法)
5. [MTP 是什么](#五mtp-是什么)
6. [MTP 模型架构](#六mtp-模型架构)
7. [MTP 的训练方法](#七mtp-的训练方法)
8. [MTP 推理与 Speculative Decoding 的结合](#八mtp-推理与-speculative-decoding-的结合)
9. [MTP 与 Speculative Decoding 的异同](#九mtp-与-speculative-decoding-的异同)
10. [KV-Cache 与状态管理](#十kv-cache-与状态管理)
11. [性能收益与限制](#十一性能收益与限制)
12. [常见方案对比](#十二常见方案对比)
13. [模型、导出、Pipeline 与硬件的协作](#十三模型导出pipeline-与硬件的协作)
14. [常见误区](#十四常见误区)
15. [总结](#十五总结)

---

## 一、背景：自回归解码为什么慢

Decoder-only 大模型按如下条件概率逐 Token 生成：

$$
P(x_{1:T})=\prod_{t=1}^{T}P(x_t\mid x_{\lt t})
$$

生成第 $t+1$ 个 Token 前必须先得到第 $t$ 个 Token：

```text
上下文 → Target Forward → Token 1
上下文 + Token 1 → Target Forward → Token 2
上下文 + Token 1 + Token 2 → Target Forward → Token 3
```

KV-Cache 避免了重复计算历史 Token 的 Key 和 Value，但没有消除 Token 之间的串行依赖。Decode 阶段每次通常只有一个 Query Token：

```text
Q: [B, Hq, 1, Dh]
K: [B, Hkv, T, Dh]
V: [B, Hkv, T, Dh]
```

这会带来三个问题：

1. 每生成一个 Token 都要调度一次完整模型。
2. 单 Token MatMul 较窄，GPU/NPU 计算单元利用率可能不足。
3. 模型权重读取、Kernel 启动和设备同步成本难以摊薄。

因此，自回归生成常常是内存带宽和调度受限，而不是纯算力受限。

---

## 二、Speculative Decoding 的核心思想

Speculative Decoding（推测解码）使用一个便宜的候选生成器先起草多个 Token，再由准确的 Target Model 一次验证整段候选。

```text
                 ┌──────────────────────┐
已有上下文 ──────▶│ Draft Model / MTP    │
                 └──────────┬───────────┘
                            │ 候选 [d1, d2, d3, d4]
                            ▼
                 ┌──────────────────────┐
已有 Target Cache▶│ Target Model         │
                 │ 一次并行验证整段候选 │
                 └──────────┬───────────┘
                            ▼
                  接受最长有效前缀
```

它将原来的多次 Target 调用：

```text
Target(d1) → Target(d2) → Target(d3) → Target(d4)
```

变成：

```text
Cheap Draft 产生 d1...d4
Target([d1, d2, d3, d4]) 一次验证
```

关键点是：

- Draft 只负责提出候选，没有最终决定权。
- Target 仍然定义最终输出分布。
- 候选越准确，一次 Target Forward 推进的 Token 越多。
- 标准接受算法可以做到相对 Target Model 的无损加速。

候选来源并不固定，可以是：

| 候选来源 | 说明 |
|---|---|
| 独立 Draft Model | 一个更小、更快的语言模型 |
| MTP Module | Target 模型附带的多 Token 预测模块 |
| Medusa Heads | 多个并行预测头，通常生成候选树 |
| EAGLE Module | 预测未来 Hidden Feature，再映射到 Token |
| Prompt Lookup | 从 Prompt 或已生成文本中匹配重复片段 |
| N-gram | 根据历史 N-gram 提出候选 |

所以 Speculative Decoding 是一套推理解码协议，而不是某一种固定模型结构。

---

## 三、Target Model 如何并行验证候选

### 3.1 一个具体例子

已有确认上下文：

```text
我 喜欢 吃
```

Draft 产生 4 个候选：

```text
[苹果, 和, 香蕉, 。]
```

Target 不需要逐个运行 4 次，而是把候选作为一个短序列输入：

```text
Target Forward(
    past_kv = cache("我 喜欢 吃"),
    input_ids = [苹果, 和, 香蕉, 。]
)
```

Target 一次产生 4 组新 logits：

```text
苹果位置的 logits：预测“苹果”之后的 Token
和位置的 logits：预测“苹果 和”之后的 Token
香蕉位置的 logits：预测“苹果 和 香蕉”之后的 Token
。位置的 logits：预测整段候选之后的额外 Token
```

### 3.2 One-token shift

验证时必须注意 logits 与候选之间错开一个位置：

| Target 概率分布 | 验证对象 |
|---|---|
| 上一轮已有的 $p(x_{t+1}\mid x_{\leq t})$ | $d_1$ |
| 输入 $d_1$ 后得到的分布 | $d_2$ |
| 输入 $d_2$ 后得到的分布 | $d_3$ |
| 输入 $d_3$ 后得到的分布 | $d_4$ |
| 输入 $d_4$ 后得到的分布 | Bonus Token |

可以写成：

$$
p_i=P(x_{t+i}\mid x_{\leq t},d_{\lt i})
$$

其中 $p_1$ 通常来自验证前已经保存的 Target logits，而本次 Forward 最后一个位置的 logits 可以产生额外的 Bonus Token。

### 3.3 为什么一次 Forward 不违反因果性

Transformer 对多个候选位置同时构造 $Q/K/V$，但使用 Causal Mask：

```text
Query \ Key    历史    苹果    和    香蕉    。
苹果             ✓       ✓      ✗      ✗      ✗
和               ✓       ✓      ✓      ✗      ✗
香蕉             ✓       ✓      ✓      ✓      ✗
。               ✓       ✓      ✓      ✓      ✓
```

候选内部的 Attention Mask 是下三角矩阵：

$$
M_{ij}=
\begin{cases}
0, & j\leq i\\
-\infty, & j\gt i
\end{cases}
$$

Attention 仍然满足：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}+M\right)V
$$

因此不同位置在硬件上可以组成一个较大的矩阵运算，但每个位置在语义上只能看到历史和它前面的候选。这叫计算并行，不是条件依赖并行。

### 3.4 利用已有 KV-Cache

Target 不会重新计算整个 Prompt：

```text
Past KV: [我, 喜欢, 吃]
New  KV: [苹果, 和, 香蕉, 。]
Query:   [苹果, 和, 香蕉, 。]
```

验证相当于一次长度为 4 的短 Prefill。相比 4 次单 Token Decode，它有更好的矩阵形状，并能摊薄权重读取与调度成本。

---

## 四、候选接受算法

### 4.1 Greedy Decoding

在 Greedy 模式下，可以直接比较 Draft Token 与 Target argmax：

```text
Draft:  [苹果, 和, 香蕉, 。]
Target: [苹果, 和, 草莓, 很]
```

从左向右：

```text
苹果 == 苹果  → 接受
和   == 和    → 接受
香蕉 != 草莓  → 拒绝并停止
```

只接受最长连续匹配前缀 `[苹果, 和]`。不能继续检查并接受后面的 `。`，因为它是在“香蕉正确”这个已经失败的条件下生成的。

通常可使用 Target 在第一个失败位置给出的 Token `草莓` 进行修正：

```text
最终推进：[苹果, 和, 草莓]
```

如果所有 Draft Token 都通过，还可以从最后一组 Target logits 中选择一个 Bonus Token：

```text
4 个 Draft 全部接受 + 1 个 Target Bonus Token
```

### 4.2 Sampling Decoding

随机采样时，简单比较 argmax 会改变 Target 的概率分布。标准 Speculative Sampling 对候选 $x$ 使用接受概率：

$$
a(x)=\min\left(1,\frac{p(x)}{q(x)}\right)
$$

其中：

- $q(x)$：Draft Model 对候选 $x$ 的概率。
- $p(x)$：Target Model 对同一候选的概率。

如果候选被拒绝，则从修正分布采样：

$$
p'(x)=
\frac{\max(0,p(x)-q(x))}
{\sum_y\max(0,p(y)-q(y))}
$$

直觉如下：

- Draft 对某个 Token 的概率不高于 Target 时，候选总能被接受。
- Draft 过度偏好某个 Token 时，只按 $p/q$ 的比例接受。
- 被拒绝后，从 Target 尚未被 Draft 覆盖的概率质量中补采样。

这样可以保证最终 Token 分布与直接从 Target 采样一致。这里的“无损”是概率分布不变，不代表每次随机运行都生成完全相同的文本。

---

## 五、MTP 是什么

MTP（Multi-Token Prediction，多 Token 预测）让模型从当前上下文预测多个未来 Token，而普通 Next-Token Prediction 只优化：

$$
P(x_{t+1}\mid x_{\leq t})
$$

MTP 还优化更远位置：

$$
P(x_{t+2}\mid x_{\leq t}),
P(x_{t+3}\mid x_{\leq t}),\ldots
$$

或者使用串联条件：

$$
P(x_{t+k+1}\mid x_{\leq t},x_{t+1:t+k})
$$

MTP 定义的是一种预测能力和训练目标，并不规定唯一的网络结构。它可以只用于训练，也可以在推理阶段作为 Speculative Decoding 的 Draft 生成器。

---

## 六、MTP 模型架构

### 6.1 普通 Decoder-only 模型

```text
Token IDs
    ↓
Embedding
    ↓
Transformer Backbone × N
    ↓
Final Norm
    ↓
LM Head
    ↓
预测 x(t+1)
```

### 6.2 串联式 MTP 架构

原生 MTP 模型通常保留普通主干，并在主干末端增加轻量预测模块：

```text
Transformer Backbone
    │
    ├── LM Head ───────────────→ x(t+1)
    │
    └── hidden h(0)
          + embedding(x(t+1))
                  ↓
             MTP Block 1
                  ├── LM Head → x(t+2)
                  │
                  └── hidden h(1)
                        + embedding(x(t+2))
                                ↓
                           MTP Block 2
                                └── LM Head → x(t+3)
```

一个典型 MTP Block 包含：

1. 对前一级 Hidden State 做 RMSNorm。
2. 对前一个未来 Token 的 Embedding 做 RMSNorm。
3. Concat 两路特征。
4. Linear Projection 将 $2D$ 投影回 $D$。
5. 一个或少量 Transformer Block。
6. Final Norm 与 LM Head。

第 $k$ 个模块可表示为：

$$
z_t^{(k)}=
W_k\left[
\operatorname{Norm}(h_t^{(k-1)});
\operatorname{Norm}(E(x_{t+k}))
\right]
$$

$$
h_t^{(k)}=\operatorname{MTPBlock}_k(z_t^{(k)})
$$

$$
p_{t+k+1}=\operatorname{softmax}(W_{vocab}h_t^{(k)})
$$

Embedding 和 LM Head 经常与主模型共享，从而控制参数增量。

### 6.3 独立多 Head 架构

更简单的方式是从同一个主干 Hidden State 直接接多个预测头：

```text
hidden h_t
   ├── Head 1 → x(t+1)
   ├── Head 2 → x(t+2)
   ├── Head 3 → x(t+3)
   └── Head 4 → x(t+4)
```

$$
p_k=\operatorname{softmax}(W_kh_t)
$$

这种结构计算便宜，但远期 Head 不知道前面实际选中了什么 Token，因此距离越远通常越不准确。

### 6.4 树状候选架构

Medusa 等方案会让每个 Head 给出多个候选，组合成候选树：

```text
             A
          /     \
         B       C
       /  \     / \
      D    E   F   G
```

Target 使用专门的 Tree Attention Mask 一次验证多条路径。这可以提高命中概率，但候选组织、位置编码、Cache 提交和验证逻辑更复杂。

### 6.5 Feature-level MTP

EAGLE 类方法不直接独立预测所有 Token，而是先预测未来 Hidden Feature：

```text
Target Hidden State
        ↓
轻量 Feature Predictor
        ↓
未来 Hidden Feature
        ↓
共享 LM Head
        ↓
Draft Token
```

它利用 Target 的特征空间，通常比完全独立的小语言模型更接近 Target。

---

## 七、MTP 的训练方法

训练时完整序列已知，所以 MTP Module 一般使用 Teacher Forcing：

```text
主干使用真实上下文       → 预测真实 x(t+1)
MTP 1 输入真实 x(t+1)   → 预测真实 x(t+2)
MTP 2 输入真实 x(t+2)   → 预测真实 x(t+3)
```

总损失通常为：

$$
\mathcal{L}
=\mathcal{L}_{NTP}
+\lambda\sum_{k=1}^{K}\mathcal{L}_{MTP}^{(k)}
$$

其中：

$$
\mathcal{L}_{MTP}^{(k)}
=-\sum_t\log P(x_{t+k+1}\mid x_{\leq t+k})
$$

$\lambda$ 控制辅助损失权重。MTP Loss 的梯度通常会进入主干，因此即使推理时移除 MTP Module，主干也可能受益于更丰富的未来监督信号。

训练与推理存在一个差异：

| 阶段 | MTP Module 的 Token 输入 |
|---|---|
| 训练 | Ground Truth Token，Teacher Forcing |
| 推理 | 上一级预测出的 Token |

因此推理时错误可能逐级传播，这也是候选最终必须由 Target 验证的原因。

---

## 八、MTP 推理与 Speculative Decoding 的结合

MTP 与 Speculative Decoding 结合后的典型循环如下：

```text
1. Target 根据已确认上下文产生可靠 Token 和 Hidden State
2. MTP Module 基于这些状态生成后续 Draft Token
3. Target 把整段候选作为短序列一次 Forward
4. Pipeline 计算最长可接受前缀
5. 提交已接受状态，裁剪或恢复被拒绝状态
6. MTP 从新的已确认位置生成下一批候选
```

例如：

```text
已确认上下文：我 喜欢 吃

Target 起点：苹果
MTP Draft： 和 香蕉 。
完整候选：[苹果, 和, 香蕉, 。]

Target 验证：[苹果, 和, 草莓, 很]
接受：[苹果, 和]
修正：[草莓]

新上下文：我 喜欢 吃 苹果 和 草莓
```

Target 验证 Forward 会同时产生多个位置的 logits，但这些 logits 主要用于验证当前候选。下一轮的一批 Draft 通常仍由 MTP Module 基于更新后的 Target 状态产生。

---

## 九、MTP 与 Speculative Decoding 的异同

| 维度 | MTP | Speculative Decoding |
|---|---|---|
| 本质 | 模型结构或训练目标 | 推理解码算法 |
| 核心职责 | 预测多个未来 Token | 生成、验证、接受和修正候选 |
| 是否要求训练 | 通常需要 | 协议本身不需要重新训练 Target |
| 是否需要 Target 验证 | 只用于辅助训练时不需要 | 必须 |
| 是否需要独立 Draft Model | 不一定 | 不一定，MTP/N-gram 也可生成候选 |
| 是否天然无损 | 直接输出 MTP 结果不保证 | 标准接受算法可保持 Target 分布 |
| 是否可以单独存在 | 可以 | 可以不依赖 MTP |

两者关系可以概括为：

> MTP 负责高效地产生候选，Speculative Decoding 负责安全地使用候选。

以下情况都成立：

```text
MTP 训练 + 普通逐 Token 推理          → 有 MTP 训练，但没有推测加速
普通 Target + 独立 Draft Model        → 有推测加速，但没有模型内 MTP
MTP Module + Target 并行验证          → MTP-assisted Speculative Decoding
```

---

## 十、KV-Cache 与状态管理

### 10.1 验证期间的临时 Cache

假设验证前 Target Cache 长度为 $T$，候选长度为 $K$：

```text
验证前：长度 T
验证后临时状态：长度 T + K
实际接受 A 个：只保留到 T + A
```

如果使用 Target 修正 Token或 Bonus Token，还需要提交相应的新状态。

### 10.2 KV-Cache 裁剪

标准 Attention 的 KV-Cache 带有显式序列维度：

```text
[Batch, KV Heads, Sequence, Head Dim]
```

拒绝候选时可以沿 Sequence 维裁剪：

```python
k_cache = k_cache[:, :, :accepted_length, :]
v_cache = v_cache[:, :, :accepted_length, :]
```

工程实现还可能使用预分配 Paged KV-Cache，此时通常调整 Block Table、逻辑长度或释放未提交页面，而不是复制整个 Cache。

### 10.3 线性注意力与循环状态

Gated DeltaNet、Mamba 等模型将历史压缩为固定尺寸状态：

$$
s_t=f(s_{t-1},x_t)
$$

一次验证多个候选后只得到最终状态 $s_{t+K}$，通常无法通过简单切片恢复 $s_{t+A}$。常见解决方案包括：

1. 验证前保存旧状态，在拒绝后重算已接受部分。
2. 为每个候选位置保存中间状态，增加显存开销。
3. 使用事务式临时状态，验证成功后再提交。
4. 限制或关闭带不可逆循环状态模型的 Speculative Decoding。

因此，混合 Full Attention 与 Linear Attention 的模型比纯 Transformer 更难支持推测解码。

### 10.4 多分支候选的 Cache

树状候选需要维护逻辑分支：

```text
历史 Cache
   ├── A → B → D
   ├── A → B → E
   └── A → C → F
```

通常不能为每条路径完整复制 KV-Cache，而要依赖共享前缀、Tree Attention、索引重排或 Paged Cache。

---

## 十一、性能收益与限制

### 11.1 平均接受长度

若一次提出 $K$ 个候选，第 $i$ 个候选在前缀仍正确的条件下通过概率为 $\alpha_i$，平均接受 Draft 数近似为：

$$
E[A]=\sum_{i=1}^{K}\prod_{j=1}^{i}\alpha_j
$$

若每个位置通过率都近似为 $\alpha$：

$$
E[A]=\sum_{i=1}^{K}\alpha^i
=\frac{\alpha(1-\alpha^K)}{1-\alpha}
$$

例如 $K=4$、$\alpha=0.8$：

$$
E[A]=0.8+0.64+0.512+0.4096=2.3616
$$

这说明候选长度翻倍不一定使有效推进长度翻倍，因为远端 Token 必须建立在整个前缀正确的基础上。

### 11.2 简化延迟模型

设：

- $C_T(1)$：Target 单 Token Decode 成本。
- $C_T(K)$：Target 一次验证 $K$ 个 Token 的成本。
- $C_D(K)$：产生 $K$ 个 Draft 的成本。
- $E[S]$：一轮最终推进的平均 Token 数，包含可能的修正或 Bonus Token。

每个输出 Token 的平均成本约为：

$$
C_{spec/token}\approx\frac{C_D(K)+C_T(K)}{E[S]}
$$

相对普通解码的理想加速比为：

$$
\operatorname{Speedup}\approx
\frac{C_T(1)\cdot E[S]}{C_D(K)+C_T(K)}
$$

只有当：

$$
C_D(K)+C_T(K)\lt C_T(1)\cdot E[S]
$$

推测解码才真正加速。

### 11.3 什么场景收益高

- Draft 与 Target 分布接近，接受率高。
- 代码、固定格式、重复文本等低熵任务。
- Target 很大而 Draft/MTP 很轻。
- 硬件执行单 Token Decode 利用率低，而短 Prefill 利用率高。
- Batch 较小，调度和权重带宽成本占比较高。

### 11.4 什么场景收益低

- 开放式创作或高 Temperature 导致候选不稳定。
- Draft 过大，生成候选本身很贵。
- 候选太长，尾部通过概率很低。
- Target 的多 Token 验证图效率不好。
- 大 Batch 已充分利用硬件，验证额外 Token 反而增加竞争。
- 状态回滚或数据搬运成本过高。

### 11.5 候选长度不是越大越好

候选长度 $K$ 增大时：

- Target 调用次数可能下降。
- 远端候选命中率下降。
- 验证计算量和临时 Cache 增大。
- 拒绝后浪费的计算增多。

实际系统通常根据接受率、Batch、序列长度和设备动态调整 $K$。

---

## 十二、常见方案对比

| 方案 | 候选生成方式 | 是否修改 Target | 主要特点 |
|---|---|---|---|
| Two-model Speculation | 独立小型 Draft LM | 否 | 通用，但需要额外模型和 Cache |
| Native MTP | 模型预训练时加入 MTP Module | 是 | Draft 与主干特征结合紧密 |
| Medusa | 多个 Token Head | 通常需要附加训练 | 可生成候选树，Head 较轻 |
| EAGLE | 预测未来 Hidden Feature | 需要训练 Adapter | 利用 Target 特征，接受率通常较高 |
| Self-speculation | Target 浅层或 Early Exit | 需要可提前退出的执行图 | 可共享权重，但调度复杂 |
| Prompt Lookup | 匹配已有文本片段 | 否 | 无模型成本，适合重复内容 |

典型原生或广义 MTP 系列包括 DeepSeek-V3/R1 的 MTP、Medusa、EAGLE，以及具有专用 MTP Assistant 的模型。具体 checkpoint 是否保留 MTP 权重、推理框架是否启用对应 Pipeline，需要分别确认。

---

## 十三、模型、导出、Pipeline 与硬件的协作

完整 MTP 推理支持通常涉及四层：

```text
模型 / Checkpoint
    │ 提供 MTP 权重、Hidden State 和共享参数
    ▼
模型导出
    │ 保留 MTP 输入输出、KV/SSM State 和动态形状
    ▼
推理 Pipeline
    │ Draft、验证、接受、修正、Cache 提交/回滚
    ▼
编译器与硬件后端
      高效执行单 Token Draft 和多 Token Verification
```

### 13.1 模型层

需要具备下列一种能力：

- MTP Heads。
- 串联 MTP Blocks。
- 独立 MTP Assistant。
- 可供 EAGLE/Medusa 使用的 Hidden State 接口。

### 13.2 导出层

需要正确导出：

- Target logits。
- MTP 所需 Last Hidden State。
- Token Embedding 或 Inputs Embeds 接口。
- 可共享的 KV-Cache。
- MTP 自身状态。
- 动态候选长度和正确的 Causal/Tree Mask。

如果导出图只有 `logits` 和普通 `present_key_values`，通常说明没有保留模型内 MTP 路径。

### 13.3 Pipeline 层

Pipeline 负责：

1. 调用 Draft/MTP 产生候选。
2. 记录 Draft 概率 $q$。
3. 构造 Target Verification 输入。
4. 对齐 Target logits 与候选 Token。
5. 执行 Greedy 或 Sampling 接受算法。
6. 处理 EOS、Stop Token、最大长度和流式输出。
7. 提交、裁剪或恢复 Cache。
8. 统计接受率并可选地调整候选长度。

模型即使包含 MTP 权重，如果 Pipeline 只读取主 LM Head 的最后一个 logits，仍会退化为普通逐 Token 解码。

### 13.4 编译器与硬件层

功能正确并不自动等于性能提升。后端需要关注：

- 单 Token Draft 的启动延迟。
- Verification 中 $L_q=K$ 的 MatMul/DPU 利用率。
- 动态 Shape 是否导致重复编译。
- KV-Cache 写入和裁剪是否产生额外拷贝。
- Target 与 MTP 是否频繁跨设备同步。
- 小 Batch、小 Sequence 算子是否被低效切分。

对 NPU 而言，理想实现应尽量保持 Target/MTP 状态常驻设备，并避免每轮把候选、Hidden State 和 Cache 搬回 Host。

---

## 十四、常见误区

### 误区 1：MTP 与 Speculative Decoding 是同一个东西

不是。MTP 是候选预测能力；Speculative Decoding 是验证和接受候选的推理算法。

### 误区 2：支持多 Token 输入就等于支持 MTP

不是。普通 Transformer 本来就能处理多 Token Prefill。原生 MTP 还需要专门训练的未来预测参数或模块。

### 误区 3：Target 并行验证意味着未来 Token 彼此独立

不是。Causal Mask 保留了严格的前缀依赖，只是矩阵计算能够并行执行。

### 误区 4：Target 一次输出多组 logits 就是下一轮的多个 Draft

通常不是。这些 logits 分别对应当前候选各位置的条件分布，主要用于验证；下一轮候选一般仍由 Draft/MTP 产生。

### 误区 5：MTP 预测的 Token 可以直接全部输出

不建议。越远的预测越容易出错，直接输出会改变质量和概率分布。推理加速通常仍需要 Target 验证。

### 误区 6：候选越多，加速越大

不一定。候选越长，远端命中率和有效计算比例通常越低，需要针对模型与设备调优。

### 误区 7：只要模型有 MTP 权重就能获得加速

不是。还需要导出层保留 MTP 图、Pipeline 实现验证协议、后端高效执行多 Token Verification。

---

## 十五、总结

普通自回归解码每次只能通过完整 Target Model 推进一个 Token。Speculative Decoding 用便宜的候选生成器先起草多个 Token，再让 Target 利用 Causal Mask 一次并行验证整段候选，从而减少昂贵的 Target 调用次数。

MTP 在普通语言模型主干之外增加多 Token 预测路径，可以采用独立 Heads、串联 MTP Blocks、候选树或 Feature Predictor。它既可以作为训练辅助目标，也可以在推理阶段充当 Speculative Decoding 的 Draft 生成器。

最重要的三个结论：

1. **MTP 负责猜，Speculative Decoding 负责验证和接受。**
2. **Target 的并行验证保留因果依赖，只把多个位置组成一次高效计算。**
3. **真正的 MTP 加速需要模型、导出、Pipeline、Cache 管理和硬件后端共同支持。**
