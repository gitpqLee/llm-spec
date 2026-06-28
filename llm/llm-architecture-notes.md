# 大语言模型架构学习笔记

> 面向开发者 / 编译器工程师的速查手册，从整体结构到算子级实现一路打通。

---

## 目录

1. [整体结构总览](#一整体结构总览)
2. [Embedding 与 LM head（unembedding）](#二embedding-与-lm-headunembedding)
3. [Position Embedding 与 RoPE](#三position-embedding-与-rope)
4. [Transformer Block 内部](#四transformer-block-内部)
5. [Self-Attention 子模块](#五self-attention-子模块)
6. [FFN 子模块：经典 FFN vs Gated FFN](#六ffn-子模块经典-ffn-vs-gated-ffn)
7. [Attention 与 FFN 的计算量对比](#七attention-与-ffn-的计算量对比)
8. [权重在哪儿：参数量分布](#八权重在哪儿参数量分布)
9. [常见命名约定（q_proj / gate_proj / up_proj …）](#九常见命名约定q_proj--gate_proj--up_proj-)
10. [IR 层面观察：高层模块在 OpenVINO/ONNX 里长什么样](#十ir-层面观察高层模块在-openvinoonnx-里长什么样)

---

## 一、整体结构总览

现代主流的语言模型（GPT、LLaMA、Qwen、Gemma、Mistral …）几乎都是 **decoder-only Transformer**，从输入到输出的整体数据流：

```
 token ids  (整数序列,  形状 [B, T])
      │
      ▼  ┌──────────────────────────┐
         │ 1. Embedding             │   token id → 向量
         │    nn.Embedding(V, D)    │   学习的查表  V=词表, D=hidden
         └──────────────────────────┘
      ▼   x  形状 [B, T, D]
         (可选: + position embedding,或在每层用 RoPE)
      │
      ▼  ┌──────────────────────────┐
         │ 2. Transformer Block × N │   N 层重复堆叠
         │    （每个 block 内部含    │
         │     attention + FFN）    │
         └──────────────────────────┘
      ▼   h  形状 [B, T, D]
      ▼  ┌──────────────────────────┐
         │ 3. Final Norm            │   RMSNorm / LayerNorm
         └──────────────────────────┘
      ▼  ┌──────────────────────────┐
         │ 4. LM Head (= unembedding)│  线性层 D → V
         │    nn.Linear(D, V)       │  常常与 embedding 共享权重
         └──────────────────────────┘
      ▼   logits  形状 [B, T, V]
      ▼  ┌──────────────────────────┐
         │ 5. Softmax → 概率分布     │  (推理时只用最后一个 token)
         └──────────────────────────┘
                argmax / sample → 下一个 token
```

四个关键概念：

| 名字 | 形状变换 | 作用 |
|---|---|---|
| **Embedding（嵌入层 / wte）** | `[B,T] int  →  [B,T,D] float` | 把离散 token id 映射成连续向量。是一张 `V × D` 的查表矩阵。 |
| **Transformer block** | `[B,T,D] → [B,T,D]` | 模型的"重复单元"，N 层串联（GPT-2 small N=12，LLaMA-7B N=32，Gemma-12B N=48）。形状不变，只在隐空间不断"加工"。 |
| **LM head（unembedding / 输出投影）** | `[B,T,D] → [B,T,V]` | 把最终隐向量投回词表维，得到每个 token 的 raw 分数 → logits。 |
| **Logits** | `[B,T,V]` | 未归一化的对数概率。Softmax 后得到下一个 token 的概率分布。 |

---

## 二、Embedding 与 LM head（unembedding）

### 共享词表（weight tying）

LM head 和 Embedding 是否共享同一张 `V × D` 矩阵，**是一个超参数选择**，叫 **weight tying / tied embeddings**：

- **共享（tied）**：`lm_head.weight = embed_tokens.weight`
  - 优点：少一份 `V·D` 参数；理论上有正则化作用
  - 代表：GPT-2、Gemma、Llama-3.2（部分）、Qwen2.5 小尺寸
- **不共享（untied）**：两张独立的 `V × D` 矩阵
  - 优点：表达力略强；适合大模型
  - 代表：LLaMA-1/2/3 大尺寸、Mistral、Gemma 2/3 的大尺寸版本

在 HuggingFace 的 `config.json` 里看 `tie_word_embeddings: true/false`。

### 在 OpenVINO IR 里识别

- 如果 tied：LM head 的 MatMul 权重 Const 节点与 embedding 的 Gather 共享同一个 `.bin` 偏移
- 如果 untied：两个独立的大权重 blob

---

## 三、Position Embedding 与 RoPE

要分两种**完全不同**的位置编码方式：

### ① 绝对位置编码 / 可学习位置编码（GPT-2、BERT、原版 Transformer 的 sinusoidal）

```
token_emb [B,T,D]
     +
pos_emb   [T,D]     ← 一张 [max_seq, D] 的可学习矩阵或固定 sin/cos 表
     │
     ▼   [B,T,D]   送入第 1 层 transformer
```

- **位置：**在 token embedding **之后**、第一个 transformer block **之前**
- **次数：**整张前向传播**只做一次** —— 之后所有层的隐状态都已经"带位置信息"地流下去了
- **限制：**`max_seq` 是固定的，外推困难

### ② 相对位置编码 / RoPE（LLaMA、Qwen、Gemma、Mistral 全家桶）

```
每一层 attention 内部:
  Q = q_proj(x)
  K = k_proj(x)
  Q = apply_rotary(Q, cos, sin)   ← 注意：作用在 Q 和 K 上，不作用在 V 上
  K = apply_rotary(K, cos, sin)
  V = v_proj(x)
  ... 然后做 SDPA
```

- **位置：**在每层的 attention 子模块内部、做 SDPA 之前
- **次数：每一层都要做一次**（N 层就做 N 次，对 Q/K 各做一次）
- **不是加到 x 上**，而是**乘到 Q/K 上** —— RoPE 的本质是给每对 (q, k) 注入它们之间的相对距离信息
- **不动残差通道**：x（即 hidden）始终是"位置无关"的语义向量，位置信息只在 Q 与 K 做内积的那一刻被引入

### ③ ALiBi（BLOOM、MPT）

直接给 attention score 加一个随距离衰减的偏置，连 cos/sin 都不需要。

### 结论

- 有绝对位置编码 → 输入端一次，后续不做
- 用 RoPE → 没有"输入端那次"，但**每层 attention 都要做一次**

---

## 四、Transformer Block 内部

现代 LLM 几乎统一为"pre-norm + 残差"结构，每个 block 内部有两条残差子模块：

```
       x  ─────────────────────────────────────┐
       │                                       │
       ▼                                       │
   ┌─────────────┐                             │
   │  Norm 1     │  (RMSNorm 或 LayerNorm)     │
   └─────────────┘                             │
       │                                       │
       ▼                                       │
   ┌─────────────┐                             │
   │ Self-       │  Q/K/V 投影 → SDPA →        │
   │ Attention   │  Output 投影                │
   └─────────────┘                             │
       │                                       │
       ▼                                       │
       + ◀────────────────  residual add ──────┘
       │
       │ ── (此处是 attention 的输出, 仍是 [B,T,D])
       │
       ▼  ──────────────────────────────────┐
       │                                    │
       ▼                                    │
   ┌─────────────┐                          │
   │  Norm 2     │                          │
   └─────────────┘                          │
       │                                    │
       ▼                                    │
   ┌─────────────┐                          │
   │  FFN (MLP)  │  hidden → 中间维 → hidden │
   └─────────────┘                          │
       │                                    │
       ▼                                    │
       + ◀──────────── residual add ────────┘
       │
       ▼
       下一层 block 的输入
```

里面有两个核心子模块：**Self-Attention** 和 **FFN(MLP)**。

---

## 五、Self-Attention 子模块

### 张量流

```
hidden   x  : [B, T, D]
   │
   ├── Linear  W_Q  D → H·d_head      → Q  : [B, T, H_q,  d_head]
   ├── Linear  W_K  D → H_kv·d_head   → K  : [B, T, H_kv, d_head]
   └── Linear  W_V  D → H_kv·d_head   → V  : [B, T, H_kv, d_head]

   QKᵀ / sqrt(d_head) → + mask → softmax → · V → attn_out : [B, T, H·d_head]

   ↓
   Linear  W_O   H·d_head → D                   → 输出 [B, T, D]
```

### MHA / GQA / MQA 区别

只是 K/V 的头数不同：

- **MHA（Multi-Head）**: `H_q == H_kv`，每个 Q-head 都有独立 KV
- **GQA（Grouped Query）**: `H_q > H_kv`，几个 Q-head 共享一组 KV（如 16Q / 8KV）
- **MQA（Multi-Query）**: `H_kv == 1`，所有 Q-head 共享同一个 KV

GQA 是目前主流（在质量和 KV-cache 大小之间取得平衡）。

---

## 六、FFN 子模块：经典 FFN vs Gated FFN

FFN 的形状变换永远是 **D → d_ff → D**（中间维通常是 D 的 4 倍左右），但实现方式有两种主要风格。

### 6.1 经典 FFN（GPT-2 / BERT 风格） —— 只有 up / down

```
x  [B,T,D]
   │
   ▼   Linear  W_up   D → d_ff        ← "up projection" (升维)
   │
   ▼   Activation: GELU / ReLU
   │
   ▼   Linear  W_down d_ff → D        ← "down projection" (降维)
   │
   ▼
```

只有两个矩阵 → 每层 FFN 共 `2 · D · d_ff` 个参数。

### 6.2 Gated FFN（LLaMA / Gemma / Qwen / Mistral …的 SwiGLU / GeGLU）—— 有 gate / up / down 三个

把 up 投影**拆成两条并行的线性变换**，其中一条经过激活函数作为"门"，去逐元素相乘另一条：

```
x  [B,T,D]
   │
   ├──── Linear  W_gate  D → d_ff  ─→  σ(·)  ──┐
   │                                            │  element-wise
   │                                            ×  multiply
   ├──── Linear  W_up    D → d_ff  ─────────────┘
   │                                            │
   │                                            ▼  [B,T,d_ff]
   │
   ▼   Linear  W_down  d_ff → D
   │
   ▼   [B,T,D]
```

数学形式：

$$
\text{FFN}(x) = W_{\text{down}}\big(\, \sigma(W_{\text{gate}} x) \;\odot\; (W_{\text{up}} x) \,\big)
$$

`σ` 的不同选择决定了名字：

| 变种 | 激活 σ | 出现在 |
|---|---|---|
| GLU | sigmoid | 原版 |
| **GeGLU** | GELU | PaLM, Gemma |
| **SwiGLU** | SiLU/Swish | LLaMA, Mistral, Qwen |
| ReGLU | ReLU | （较少用） |

### 6.3 "gate" 到底什么意思

"gate" 这个词来自电子电路里的**逻辑门 / 开关**比喻：

- `value` (= `W_up · x`) 是要传递的"信号"
- `gate_signal` (= `σ(W_gate · x)`) 是"门控值"，决定 value 的每个元素能"通过多少"
  - 如果 gate_signal[i] ≈ 1 → value[i] 几乎全通过
  - 如果 gate_signal[i] ≈ 0 → value[i] 被压制（"门关上"）
  - 中间值 → 部分通过

这个"软开关"思想最早来自 LSTM 的 input/forget/output gate。Noam Shazeer 在 2020 年那篇 *GLU Variants Improve Transformer* 里把它套到 FFN 上，证明能改善 perplexity。

### 6.4 对比直觉

- **经典 FFN** 的非线性是"对升维后的所有通道**一视同仁**地施加非线性"
- **Gated FFN** 的非线性是"先决定**哪些通道要保留多少**，再传递" —— 表达力更强，因为它能让网络在不同 token、不同位置上**动态选择通道**

代价：
- 多一个 `D × d_ff` 矩阵 → 参数量增加 50%
- 计算量增加 50%
- 但实测 perplexity 改善明显，所以**所有现代 LLM 都改用 gated FFN**，并把 `d_ff` 减小到 `≈ 2.67·D ~ 4·D` 来部分抵消额外参数

### 6.5 三个矩阵的命名对照（HuggingFace 风格）

| 名字 | 形状 | 角色 |
|---|---|---|
| `gate_proj` (有时叫 `w1`、`gate`、`gating`) | D → d_ff | **门控分支**：算出门信号，过激活函数 |
| `up_proj`   (有时叫 `w3`、`fc_in`)        | D → d_ff | **值分支**：被门信号"按位调节" |
| `down_proj` (有时叫 `w2`、`fc_out`)       | d_ff → D | **降维投影**：把加权后的中间特征压回隐藏维 |

所以 "up/down/gate" 这三个词的含义就是：

- **up**：把 hidden 维度**升到**更大的中间维（D → d_ff）
- **down**：把中间维**降回**到 hidden 维（d_ff → D）
- **gate**：另一条**门控**分支，用来做按位乘法的"开关信号"

---

## 七、Attention 与 FFN 的计算量对比

**绝大多数情况下是 FFN 更大**，但要看 sequence length。

### 计算公式（单层、单 batch、序列长 T）

- **Attention** 的 MatMul 计算量：
  - QKV 投影：`3 · T · D²`
  - O 投影：`T · D²`
  - QKᵀ 和 attn·V：`2 · T² · D`
  - 合计 ≈ `4·T·D² + 2·T²·D`

- **FFN（Gated）** 的 MatMul 计算量：
  - gate / up / down 三个：`3 · T · D · d_ff`
  - 常见 `d_ff ≈ 3.5·D`（SwiGLU）或 `4·D`（GeGLU）
  - 合计 ≈ `10~12 · T · D²`

### 比较

```
FFN / Attention ≈ (10~12·T·D²) / (4·T·D² + 2·T²·D)
                = (10~12·D) / (4·D + 2·T)
```

- 当 `T << D`（短上下文，比如 decoding 时 T=1）：分母 ≈ 4D，比值 **≈ 2.5~3，FFN 完胜**
- 当 `T ≈ D`（中等上下文，比如 prefill T=2K, D=4K）：FFN ≈ 1.5~2× attention
- 当 `T >> D`（长上下文，T=128K, D=4K）：attention 的 `T²·D` 项开始爆炸，**attention 反过来比 FFN 大**

### 现实经验

| 场景 | 主要计算量瓶颈 |
|---|---|
| Decoding (T=1, 一次出一个 token) | **FFN**（且是 memory-bound：每生成 1 token 都要把整套权重从 DRAM 读一遍） |
| Prefill (T=1k~4k, 常见对话) | **FFN ≈ Attention**，FFN 略大 |
| 长上下文 prefill (T=32k~128k) | **Attention 主导**（Q·Kᵀ 的 T² 项） |

所以推理优化里通常 attention 关注的是 **内存带宽（KV-cache）和长序列 T² 缩放**，FFN 关注的是 **算力利用率和权重读取带宽**。

---

## 八、权重在哪儿：参数量分布

Embedding 矩阵的参数量是固定的 `V · D`（词表 × 隐维），而 transformer block 的参数量是 `N · (12·D² 左右)`（每层约 12D²，N 层）。

| 模型 | V | D | N | Embedding 占比 | Transformer 占比 |
|---|---|---|---|---|---|
| GPT-2 small (124M) | 50257 | 768 | 12 | 39 M / 124 M ≈ **31%** | 约 60% |
| LLaMA-7B | 32000 | 4096 | 32 | 131 M / 7 B ≈ **2%** | 97% |
| LLaMA-65B | 32000 | 8192 | 80 | 262 M / 65 B ≈ **0.4%** | 99% |
| Gemma 3 12B | 256000 | 3840 | 48 | 983 M / 12 B ≈ **8%** | 92% |

规律：

- **小模型**（< 1 B）embedding 占比可能高达 1/3，因为 transformer 还很浅
- **大模型**（≥ 7 B）embedding 几乎可以忽略，权重全在 transformer block 里
- **大词表的模型**（Gemma 用了 256K SentencePiece vocab）embedding 又会重新变大

> 对大模型而言，"权重在哪儿"的答案就是：**FFN > Attention >> Embedding**。一个 LLaMA 块里 FFN 的三个矩阵差不多占整层 2/3 参数。

---

## 九、常见命名约定（q_proj / gate_proj / up_proj …）

不同代码库叫法不一样，下表是对照表：

### Attention

| HF 名字 | 别名 | 含义 |
|---|---|---|
| `q_proj` | `wq`, `W_Q` | Query 投影矩阵 |
| `k_proj` | `wk`, `W_K` | Key 投影矩阵 |
| `v_proj` | `wv`, `W_V` | Value 投影矩阵 |
| `o_proj` | `wo`, `W_O`, `out_proj` | 多头拼接后再投回 D 的"输出投影" |
| `qkv_proj` | — | 把 Q/K/V 三个矩阵合并成一个大矩阵以加速 |

### FFN（Gated）

| HF 名字 | 别名 | 形状 | 角色 |
|---|---|---|---|
| `gate_proj` | `w1`, `gate`, `gating` | D → d_ff | 门控分支（过激活） |
| `up_proj`   | `w3`, `fc_in`         | D → d_ff | 值分支 |
| `down_proj` | `w2`, `fc_out`        | d_ff → D | 降维投影 |

### FFN（经典）

| HF 名字 | 别名 | 形状 |
|---|---|---|
| `fc1` / `intermediate` / `c_fc` | up | D → d_ff |
| `fc2` / `output` / `c_proj`     | down | d_ff → D |

### Norm

- `input_layernorm`：每个 block 第一个 norm（attention 之前）
- `post_attention_layernorm`：attention 之后、FFN 之前
- `pre_feedforward_layernorm` / `post_feedforward_layernorm`：Gemma 等额外引入
- `q_norm` / `k_norm`：对 Q/K 投影结果再做一次 norm（Gemma 3、Qwen3 有）

---

## 十、IR 层面观察：高层模块在 OpenVINO/ONNX 里长什么样

一个 LLM 在被导出为 OpenVINO / ONNX IR 时，所有这些"高层模块"都被拆成了原子算子：

| 高层概念 | 在 IR 里你看到的样子 |
|---|---|
| `Embedding` lookup | `Gather`（按 token id 取行）；在已经把 embedding 抽出做外部 lookup 的模型里则**没有**这个算子，直接以 `[B,T,D]` 作为输入。 |
| `RMSNorm` | `Power(2) → ReduceMean → Add(eps) → Sqrt → Divide → Multiply(scale)` 一组 6~7 个原语。 |
| `q_proj` / `k_proj` / `v_proj` / `o_proj` / `gate_proj` / `up_proj` / `down_proj` | 全都是 `MatMul`（前面通常带 `Convert` 做 FP16→FP32）。靠"输出最后一维"很容易识别它们：D / d_ff / d_kv 各对应不同 FC 形状。 |
| `RoPE` | `Multiply(cos)` + `Multiply(sin)` + `Add` + `Reshape/Concat`（把张量切半再旋转）。 |
| `SDPA` | 两个 `MatMul`（QKᵀ 和 attn·V）+ 一个 `SoftMax` + 中间的 `Add(mask)`、`Multiply(1/√d)`。 |
| `Gated activation` (SwiGLU/GeGLU) | `Gelu` 或 `Swish` + 一个 `Multiply`（把门和 up 相乘）。 |
| `LM head` | 一个大 `MatMul`，输出维等于词表大小 V。在多分图导出里它常常**单独放在最后一个子图**。 |

所以当你看到 IR 里有 `gate_proj` / `down_proj` / `up_proj` 这种名字，就直接对应到 FFN 三个矩阵；看到 `q_proj` / `k_proj` / `v_proj` / `o_proj` 就对应 attention 的四个投影。这些名字是从原始 PyTorch 模型继承下来的层名（`model.layers.0.mlp.gate_proj` 之类），导出工具一般会保留作为算子的 `friendly name`。

---

## 附录：把 Gemma / LLaMA 标准 block 完整算子流写下来

```
         x  ───────────────────────────────────────┐
            │                                      │
            ▼  RMSNorm  (input_layernorm)          │
            ▼                                      │
            ├─→ q_proj ─┐                          │
            ├─→ k_proj ─┼─→ RoPE → SDPA →          │
            ├─→ v_proj ─┘   attn_out → o_proj      │
            ▼                                      │
            + ◀─────────── residual add ───────────┘   ← attention 残差
            │
            x' ──────────────────────────────────────┐
            │                                        │
            ▼  RMSNorm  (post_attention_layernorm)   │
            ▼                                        │
            ├─→ gate_proj ─→ SiLU/GELU ─┐            │
            ├─→ up_proj  ───────────────┴─→ ⊙ →      │
            │                              down_proj │
            ▼                                        │
            + ◀───────────  residual add  ───────────┘  ← FFN 残差
            │
            ▼   下一层
```

模型总参数主要由 **N × (4·D² + 3·D·d_ff)** 决定：
- 第一项是 attention 的 QKVO 投影（每个 D×D，共 4 个）
- 第二项是 FFN 的 gate/up/down（前两个 D×d_ff，第三个 d_ff×D）

N 越大越深、D 越大越宽、d_ff 越大表达力越强。

---

## 速查 cheatsheet

| 想搞清楚的事 | 看这里 |
|---|---|
| 模型整体长什么样 | §1 |
| Embedding 和 LM head 是不是共享 | §2 |
| 模型用绝对位置编码还是 RoPE | §3 |
| Block 里两个残差是怎么连的 | §4 |
| MHA / GQA / MQA 区别 | §5 |
| FFN 三件套 gate/up/down 的含义 | §6 |
| Attention 还是 FFN 算得多 | §7 |
| 权重大头在哪 | §8 |
| IR 算子名怎么对应到高层模块 | §10 |
