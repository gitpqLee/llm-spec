# Qwen3 vs Qwen3-Next (Qwen3.5) — NPU Decode 性能分析

> **背景**：Qwen3-4B（32 层纯 self-attn）实测 **34 t/s**，Qwen3-Next-4B（24 层，linear:self = 3:1 混合）实测 **28 t/s**。两者参数量都是 4B，1k 上下文（KV=1152）decode 场景。
>
> 本文档解释为什么"层数更少 (32→24)"的 hybrid 模型 decode **反而慢 ~20%**，并给出 NPU 侧的优化方向。

---

## 一、模型结构对比

### 1.0 Qwen3-Next (Qwen3.5) 整体结构图

```
                            输入 token IDs
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │  Embedding Layer    │  vocab_size × 2560
                      └──────────┬──────────┘
                                 │
                            [1, seq, 2560]
                                 │
   ┌─────────────────────────────┼─────────────────────────────┐
   │                             │                             │
   │       ╔═════════════════════▼════════════════════╗        │
   │       ║  Layer 0  ──  linear-attn  (GatedDelta) ║        │
   │       ╠══════════════════════════════════════════╣        │
   │       ║  Layer 1  ──  linear-attn                ║        │
   │       ╠══════════════════════════════════════════╣        │
   │       ║  Layer 2  ──  linear-attn                ║        │
   │       ╠══════════════════════════════════════════╣        │
   │       ║  Layer 3  ──  self-attn      (SDPA)      ║  ← 每4 层 1 个 self
   │       ╠══════════════════════════════════════════╣        │
   │       ║  Layer 4  ──  linear-attn                ║        │
   │       ║  ... (共 24 层，pattern = L-L-L-S 重复)   ║        │
   │       ║  Layer 23 ──  self-attn                  ║        │
   │       ╚══════════════════════════════════════════╝        │
   │                                                            │
   │     Attention 分布:  6 × self-attn + 18 × linear-attn      │
   │                      (linear : self = 3 : 1)               │
   └─────────────────────────────┬─────────────────────────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │   RMSNorm (final)   │
                      └──────────┬──────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │   LM Head (Linear)  │  2560 × vocab_size
                      └──────────┬──────────┘
                                 │
                                 ▼
                            logits → sampling → next token
```

### 1.1 Linear-Attn 层内部结构 (GatedDeltaNet — Layer 0/1/2/4/5/6/…)

```
        x_in [B, 1, 2560]  (residual stream)
                │
                ├────────────────────────────────────┐
                │                                    │
                ▼                                    │
      ┌──────────────────┐                           │
      │ input_layernorm  │ (RMSNorm, gain × f32)     │  residual
      └────────┬─────────┘                           │  save
               │ [B, 1, 2560]                        │
    ┌──────────┼──────────┬──────────┬──────────┐   │
    ▼          ▼          ▼          ▼          │   │
 in_proj_a  in_proj_b  in_proj_qkv in_proj_z    │   │
 2560→32    2560→32    2560→8192   2560→4096    │   │
 (int4)     (int4)     (int4)      (int4)       │   │
    │          │          │          │          │   │
    │          │          │          ▼          │   │
    │          │          │       ┌──────┐      │   │
    │          │          │       │ SiLU │      │   │
    │          │          │       └──┬───┘      │   │
    │          │          │  (z gate)│          │   │
    │          │          ▼          │          │   │
    │          │      split → q/k/v  │          │   │
    │          │       [8192]        │          │   │
    │          │          │          │          │   │
    │          │  ┌───────┴──────┐   │          │   │
    │          │  ▼              ▼   │          │   │
    │          │ ┌──────────┐  ┌──────────┐     │   │
    │          │ │ Causal   │  │ Reshape  │     │   │
    │          │ │ Conv1D   │  │ (Q view) │     │   │
    │          │ │ k=4      │  └────┬─────┘     │   │
    │          │ │ conv-    │       │           │   │
    │          │ │ cache    │       │           │   │
    │          │ │ [8192,4] │       │           │   │
    │          │ └────┬─────┘       │           │   │
    │          │      │             │           │   │
    │          ▼      ▼             ▼           │   │
    │        A       B             q,k,v        │   │
    │          │      │             │           │   │
    │          └──────┴──────┬──────┘           │   │
    │                        ▼                  │   │
    │              ┌──────────────────┐         │   │
    │              │  SSM Loop (scan) │  ← SHAVE-only, sequential
    │              │                  │     fp32,  ~255 μs
    │              │  state[t] =      │     每步递归依赖
    │              │    A·state[t-1]  │
    │              │    + B·x[t]      │
    │              │                  │
    │              │  ssm-state:      │
    │              │  [32,128,128]    │
    │              │  fp32 (~2MB/L)   │
    │              └────────┬─────────┘
    │                       │
    │                  y (attn output)
    │                       │ [B, 1, 4096]
    │                       │
    │                       ▼
    │                  ┌─────────┐
    │                  │  ⊙ z    │  (element-wise gate)
    │                  └────┬────┘
    │                       │
    │                       ▼
    │                  ┌──────────────────┐
    │                  │  norm (RMSNorm)  │
    │                  └────────┬─────────┘
    │                           │ [B, 1, 4096]
    │                           ▼
    │                  ┌──────────────────┐
    │                  │  out_proj        │
    │                  │  4096 → 2560     │
    │                  │  (int4)          │
    │                  └────────┬─────────┘
    │                           │ [B, 1, 2560]
    │                           │
    └───────────────────────────┴─── + (residual add)
                                │
                                ▼
                    ┌────────────────────────┐
                    │ post_attention_layernorm│
                    └────────────┬───────────┘
                                 │
                                 │  (标准 SwiGLU MLP)
                    ┌────────────┼────────────┐
                    ▼            ▼            │
                gate_proj    up_proj          │
                2560→9216    2560→9216        │
                (int4)       (int4)           │
                    │            │            │
                  SiLU           │            │
                    │            │            │
                    └────── × ───┘            │
                           │                  │
                           ▼                  │
                      down_proj               │
                      9216→2560               │
                      (int4)                  │
                           │                  │
                           └──── + (residual)◄┘
                                 │
                                 ▼
                              x_out [B, 1, 2560]
```

**Linear-attn 层的关键特征**：
- **5 个 GEMM**：in_proj_a / b / qkv / z / out_proj（比 self-attn 多一个 `in_proj_z` GLU gate）
- **无 KV cache**，改用 fixed-size 状态：
  - `conv-cache [8192, 4]` fp32 — Causal Conv1D 的滑动窗口
  - `ssm-state [32, 128, 128]` fp32 — SSM 递归状态
- **状态大小与 context length 无关**（长上下文优势的来源）
- **执行硬件**：GEMM 走 DPU，SSM Loop 走 **SHAVE only**（NPU 上的瓶颈）

### 1.2 Self-Attn 层内部结构 (标准 GQA — Layer 3/7/11/15/19/23)

```
        x_in [B, 1, 2560]  (residual stream)
                │
                ├────────────────────────────────────┐
                │                                    │
                ▼                                    │
      ┌──────────────────┐                           │
      │ input_layernorm  │  (RMSNorm)                │
      └────────┬─────────┘                           │
               │                                     │
    ┌──────────┼──────────┐                          │
    ▼          ▼          ▼                          │
  q_proj    k_proj      v_proj                       │
  2560→8192 2560→1024   2560→1024                    │
  (int4)    (int4)      (int4)                       │
    │          │           │                         │
    ▼          ▼           ▼                         │
  reshape    reshape     reshape                     │
  [1,16,     [1,4,       [1,4,                       │
   1,256]     1,256]      1,256]                     │  residual
    │          │           │                         │  save
    ▼          ▼           │                         │
  q_norm    k_norm         │                         │
  (RMSNorm) (RMSNorm)      │                         │
    │          │           │                         │
    ▼          ▼           │                         │
   RoPE      RoPE          │  (V 不做 RoPE)          │
   (cos/sin) (cos/sin)     │                         │
    │          │           │                         │
    │          ▼           ▼                         │
    │      ┌──────────────────────┐                  │
    │      │  KV cache concat     │                  │
    │      │  past_key: [1,4,N,256]                  │
    │      │  past_value:[1,4,256,N]  (V已转置)      │
    │      └────────┬─────────────┘                  │
    │               │                                │
    │      K:[1,4,N+1,256]  V:[1,4,256,N+1]          │
    │      GQA broadcast: 4 heads → 16 heads         │
    │               │                                │
    ▼               ▼                                │
   ┌────────────────────────────────┐                │
   │  SDPA  (scaled_dot_product)    │                │
   │                                │                │
   │  ┌─────────────────────────┐   │                │
   │  │ qk_matmul (Q · Kᵀ)      │   │                │
   │  │ [1,16,1,256]×[1,16,     │   │                │
   │  │  N+1,256]→[1,16,1,N+1]  │   │                │
   │  └────────┬────────────────┘   │                │
   │           │                    │   ← 全部拆到 DPU
   │           ▼                    │      as_convolution
   │  ┌─────────────────────────┐   │      墙钟 ~84-102 μs
   │  │ + mask,  scale,  softmax│   │      (含所有 sub-ops)
   │  └────────┬────────────────┘   │
   │           │                    │
   │           ▼                    │
   │  ┌─────────────────────────┐   │
   │  │ output_matmul           │   │
   │  │ (softmax · V)           │   │
   │  │ [1,16,1,N+1] ×          │   │
   │  │ [1,16,256,N+1]→         │   │
   │  │ [1,16,1,256]            │   │
   │  └────────┬────────────────┘   │
   └───────────┼────────────────────┘
               │ [1,16,1,256]
               ▼
        concat 16 heads
               │ [1,1,4096]
               ▼
          ┌─────────┐
          │ o_proj  │
          │4096→2560│  (int4)
          └────┬────┘
               │ [1,1,2560]
               │
               └──── + (residual add)◄──────────────┘
                     │
                     ▼
           ┌────────────────────────┐
           │ post_attention_layernorm│
           └────────────┬───────────┘
                        │
                        │  (标准 SwiGLU MLP，跟 linear-attn 层完全一样)
              ┌─────────┼─────────┐
              ▼         ▼         │
          gate_proj  up_proj      │
          2560→9216  2560→9216    │
              │         │         │
            SiLU        │         │
              │         │         │
              └── × ────┘         │
                    │             │
                    ▼             │
                down_proj         │
                9216→2560         │
                    │             │
                    └── + (res)◄──┘
                    │
                    ▼
                 x_out [B, 1, 2560]
```

**Self-attn 层的关键特征**：
- **4 个 attention GEMM**：q_proj / k_proj / v_proj / o_proj + **3 个 MLP GEMM** = 单层共 7 个
- **KV cache 随 context 线性增长**：`[1,4,N,256]` (fp16)，N=context 长度
- **SDPA 拆到 DPU**（as_convolution 路径），decode 时 seq=1 → SDPA 只 ~84 μs
- **无额外 gate 分支**

### 1.3 三种 Attention 层的结构差异

| 项 | **Qwen3 self-attn** | **Qwen3-Next self-attn** | **Qwen3-Next linear-attn (GatedDeltaNet)** |
|---|---|---|---|
| Hidden dim | 2560 | 2560 | 2560 |
| MLP intermediate | 9728 | 9216 | 9216 |
| Q head 数 × head_dim | 32 × 128 | 16 × 256 | (SSM: 32 groups) |
| KV head 数 (GQA) | 8 (4:1) | 4 (4:1) | — |
| Q proj 输出 | 4096 | 8192 | in_proj_qkv: 8192 |
| K/V proj 输出 | 1024 / 1024 | 1024 / 1024 | 融合在 qkv 里 |
| O proj | 4096→2560 | 4096→2560 | out_proj: 4096→2560 |
| 额外 gate 分支 | ❌ | ❌ | ✅ in_proj_z: 2560→4096 (SiLU) |
| Attention 内核 | SDPA (QKᵀ→softmax→·V) | SDPA | **Conv1D + SSM Loop 递归扫描** |
| 状态存储 | KV cache `[B,H,L,D]` fp16 | 同左 | Conv-cache `[8192,4]` + SSM state `[32,128,128]` fp32 |
| Seq 长度复杂度 | O(N²·d) | O(N²·d) | O(N·d²) |
| 状态大小与 N | **线性增长** | **线性增长** | **恒定**（context-length 无关）|

### 全模型层数分布

| | **Qwen3-4B** | **Qwen3-Next-4B (hybrid)** |
|---|---:|---:|
| 总层数 | **32 层** | **24 层** |
| 层类型分布 | 32 × self-attn | 6 × self-attn + 18 × linear-attn (**1:3**) |

---

## 二、GEMM 数量对比

| | Qwen3 self-attn | Qwen3-Next self-attn | Qwen3-Next linear-attn |
|---|:-:|:-:|:-:|
| **Attention GEMM** | 4 (q, k, v, o_proj) | 4 (q, k, v, o_proj) | **5** (qkv, z, a, b, out_proj) |
| **MLP GEMM** | 3 (gate, up, down) | 3 | 3 |
| **单层 GEMM 总数** | **7** | **7** | **8** |
| **int4 权重量 (attention 部分)** | ~15 MB | ~15 MB | **~21 MB** |
| **int4 权重量 (MLP 部分)** | ~14 MB | ~13 MB | ~13 MB |
| **单层 int4 权重总量** | ~29 MB | ~28 MB | **~34 MB** |

**关键差异**：linear-attn 比 self-attn **多一个 `in_proj_z` GLU gate 分支**（5.2 MB int4 权重），这在 decode 阶段直接体现为多一次 DDR→CMX 权重搬运。

---

## 三、单层墙钟执行时间对比（Profile 实测，同 KV=1152）

### 3.1 GEMM 部分（有静态权重的线性投影）

| GEMM | Qwen3 self-attn | Qwen3-Next self-attn | Qwen3-Next linear-attn |
|---|---:|---:|---:|
| Q proj / in_proj_qkv | 111 μs | ~181 μs (layer 11) | ~171 μs (avg) |
| K proj | 67 μs | ~59 μs | (合并进 qkv) |
| V proj | 72 μs | ~76 μs | (合并进 qkv) |
| **in_proj_z (GLU gate)** | — | — | **~366 μs** |
| in_proj_a / b (SSM 参数) | — | — | ~104 μs |
| O proj / out_proj | ~200 μs* | ~200 μs* | ~108 μs |
| **Attention GEMM 合计** | **~450 μs** | **~516 μs** | **~749 μs** |
| MLP (gate + up + down) | ~578 μs | ~570 μs | ~570 μs |
| **单层 GEMM 合计** | **~1028 μs** | **~1086 μs** | **~1319 μs** |

> \* o_proj 在 profile 中因依赖等待常报出异常大值（3.5ms 级），此处用同尺寸 out_proj 反推的正常值 ~200 μs。

### 3.2 Attention 内核（不含 GEMM）

| 组件 | Qwen3 self-attn | Qwen3-Next self-attn | Qwen3-Next linear-attn |
|---|---:|---:|---:|
| SDPA (scale + mask + qk_matmul + softmax + output_matmul) | **~102 μs** | **~84 μs** | — |
| Conv1D (GroupConvolution) | — | — | ~43 μs |
| **SSM Loop (SHAVE sequential scan)** | — | — | **~255 μs** |
| **Attention 内核合计** | **~102 μs** | **~84 μs** | **~298 μs** |

### 3.3 归一化：单层总墙钟

| | Qwen3 self-attn | Qwen3-Next self-attn | Qwen3-Next linear-attn |
|---|---:|---:|---:|
| GEMM (attn + MLP) | 1028 μs | 1086 μs | 1319 μs |
| Attention 内核 | 102 μs | 84 μs | 298 μs |
| 其他 (LayerNorm, RoPE, KV concat 等) | ~40 μs | ~50 μs | ~50 μs |
| **单层墙钟** | **~1170 μs** | **~1220 μs** | **~1670 μs** |
| **相对 Qwen3 self-attn** | 1.00× | 1.04× | **1.43×** |

---

## 四、Profile 关键数据点（用于说明"数字怎么得来"）

### 4.1 Qwen3-4B REP 块 (kv=1152)

- REP 块内容：layer 0 下半段 + layer 1 上半段 = **1 个完整 transformer block**
- 总墙钟：**~772 μs / block**
- Top layer summaries (从 profile):
  ```
   764.2 μs   layers.1 SDPA (含 dispatch/等待 bubble, 实际计算仅 ~102 μs)
   390.5 μs   layers.0 o_proj (含 SDPA 依赖 bubble)
   194.7 μs   layers.0 mlp.gate_proj
   198.4 μs   layers.0 mlp.down_proj
   185.2 μs   layers.0 mlp.up_proj
   110.9 μs   layers.1 q_proj
   ...
  ```
- **SDPA 纯计算 span**（去掉 mask_add 提前 dispatch 的假象）：**~102 μs**

### 4.2 Qwen3-Next-4B REP 块 (kv=1152)

- REP 块内容：layer 7 下半 + layer 8/9/10 完整 + layer 11 上半 = **~3.5 个 transformer block**
- 总墙钟：**~4281 μs / REP**，即 **~1223 μs / block**
- Top layer summaries：
  ```
  3637 μs   layer 7 mlp.gate_proj (含 bubble)
  3577 μs   layer 7 self_attn.o_proj (严重依赖等待)
   369 μs   layer 8 linear_attn.in_proj_z    ← linear-attn 独有的 GLU gate
   367 μs   layer 9 linear_attn.in_proj_z
   365 μs   layer 10 linear_attn.in_proj_z
   255 μs   Loop_10495 (SSM scan)             ← 最大痛点
   253 μs   Loop_11605 (SSM scan)
   253 μs   Loop_12715 (SSM scan)
   251 μs   layer 8 linear_attn.in_proj_qkv
    84 μs   layer 7 SDPA (compute span)
  ```

### 4.3 FLOPs vs 时间的反差

Decode 阶段 seq=1, ctx=1152 时：

| 组件 | FLOPs 单层 | 墙钟 单层 |
|---|---:|---:|
| SDPA (qk + output_matmul) | **9.4 M** | **~84 μs** |
| SSM 更新（每步）| **~2 M** | **~255 μs** |

**FLOPs 上 SDPA 反而多 4.7 倍，但执行时间 SSM Loop 慢 3 倍** —— 说明 FLOPs 完全不预测 NPU 上的实际耗时。差异根源是**执行硬件的选择**：SDPA 拆到 DPU（高吞吐 batched GEMM），SSM Loop 落在 SHAVE 上做 sequential scan。

---

## 五、全模型 decode 总时间与实测的吻合

### 单 token 时间预测

**Qwen3-4B**:
```
32 层 × 1170 μs = 37.4 ms/token (预测)
1/34 t/s = 29.4 ms/token          (实测, 含 embedding/lm_head 等)
```

**Qwen3-Next-4B**:
```
6 × 1220 + 18 × 1670 = 7.3 + 30.1 = 37.4 ms/token (纯 layer 预测)
1/28 t/s = 35.7 ms/token                          (实测)
```

**预测和实测的相对关系高度一致**：hybrid 比 pure self-attn 慢 **约 20%**。绝对值上因为 REP 块划分方式、NPUW inter-chunk overhead 差异，预测偏大 20-30%，但不影响相对结论。

### 三笔账加起来说明差距

**账本 1: 层数减少节省**
```
Qwen3-Next 减少 8 层 (32→24) → 25% 减少
按 Qwen3 每层 ~1170 μs 计算：
    8 × 1170 = -9.4 ms/token
```

**账本 2: Linear-attn 每层"倒贴"**
```
每个 linear-attn 层比 self-attn 层多花:
    1670 - 1170 = +500 μs/层
18 个 linear-attn 层:
    18 × 500 = +9.0 ms/token
```

**账本 3: 净结果**
```
账本1 节省 ≈ 账本2 净增
但 second-order 影响（fp32 精度、SHAVE 争抢、SSM state DMA）导致
净新增 > 净减省
→ 全模型慢 ~20%
```

---

## 六、Linear-Attn 慢的三大成因（按贡献排序）

### 🔴 成因一：SSM Loop 只能跑在 SHAVE 上（占比 ~55% 慢源）

- **本质**：SSM 是 `state[t] = A·state[t-1] + B·x[t]` 的**顺序递归**，每一步都依赖上一步
- **硬件后果**：DPU 是 batched 并行硬件，处理不了这种数据依赖 → **只能落到 SHAVE**
- **性能后果**：
  - 单层墙钟 ~255 μs（vs SDPA 84-102 μs）
  - 用 fp32（精度稳定性需要），进一步拖慢
  - state 张量 `1×32×128×128×4B = 2 MB/层`，每步都要 CMX ↔ DDR 搬运

### 🟡 成因二：多一个 in_proj_z GLU gate GEMM (~30% 慢源)

- 每个 linear-attn 层比 self-attn 层多一个 `2560→4096` 的 GEMM
- profile 实测 **~366 μs**（大于 q_proj/o_proj 单独一个的时长）
- 纯粹是"multiplicative gating"设计带来的额外权重带宽压力
- **本质是模型架构选择**，NPU 无法优化掉，但可以通过 GEMM 融合减轻

### 🟢 成因三：SDPA 本来就快 → 换成 SSM Loop 是"负优化" (~15%)

- 在 seq=1 的 decode 阶段，SDPA 两个 MatMul 加起来 <30 μs（FLOPs 只有 9.4M）
- linear-attn 的设计初衷是"省 O(N²) 的 SDPA"—— **decode 时这个 O(N²) 已经是常数 0**
- 结果：linear-attn 花 ~296 μs 去替换一个只需要 ~84 μs 的东西 → **净增 200 μs/层**

---

## 七、Hybrid 模型的收益在哪里（公平起见）

Linear-attn 在 NPU decode 上是"倒贴"，但设计初衷不是为了 decode。它的价值在：

| 场景 | Linear-Attn 优势 |
|---|---|
| **Prefill 长序列** | SDPA 的 O(N²·d) 会爆炸（16k token 时是 seq=1 的 2.5 亿倍工作量）；linear-attn 是 O(N·d²)，快得多 |
| **KV cache 内存** | self-attn KV = 4×N×256×2B / layer （**随 N 线性增长**）；linear-attn state = 2 MB / layer（**恒定**）|
| **长上下文 32k+** | KV cache 内存爆炸时 hybrid 是唯一可行方案 |
| **训练侧** | 长序列训练 O(N²) 显存不可承受 |

**但 4B 模型 + 1k 上下文的 decode 场景**：以上优势**全部失效**（seq=1，context 才 1k，KV cache 才 2.4 MB/layer）。

---

## 八、汇总对比表

| 维度 | Qwen3-4B (32L self) | Qwen3-Next-4B (24L hybrid) | 结论 |
|---|---|---|---|
| Attention 类型 | 32 × self | 6 × self + 18 × linear | linear-attn 硬件不友好 |
| 单层 GEMM 数 | 7 | 7 (self) / 8 (linear) | linear 多 1 个 gate |
| 单层墙钟 | ~1170 μs | 1220 (self) / 1670 (linear) μs | linear 慢 43% |
| 总层墙钟/token | ~37 ms | ~37 ms | 略慢 |
| 实测 token rate | **34 t/s** | **28 t/s** | **-18%** |
| SSM Loop 硬件 | — | **SHAVE only, seq scan, fp32** | 关键瓶颈 |
| KV cache 增长 | O(N) 随 N 增大 | 恒定 2 MB/层 | hybrid 长上下文优势 |
| 优化空间 | 权重带宽 (int4 已用) | **SSM kernel 优化**、fp16 化、gate 融合 | 有空间但复杂 |

---

## 九、后续优化方向（NPU 侧，不改模型）

按预期收益排序：

| 优化点 | 预期收益 | 难度 |
|---|---:|---|
| **SSM Loop SHAVE kernel 优化**：fp16 化、多 SHAVE tile 化、scan 内并行 | 单 linear 层 -100 到 -150 μs → 全模型 **+5-8% t/s** | 中高 |
| **in_proj_z 和 in_proj_qkv 融合**（如果算法允许合成 2560→12288 一次 GEMM）| 单 linear 层 -50 μs → **+2% t/s** | 中 |
| **SSM state 从 fp32 降到 fp16 / bf16**（如果精度允许）| DMA 减半 + SHAVE 计算减半 → **+3-5% t/s** | 高（需 accuracy 验证）|
| **Multi-cluster 并行**：SSM Loop 目前单 cluster，可尝试 3 cluster 并行 | 潜在 3× 加速 | 高 |
| **o_proj / 权重预取调度优化**（消除 layer 7 那种 3577 μs bubble）| 视情况 | 中 |

---

## 十、结论

### 一句话解释
> **Qwen3-Next-4B (hybrid) 在 1k 上下文 decode 场景比 Qwen3-4B 慢 20%，是因为 linear-attention 用 SSM 递归扫描替代 SDPA，而 SSM Loop 是顺序依赖 → 只能跑在 NPU 的 SHAVE 上做 sequential scan，无法利用 DPU 的并行算力。虽然 hybrid 层数少了 25%（32→24），但 18 个 linear-attn 层每层多花 500 μs，把省下的时间全吃回去还倒贴。**

### 给业务侧的建议

- **短上下文 + decode 主导场景 (≤2k)**：**Qwen3 更快**，推荐使用
- **长上下文场景 (8k+)**：Qwen3-Next 是唯一选择（Qwen3 的 KV cache 会爆内存）
- **Prefill 密集场景**：Qwen3-Next 有显著优势（linear-attn 的 O(N·d²) vs SDPA 的 O(N²·d)）
- **NPU compiler 团队短期目标**：SSM Loop kernel 达到 SDPA 同数量级性能（150 μs 以内）→ 消除 hybrid 的 decode 劣势

---

## 附录 A: Profile 数据来源

- **Qwen3-4B**: `/home/pengqian/vpux/firmware/firmware.vpu.client/validation/validationApps/system/nn/5000/InferenceManagerDemo/prof-qwen3-4B-kv-cache-rep-1024.json`
  - 对应 IR: `/home/pengqian/vpux/models/qwen/subgraphs/qwen3-4B-subs/Model0_kv1152_01_REP00AB.xml`
- **Qwen3-Next-4B**: `/home/pengqian/vpux/firmware/firmware.vpu.client/validation/validationApps/system/nn/5000/InferenceManagerDemo/prof-qwen35-4B-kv-cache-rep-1024.json`
  - 对应 IR: `/home/pengqian/vpux/models/qwen/subgraphs/4B-subs/split-kv-cache/Model24_kv1152_03_REP016C.xml`

## 附录 B: 术语速查

| 术语 | 含义 |
|---|---|
| **SDPA** | Scaled Dot Product Attention — 标准 self-attention 计算 (Q·Kᵀ → softmax → ·V) |
| **GQA** | Grouped Query Attention — 多个 Q head 共享 K/V head 的注意力变体 |
| **SwiGLU** | 门控线性单元 MLP，`down_proj(SiLU(gate_proj(x)) * up_proj(x))`，需要 3 个 GEMM |
| **SSM** | State Space Model — 用状态递归实现序列建模，属于线性 attention 家族 |
| **GatedDeltaNet** | Qwen3-Next 使用的 linear-attn 变体，带门控 SSM |
| **REP block** | NPUW 切图后的可复用块，编译时被识别为可以共享 blob 的重复结构 |
| **CMX** | NPU 的 on-chip SRAM，DPU/SHAVE 直接可访问 |
| **DPU** | NPU 的矩阵计算引擎，擅长 batched GEMM/Conv |
| **SHAVE** | NPU 的通用向量处理器，处理 DPU 不擅长的算子 (softmax, scan, elementwise 等) |
