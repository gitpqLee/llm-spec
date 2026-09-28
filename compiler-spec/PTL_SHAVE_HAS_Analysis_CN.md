# PTL / NPU5 SHAVE 硬件专题报告

## 0. 来源、范围与阅读方式

- 原始文件：[NPU5 IP Hardware Architecture Specification.pdf](../NPU5%20IP%20Hardware%20Architecture%20Specification.pdf)。
- 正文版本：2026-09-10；PDF 导出时间：2026-09-28；共 714 页。
- 本报告的 `p. N` 均为 PDF 物理页码，从 1 开始，与页脚的 `N/714` 一致。
- 对 714 页进行了全文文本检索，207 页命中 SHAVE / ACT-SHV / VAU / SAU / IAU / CMU / LSU / VLIW。精读核心章节及关联约束，另对无文本层的地址图 p. 604-606 做了本地 OCR。
- [逐页证据索引](evidence-index.md)、[完整命中行](shave_hits.json)、[带页码全文](has.txt)、[来源元数据与 SHA-256](provenance.json)、[外部引用链接](external-links.json)。
- 本报告是专题归纳，不是完整 HAS 或 ISA 的逐字翻译；无关的 DPU/CPU 寄存器没有展开。检索不等于逐张审核所有 RTL/DFx 图，外部链接中的 ISA、寄存器手册、HSD 和 SAS 尚未读取。
- **“HAS 事实”**指本文档明确写出的内容；**“开发含义”**为基于该事实的工程推论，不代表当前 compiler/runtime 已实现或已启用。
- 本文档混有 PTL、WCL、ARLS-R、其他代际的通用参数和遗留文字。PTL 实际配置优先采用 §3.5.3 的专用表，不把通用上限当作产品配置。

## 1. 最重要的结论

1. **PTL 默认是 3 个 NCE tile，每 tile 2 个 ACT-SHAVE，共 6 个 SHAVE。** 每 tile 有 1536 kB CMX，总 CMX 为 4.5 MB。不是此前用于讲解 NPU4/LNL 的 6-tile 配置。[p. 5, p. 106]
2. **ACT-SHAVE 是 8-slot VLIW 向量 DSP。** 512-bit VAU、512-bit CMU、两个 256-bit LSU，以及标量、整数、分支和谓词单元共同组成可编程核心。[p. 333]
3. **CMX 数据不经过 SHAVE L1/L2。** 指令与远端数据路径访问 DDR/cache；近端数据路径访问 CMX。PTL 集成表明确不支持从 CMX 取指。[p. 152, p. 158]
4. **本地 CMX 可读写，其他 tile 的 CMX 只支持写。** 不能把跨 tile 内存视为任意共享读写 SRAM。[p. 147, p. 175]
5. **SHAVE 与 DPU/DMA 靠任务 FIFO 和硬件 barrier 协作。** HAS 强烈推荐用 barrier 同步这些执行单元；不是 GPU 式大量线程硬件调度模型。[p. 125-131]
6. **SHAVE L2 是全体 SHAVE 共享的 256 KB，32 个 partition，4 个 subgroup。** 不是每核 256 KB，也不是 32 份独立物理缓存。[p. 152, p. 348-353]
7. **PTL 新增 FP8 load/store 支持不等于证明了原生 FP8 算术吞吐。** 完整指令定义位于外部 ISA 文档。[p. 6, p. 335, p. 348]
8. **SHAVE 可以通过受控机制请求 DMA，但不能任意控制 DMA 配置。** 固件必须先授权 Link Agent，硬件附加 context。是否能在产品接口使用还需检查 runtime。[p. 237, p. 634]
9. **硬件支持调试和 profiling。** DCU 提供断点、单步、最近指令历史和计数器；可用的软件工具与设备授权是另一个问题。[p. 617, p. 671, p. 681-683]
10. **HAS 不能给出一个可信的实际 SHAVE TFLOPS 或整网速度。** L2 性能结果小节为空，指令吞吐表不在本 PDF；端口峰值、仿真目标和实测性能必须分开。[p. 148-150, p. 434, p. 615]

## 2. 章节地图

| 主题 | HAS 章节 | PDF 页码 | 关注点 |
| --- | --- | --- | --- |
| PTL 配置与代际增量 | §2.6-2.7, §3.5.3 | p. 5-6, p. 106 | 3 tile、6 SHAVE、FP8、L2 warmer、IRQ |
| 系统与主机接口 | §3.1, §3.3 | p. 11, p. 17-18, p. 23-35, p. 59-62 | 用户权限、MMU、地址宽度、错误传播 |
| NCE 资源虚拟化 | §3.5.4.2-4 | p. 109-125 | JOB_X_MAP、逻辑 tile、隔离 |
| FIFO 与 barrier | §3.5.4.5-6 | p. 125-131 | 工作分发、IPC、同步、计数器 |
| CMX 与互连 | §3.5.4.7 | p. 133-150 | bank、LSU 带宽、延迟、swizzle、错误 |
| 浮点语义 | §3.5.4.8 | p. 150-151 | denormal、NaN、clamp、BF16、FP8 |
| SHAVE 集成 | §3.5.4.9, §3.5.6 | p. 151-158 | GPI/GPO、P_ID、自门控、near/far |
| NCE 安全与地址 | §3.5.7, §3.5.10-19 | p. 159-175, p. 187, p. 196-206 | context、远端写、控制平面 |
| DMA 协同 | §3.7.3.3.5, §16.4.1.1.5 | p. 237, p. 634 | SHAVE 调度的数据搬运、受控 job push |
| SHAVE Core | §3.9 | p. 333-348 | VLIW、寻址、L1、寄存器、异常、中断 |
| SHAVE L2 | §3.10 | p. 348-360 | partition、仲裁、计数器、ECC、错误 |
| 系统 AXI 接口 | §4.3.3 | p. 431-435 | outstanding、延迟目标、DDR 通路 |
| 时钟与电源 | §6-7, §9 | p. 478-501, p. 549-559, p. 580-588 | 门控、上下电、复位清理 |
| 地址图与寄存器入口 | §11.1.2-11.2 | p. 602-606 | 无文本层地址图、外部 register map |
| 性能与时间戳 | §13.4.2, §14 | p. 615, p. 617-621 | 性能数据缺口、计数器、profiling timer |
| 信任边界 | §16.4-16.5 | p. 633-645 | Ring3 kernel、隔离、context switch |
| DFx 与调试 | §17-18 | p. 649-658, p. 665-671, p. 681-683 | 授权、cross-trigger、DCU、trace |
| 相关 errata | §23 | p. 690-711 | barrier、L2 错误、恢复、精度与 DMA |

## 3. 硬件定位与 PTL 资源

### 3.1 它在 NPU 中是什么

NCE tile 同时包含 DPU、ACT-SHAVE 和 CMX。DPU 执行专用神经网络计算，ACT-SHAVE 执行不适合 DPU 的标准层及自定义层。SHAVE DSP 函数应包含在 graph-file 中，并通过 barrier 与其余调度同步。[p. 104, p. 151]

```text
受信 RISC-V 固件: 分配资源、配置 context、启动与管理任务
                           |
                   NCE 任务 FIFO / barriers
                           |
          +----------------+----------------+
          |                |                |
      DPU 描述符       SHAVE 程序          DMA 描述符
          |                |                |
      专用计算阵列      向量 DSP 核       数据搬运引擎
          +----------------+----------------+
                           CMX
```

这里的 ACT-SHAVE 不等于运行可信固件的 RISC-V。历史上的 SHAVE-NN 管理逻辑也不能混为一谈：DPU 章节说明该角色从 VPU4 起被状态机替代。[p. 11, p. 224, p. 633-636]

### 3.2 PTL 实际配置

| 项目 | PTL 默认值 | 来源 |
| --- | --- | --- |
| NCE tile | 3 | p. 106 |
| ACT-SHAVE / tile | 2 | p. 106 |
| ACT-SHAVE 总数 | 6，按 3 × 2 推得 | p. 106 |
| DPU / tile | 1 | p. 106 |
| CMX / tile | 1536 kB | p. 106 |
| CMX 总量 | 4.5 MB | p. 106 |
| SHAVE L2 | 256 KB，全体 SHAVE 共享 | p. 152, p. 348 |
| SHAVE IL1 / DL1 | 每核 4 KB / 1 KB | p. 333, p. 336-337 |
| DMA engine | 1，外部接口配置 2 × 64 B | p. 106 |

**开发含义：**实际 job 可能只分到部分 tile；熔断、资源分配和电源状态都影响可用资源。不要将“芯片有 6 个 SHAVE”写死为每个 kernel 都有 6 个可用执行实例。

## 4. SHAVE Core：指令并行与寄存器

### 4.1 八个功能单元

| 单元 | 作用 | 编程时需要关注 |
| --- | --- | --- |
| PEU | 谓词执行 | 控制操作是否执行、条件码更新模式 |
| BRU | 分支与 repeat | 循环、跳转、delay slot、可抢占性 |
| LSU0 / LSU1 | 两个 256-bit load/store 单元 | 访存流水、near/far 路径、端口冲突 |
| VAU | 512-bit 向量算术 | 数据并行、算术依赖、累加 |
| SAU | 32-bit vector/scalar arithmetic | 标量或窄向量操作，并非只有浮点 |
| IAU | 32-bit 整数算术 | 索引、地址计算、整数运算 |
| CMU | 512-bit compare/move | 比较、数据移动、转换与条件状态 |

来源：§3.9.1-2，[p. 333-334]。8-slot 不代表 8 个任意操作都能同时执行，更不代表 8 个独立 512-bit ALU。操作组合必须满足功能单元、数据依赖和寄存器端口规则。

512 bit 在存储容量上能装 32 个 FP16 或 16 个 FP32；这只是 lane 容量推导，不是所有指令的每周期吞吐承诺。三角函数支持被列为 feature，但其精度、latency、throughput 需要 ISA。[p. 333, p. 348]

### 4.2 寄存器资源

| 寄存器组 | 容量 | 并行访问 |
| --- | --- | --- |
| VRF | 32 × 512 bit = 2048 B | 最多 6 read + 6 write，共享端口 |
| IRF | 32 × 32 bit = 128 B | 最多 12 read + 6 write，共享端口 |
| TRF | 最多 64 × 32 bit = 256 B | 每项独立读取，特殊写入优先级 |

来源：[p. 340-343]。DCU 占用 VRF/IRF 的 port 0 时有优先权；HAS 建议 DCU 访问前 halt SHAVE，避免破坏运行状态。

**关键 VRF 端口共享：**

| 方向 | 端口 | 使用者 |
| --- | --- | --- |
| Read | 0 | VAU.VIA / DCU.VIA |
| Read | 1 | VAU.VIB |
| Read | 2 | SAU.VIA / CMU.VIB |
| Read | 3 | CMU.VIA |
| Read | 4 | LSU0.VIA / CMU.VIC |
| Read | 5 | LSU1.VIA |
| Write | 0 | VAU.VOA / DCU.VOB |
| Write | 1 | CMU.VOA |
| Write | 2 | LSU0.VOA |
| Write | 3 | LSU1.VOA / CMU.VOD |
| Write | 4 / 5 | CMU.VOB / CMU.VOC |

IRF 也存在多处共享，例如 read port 10 由 CMU.IIC 与 BRU.IIA 共用；完整 read/write 分配表在 [p. 341-342]。

**开发含义：**没有值依赖的两条指令，也可能因端口共享不能放进同一 VLIW bundle。展开循环和预取要同时预算仅 32 个向量寄存器的压力，不能只算 arithmetic 数量。

### 4.3 TRF、寄存器旋转与条件码

- `P_GPI/P_GPO`：与系统同步和门控逻辑交换状态。
- `P_ID`：核心与平台身份信息；`P_CFG`：数值行为配置。
- `B_ISR_SA/B_ISR_RA/B_STS/B_CFG/B_PRE`：中断、返回、状态、保护与软件抢占。
- `B_IP/B_LBEG/B_LEND/B_SREPS/B_MREPS`：指令地址和 repeat 状态。
- `I_STS/S_STS/C_STS/V_STS`：运算状态；`I_CCR/S_CCR/C_CCR_0..7`：条件码。
- TRF 还映射 SAU 与 VAU accumulator。完整地址表见 [p. 342-343]。
- TRF 写入优先级：DCU > `CMU.CPIT` > 普通指令更新。[p. 342]

寄存器 rebasing 由 `B_CFG.BASE/RMOD` 或 `BRU.RFB/RPI` 控制：[p. 344]

| RMOD | 旋转范围 | 地址变换 |
| --- | --- | --- |
| 0 | 0..31 | `(A + BASE) % 32` |
| 1 | 0..7 | 范围内 `(A + BASE) % 8`，其余不变 |
| 2 | 0..15 | 范围内 `(A + BASE) % 16`，其余不变 |
| 3 | 0..23 | 范围内 `(A + BASE) % 24`，其余不变 |

CMU 支持 scalar 或 16/32/64 元素比较，产生 unordered/less/equal/greater 条件；IAU 的隐式比较按 signed 32-bit 与零比较。条件码默认覆盖，可由 PEU 的 HOLD/ANDACC/ORACC 等模式调整。[p. 344-345]

**开发含义：**rebasing 可支持软件流水中的寄存器轮换，但它不是新增寄存器容量；谓词代码必须避免无意覆盖仍要使用的条件码。

## 5. 内存路径与寻址

### 5.1 三条通路必须分开

| 路径 | DDR | CMX | 缓存 |
| --- | --- | --- | --- |
| Instruction | 支持 | 不支持 | 指令 L1、共享 L2 路径 |
| Far Data | 支持 | 不支持 | 数据 L1、共享 L2 路径，受 bypass/policy 控制 |
| Near Data | 不支持 | 支持 | CMX 不进入 L1/L2 |

来源：[p. 152, p. 158, p. 336-339]。

**开发含义：**优化 CMX 数据复用应考虑 bank/NoC/LSU，而不是期待 DL1 命中；热代码体积则要考虑 4 KB IL1。不能把“CMX 里放数据”推广成“CMX 里执行指令”。

### 5.2 CMX 与远端 tile

- PTL 每 tile 1536 kB；内存结构描述为 16 banks、32 SRAM，按 32 B 访问粒度交织。[p. 106, p. 133]
- 每个 SHAVE 两个 32 B/cc 低延迟接口；无冲突时 request 到 response 的预期延迟为 8 clocks。[p. 133, p. 148]
- 端口表对每个 SHAVE LSU 列出最多 16 outstanding reads，受 reorder buffer 限制。这是 CMX LSU 路径，不是 DDR L2 的 outstanding 数。[p. 143]
- 本地 SRAM 支持读写；远端 tile 经 ITI 仅支持写；多播和广播读无效，返回零数据并报错。[p. 147, p. 175]
- SRAM 侧总线表要求 32 B 对齐，写操作支持 byte enable。该表是总线接口约束，不能直接断言 C/C++ 每次标量访问都必须是 32 B；具体 LSU 指令如何拆分、对齐需看 ISA。[p. 147]
- LSU 接口面向严格 in-order response 的请求者，由 LSU2NSIP bridge 的 reorder buffer 配合实现。[p. 187]
- `SHAVE ORDER FENCE` 出现在时钟图中，但 HAS 没有在同名章节给出完整内存一致性规则。不能因此推出所有跨 LSU 或跨 SHAVE 访问自动有序。[p. 155, p. 501]

### 5.3 CMX swizzle

原生 swizzle 位于 GIF2NSIP、LSU2NSIP、AXI2NSIP 等桥中，作用于 CMX 内存地址范围，不作用于控制空间。`CMX_SWIZZLE_SETUP` 应在 boot/reset 后设置，不能在运行中随意改动。[p. 140-142]

**开发含义：**bank 冲突分析要结合系统的 swizzle 配置；不能在 kernel 中再凭直觉做一次相同物理 swizzle，否则可能重复变换。

### 5.4 虚拟 CMX 地址与本地窗口

SHAVE 只能通过 `0x4xxx_xxxx` 访问 CMX，不能照搬 RISC-V 的 `0x2Cxx_xxxx` 地址。PTL 使用的 tile-select 位编码为 bitmask，而不只是线性 tile 编号。[p. 120-121, p. 174-175, p. 602]

| PTL 3 × 1.5 MB 配置的例子 | 地址范围/起点 |
| --- | --- |
| SHAVE 所在 tile 的本地别名，tile-select=0 | `0x40000000..0x4017FFFF` |
| Virtual tile 0 | `0x40200000..0x4037FFFF` |
| Virtual tile 1 | `0x40400000..0x4057FFFF` |
| Virtual tile 2 | `0x40800000..0x4097FFFF` |
| Multicast virtual tiles 0+1 | `0x40600000` 起，仅合法写用途 |
| Multicast virtual tiles 0+1+2 | `0x40E00000` 起，仅合法写用途 |

来源：3 × 1.5 MB 列，[p. 174-175]。这些范围还受 job context、实际分配和全局保留窗口约束，并非 kernel 可无条件使用的全部空间。

### 5.5 Local address space 与 DDR window

- SHAVE 自身有 64 KiB local address space，用于控制、诊断、通信寄存器，不是 64 KiB 用户 SRAM。[p. 335]
- LSU 以地址高 nibble 为 0、低 16 bit 为寄存器偏移访问；`0x800..0xFFC` 禁止 LSU 访问；`SVE_LCL_ACCESS` 可按 4 KiB 分组阻断访问，其自身不能由 LSU 配置。[p. 335]
- 四个 window：高字节 `0x1C/1D/1E/1F` 对应 A/B/C/D；地址为 `(addr & 0x00FFFFFF) + OFFSET_WIN_*`，offset 要求 1 KB 对齐。[p. 335]
- SHAVE 的基础地址空间为 32 bit，DDR 映射在 `0x80000000..0xFFFFFFFF`；通过各 window 和非 window 的 page 寄存器扩展到 48 bit。[p. 335-336]
- 扩展公式：`(page_bits << 31) | (address & 0x7FFFFFFF)`；在缓存查找前进行扩展。[p. 336]

**开发含义：**宿主侧/descriptor 的 48-bit 地址不能未经约定直接截断成 SHAVE 的 32-bit 指针。window/page 配置属于执行环境契约。

## 6. L1 缓存

| 属性 | IL1 | DL1 |
| --- | --- | --- |
| 容量 | 4 KB | 1 KB |
| 组织 | 2-way，LRU | direct mapped |
| Cache line | 16 B | 16 B |
| 读写 | 只读指令 | 数据读写，write-back / write-through |
| CMX 数据 | 不适用 | 不缓存 |

来源：[p. 336-338]。

**IL1：**使用前 invalidate，执行 invalidate 时 SHAVE 应 halt 且 `IL1_STATUS.BUSY=0`。Lock 保留已分配 line，miss 时仍可取数据，但已满 set 不再替换；bypass 不更新缓存。[p. 336-337]

**DL1：**默认 bypass，使用前必须 full invalidate；两个请求同时到达时 LSU1 优先。Tag 包含 L2 partition ID，地址相同但 partition 不同仍可能 miss。[p. 337-339]

**DL1 操作规则：**

- write-back 的 dirty line 被替换时先写回；write miss 需要取回 line 并合并写入。
- lock 会立即保护已分配 line，必要时先 flush/invalidate。
- 切换到 write-through 前建议 flush。
- bypass 状态不应下发 flush/invalidate。
- full flush/invalidate 完成前不接收后续请求。

**LSU PFO 家族：**`CPFL1`、`CPFL1L`、`CPFL2`、`CPFL2O`、`CFLUSH`、`CINVAL`、`CINVAL_ALL`、`CFLUSH_ALL`、`CFLUSH_INV_ALL`、`CLOCK_ON/OFF`、`BYPASS_ON/OFF`、`WTHRU_ON/OFF`。[p. 339]

其中 `CPFL2O` 是 optimistic prefetch，只有 cache port 不 stall 时才发出，不能当作有完成保证的数据加载。`CPFL1L` 的临时 line lock 在后续命中访问或 invalidate 时可解除，不应与全局 lock 模式混为一谈。

缓存策略通过 `WIN_CPC/NW_CFG` 配置；启用 secure policy 时使用 `WIN_CPC_SEC/NW_CFG_SEC`。所有策略都携带 L2 partition ID。[p. 339]

## 7. 共享 L2：容量、分区、并行与错误

### 7.1 基本配置

256 KB、64 B line、2-way LRU、write-back；32 个独立配置的 partition；4 个 subgroup；位于 NCE spine，全部 SHAVE 共享。[p. 152, p. 348-352]

每个 partition 的大小是最小粒度的 2 的幂倍数，offset 按大小对齐。默认每个 partition 都映射整份 256 KB；重叠 partition 会共享缓存行并发生竞争。**本 PDF 正文将最小粒度留成 `{SHV_L2C_MIN_PARTITION_SIZE}`，图有 16 KB 示例，但不能仅凭示例保证最小值就是 16 KB。**[p. 350-351]

### 7.2 Subgroup 与 partition 的区别

先由地址确定 subgroup，再在 subgroup 内映射 partition。每个 partition 的 sets 均分到四个 subgroup，不是“一个 partition 独占一个 bank”。[p. 352-353]

HAS 的例子说明：较小 partition 可能使一个连续 32 KiB 地址块内部就产生冲突；四个相隔 32 KiB 的 4 KiB 数据块则可能利用完整的 16 KiB partition。实际效果依赖地址映射，不只是总工作集大小。[p. 353]

每个 subgroup 独立仲裁，使用 modified round-robin；priority 被让出后从实际获胜 source 的下一个 source 继续，以改善活跃请求者数量不均时的公平性。[p. 353-355]

**开发含义：**多个 SHAVE 并行会争用共享 L2/DDR 通路；分区可能降低互相驱逐，却也缩小可用工作集。partition 分配是受信固件职责，不是普通 kernel 自行设置的资源。[p. 634, p. 644-645]

### 7.3 八个 L2 统计计数器

- 每个 32 bit，可两两合并为 64 bit；溢出时饱和，不是自然 wrap。
- 支持按 partition、source、read/write、clean/dirty 过滤的 hit/miss 计数。
- 支持累计 hit/miss cycles、AXI stall、non-blocking capacity exceeded、order tracking capacity exceeded。
- 合并 64-bit counter 先读低 32 bit 触发锁定，再读高 32 bit 解锁恢复计数；写任一配对 counter 也会解锁。
- 这里 `clean hit` 表示不依赖未完成操作，`dirty hit` 表示存在依赖，例如等待之前 miss 的 fill；不是通常说的数据 dirty 位。

来源：[p. 357-358]。这些 L2 counter 与每核 execution counter 是两组不同资源。

### 7.4 ECC、AXI 错误与初始化

- 数据 RAM 支持 SECDED，tag RAM 不做 ECC；ECC 默认开启，开关只应在 boot、工作负载开始前配置。[p. 358]
- 可纠错数据在 RAM 出口修正，默认支持写回纠正；不可纠错时返回零，错误响应是否传回还受配置控制。[p. 359]
- 要将不可纠正错误传给 SHAVE，固件必须配置 `SL2_ICR.UERR` 与 `SL2_IHR.UERR`；AXI 错误类似使用 `AERR`。[p. 359]
- 错误只直接返回触发事务的 SHAVE，固件需显式处理同 context 的其他 SHAVE。[p. 359]
- 上电、退出 reset 后须由固件 invalidate tag memory。[p. 360]
- L2 debug 通过 `SL2_DBG_CTRL/SL2_DBG_DATA` 间接访问 RAM，需要处理 STOP/BUSY；不是用户 kernel 的普通数据通路。[p. 356-357]
- 对 AXI 错误必须同时检查 ERR#061 的 `QOE` 要求，不能只执行主章节的泛化恢复建议。[p. 709]

## 8. 与 DDR、MMU 和宿主缓存的关系

### 8.1 SHAVE DDR 路径不是 DMA 路径

SHAVE L2 miss 与 RISC-V cache/system traffic、fence monitor 共用低吞吐、延迟敏感的 `SOC_AXI_2`；其数据总线为 128 bit，最大事务不超过 64 B。[p. 431-435]

- SHAVE L2 有 4 个 read AxID、4 个 write AxID，每个 subgroup/lane 一个；Top NoC 对它们重映射，避免与其他 initiator 冲突。[p. 432-433]
- SHAVE L2 最多 **32 个 outstanding，读写合计**；不是读 32 加写 32，也不是每个 SHAVE 各 32。[p. 433-434]
- L2 active access 使用完整总线宽度；事务不越过 64 B cache-line 边界，不使用 FIXED/WRAP burst。[p. 435]
- p. 434 的 128 VPU clock 延迟目标和约 1 GB/s 的粗估依赖该接口假设，不能当作 SHAVE 的实测 DDR 峰值。
- §13.4.2 的 L2 throughput/latency 小节为空，没有可引用的实验结果。[p. 615]

HAS 推荐张量数据由 DMA 搬到 CMX，理由是 DMA 更能用 outstanding transactions 隐藏 DDR 延迟。[p. 152] 不应使用旧知识库的“12 vs 256”或固定“低于 DMA 10%”作为本版 PTL 的保证值。

### 8.2 虚拟地址、context 与 coherency

- ACT-SHAVE/DMA 支持 48-bit 虚拟地址相关配置；实际 DRAM 访问受 MMU/SoC IOMMU 和 context 映射约束。[p. 23-35]
- 用户 kernel 不能任意指定可信的 stream/substream/PASID 属性，这些由固件/硬件配置；生产固件不应给 ACT-SHAVE 分配 global context。[p. 35, p. 62]
- SHAVE cache miss、DMA 等内部流量可通过页表属性标记宿主 LLC coherent/non-coherent。[p. 62]
- 宿主 LLC coherency 不等于 SHAVE L1/L2、DMA 与所有缓存自动完全一致；交接和 flush/invalidate 仍要遵守 runtime 协议。
- 错误后仍需接收已发出的 pending responses，避免 reset 引起互连锁死。[p. 34, p. 142]

## 9. 任务分发、barrier 与核心身份

### 9.1 FIFO

| Block | 功能 | PTL 相关尺寸 |
| --- | --- | --- |
| FIFO0 | DPU workload pointer | 每队列 256 × 16 bit |
| FIFO1 | SHAVE IPC transmit | 每队列 64 × 16 bit |
| FIFO2 | SHAVE IPC receive | 每队列 16 × 32 bit |
| FIFO3 | SHAVE workload pointer | 独立模式 128 × 16 bit；共享模式 256 × 16 bit |

来源：[p. 125-129]。实现数量按 tile 数向上取 2 的幂，不能把 padding 出来的 FIFO 实例误当成额外 SHAVE。

`DIM_CFG=0` 为每 SHAVE 独立 FIFO；`DIM_CFG=1` 为同 tile SHAVE 竞争共享 FIFO。HAS 建议使用共享模式，但当前产品 runtime 是否采用需另核查。[p. 128-129]

普通读会 pop，atomic read 可免去先检查状态、空队列返回 0；fill 寄存器支持一次写入最多四项。workload pointer 的 16 bit 存储的是地址片段，不能理解为完整指针。[p. 128-129]

### 9.2 Barrier

- 架构最多 128 个；实际为 `16 × 物理 tile 数`，PTL 默认推得 48 个；每个 job 可用数为 `16 × 分配 tile 数`。[p. 129]
- 每个 barrier 有 8-bit producer count、8-bit consumer count 和各自 zero-hit 状态。[p. 129-130]
- `PDEC/CDEC` 更新计数，多个 barrier 可由一次操作更新。[p. 130]
- 每个 barrier 有 32-deep 初始化 FIFO；producer 和 consumer 同时归零后可自动装载下一项。[p. 130]
- 分为 inference-time 寄存器和仅固件可访问的配置/IRQ 寄存器。[p. 130-131, p. 161]
- HAS 强烈建议 DPU/DMA/SHAVE 之间以 hardware barrier 同步；IPC 同步会损失性能。[p. 131]

**开发含义：**同步设计不仅要保证数值对，还要保证 producer/consumer 次数、资源重用时机、scratch 生存期和退出路径正确。需要结合 ERR#032/#060/#065/#067。[p. 701, p. 708-711]

### 9.3 GPI/GPO 与 P_ID

`P_GPI[31:16]` 暴露一组 16 个 producer-zero 状态；`GPO[2:0]` 选择组，`GPI[15:13]` 返回当前组。`GPI[8]` 为 workload FIFO empty，`[7:3]` 为 IPC fill level，`[2:0]` 为 almost-full/full/empty。[p. 152]

`GPI[10:9]` 反映 job 的 tile 分配编码。不要直接把原始位值当作 tile 数；需解码 `JOB_TILE_NUM`，PTL 编码超出配置上限时会被截到实际 tile 数。[p. 114-115, p. 152]

`P_ID` 包含 ISA class `[31:16]`、tile 总数 `[15:12]`、每 tile SHAVE 数 `[11:10]`、tile ID `[7:5]`、SHAVE 实例 ID `[4]`。[p. 152] 物理身份与 compiler 使用的虚拟 tile 索引不能混用。

## 10. 虚拟资源分配与安全边界

`JOB_X_MAP` 将 job 的虚拟 tile/FIFO/barrier 映射到固件分配的物理资源。物理 tile 不必连续；被熔断或 power-gated 的 tile 必须隔离。编译好的工作负载使用虚拟资源，而不是根据固定物理 tile 编号编译不同程序。[p. 109-125]

ACT-SHAVE 执行不可信用户 kernel，属于 HAS 所称 Ring3 trust model；RISC-V 固件属于受信控制方。这不表示 SHAVE 就是运行 Linux 用户线程的 CPU。[p. 633-636]

- L2 tag 同时匹配地址与 context ID；context-aware caching 默认存在。[p. 160, p. 163, p. 634]
- partition 分配由可信固件完成；SHAVE 只可对授权 partition 请求 cache 操作，不能自由读写共享 L2 配置与全局 profiling。[p. 644-645]
- 越权读返回零并带 error，越权写丢弃，同时向 RISC-V 报告；不能把零输出一律当作算术问题。[p. 125, p. 639]
- 用户态全局控制权限、Sideband Services 和各类管理寄存器受限。[p. 17-18, p. 161-163]

**Context switch/retirement 的职责：**清空相关 FIFO、处理 barrier 状态、invalidate SHAVE DL1/IL1、清理对应 L2 partition、擦除 CMX、清除 profiling 与非复位状态。L2 context ID 重用时的失效尤其重要。[p. 639]

## 11. SHAVE 能不能驱动 DMA / DPU？

不能用“完全可以”或“完全不可以”概括。

1. 普通静态图的 tensor movement 由 compiler 调度 DMA；这是常规路径。
2. DMA 配置章节明确列出 SHAVE-scheduled movement 场景。[p. 237]
3. 安全章节定义 Ring3 job-push：固件预分配 DMA LA，硬件从 requester/context 属性建立授权关系；用户只提交允许的 job 字段。[p. 634]
4. 虚拟 LA ID 为 2 bit，可选择最多四个虚拟 LA；这不是对整个 DMA 的任意控制权限，也不是保证 runtime 一定给足四个。[p. 634]
5. 完成可通过 descriptor 指定位置的完成地址写回通知，仍使用该 job 的 user context。[p. 634]
6. p. 644 的安全概述又使用“不能直接控制 DMA/M2I”的宽泛措辞。应解释为无任意管理权限，并以受控接口定义核查具体调用。

**开发含义：**若开发运行时 tiling 或复合 kernel，需要同时检查 LA 授权、DMA descriptor、completion、barrier 和 preemption 合约。该 PDF 本身没有定义当前代码库的 C++ API、ABI 或 kernel 内 DPU MatMul API，不能据此宣称任意 SHAVE kernel 都能调用它们。

## 12. 数值语义与精度风险

### 12.1 FP32 / FP16

HAS 在 IEEE 754 基础上列出差异：[p. 150-151]

- 默认存在 denormal flushing；某些转换可通过专用位开启 denormal 支持。
- `P_CFG.ILFP16D=1` 影响 SHAVE FP16→FP32 load 与 FP32→FP16 store 的指定 denormal 行为，不是所有算术的通用 denormal 开关。
- `P_CFG.ILFP16C=1` 可使 FP32→FP16 store 的溢出结果 clamp 到最大有限值。
- 常规 NaN 输出采用规范化形式，不保留 payload；`CMU.MAX/MIN/MAXMIN` 对双 NaN 有例外，max 返回 A，min 返回 B。
- tininess 的检测时机也有明确约定，精度参考应匹配这些规则。

### 12.2 BF16 / FP8

- BF16→FP32 load 和 FP32→BF16 store 明示 flush input denormal to zero。[p. 151]
- ERR#012 讨论 VPU5 对 BF16 denormal 的默认变化，并引用 DPU 的 `mpe_daz/ppe_fp16_ftz`。不要将 DPU 行为直接套到 SHAVE 的转换指令。[p. 696]
- FP8 使用 BF8/HF8 相关定义；BF8 input QNaN/SNaN 的 invalid exception 行为有专门说明。[p. 151]
- PTL SHAVE 更新明确是 FP8 load/store instruction support；具体转换、rounding、算术支持与吞吐需外部 ISA 确认。[p. 335, p. 348]

### 12.3 异常与 carry

整数异常包括 unsigned overflow、signed positive/negative overflow、upper/lower saturation、divide by zero；只有列出的特定指令更新对应状态，不是任意整数溢出都会自动报告。[p. 345-347]

`V_STS/S_STS/I_STS/C_STS` 汇总运算异常，状态为 sticky，需要 `CMU.CPIT` 或 DCU 显式清除；不能将上个 kernel 留下的位误认为本次产生。异常 flag 也不等于必然触发 CPU 式 trap。[p. 347, p. 682]

VAU 按 32-bit lane 存储 carry，SAU/IAU 有各自 carry 字段；subtract 中可解释成 no-borrow。carry-producing 指令清单见 [p. 347-348]。

**推荐精度用例（推论）：**非对齐尾部、极小 denormal、正负零、NaN/Inf、溢出边界、量化饱和、signed/unsigned load、累加误差，以及 kernel 间状态清理。

## 13. 中断与抢占

SHAVE 可接收 IRQ 跳入 ISR；PTL 增量明确提到 RISC-V 通过 IRQ 请求 SHAVE preemption。[p. 6, p. 340]

下列情况暂不能响应：[p. 340]

- 正执行 branch 或 branch delay slot。
- 正执行 `BRU.RPI/RPIM`。
- 已在 ISR 中。
- 当前指令被 `B_CFG.PROT` 保护。

循环必须含有可中断的非 branch / 非 delay-slot 指令；不能为了压缩循环而无意让它长期不可抢占。软件可通过向 `B_PRE` 写非零触发请求。

进入 ISR 时保存返回地址并置 active 状态，完成时应清除 `B_STS.ACT`。正文还混有旧 `ISR_SA/ISR_RA/ISR_STAT` 命名；TRF 表给出 `B_ISR_SA/B_ISR_RA/B_STS/B_CFG`，实施时应以 PTL register map 为准。[p. 340, p. 342-343]

**边界：**硬件 IRQ 支持不等于完整 kernel 状态保存/恢复 ABI。VRF、IRF、TRF、DMA in-flight 和 scratch 如何保存由软件协议决定，需 SAS/runtime 补充。

## 14. 自时钟门控、电源与复位

### 14.1 Kernel 等待时的 self clock gating

`GPO[7]` 使能 self clock gating，其他 GPO 位选择 barrier/FIFO wake 条件；只有 cache 不 busy 且无 pending requests 时硬件才停钟。kernel 等待 `GPI[11]` awaken，醒来后清使能、等待 awaken 撤销，再检查真正的依赖是否满足。[p. 152-155]

**barrier 状态和 FIFO non-empty 必须使用 LEVEL HIGH 唤醒，不得用 rising/falling edge。** 否则条件可能在开启门控前已成立，导致丢失唤醒。[p. 154-155]

APB 访问也可能唤醒，醒来不代表任务依赖已经满足。示例中存在 GPI/GPO 命名和注释不一致，不能不核对寄存器定义就直接复制。[p. 153-155]

### 14.2 时钟与电源

每个 SHAVE 是独立 power-gated domain，与 tile、DPU 的电源层级有关；共享电源轨不等于所有核心必须同开同关。[p. 199]

PTL 的 `NCE_ACT_SHAVE_0..5_CLK_EN` 对应六个核心。GALS/clock-domain crossing 会影响延迟，不能把所有章节的 cycle 一律按 DPU frequency 换算。[p. 478-501]

上电前 tile 需已供电；HAS 要求固件一次只给一个 ACT-SHAVE 上电，避免 di/dt 事件。[p. 549-553]

下电前 drain outstanding reads，再执行必要的 APB read、时钟门控、隔离、reset 和 power switch 顺序；不能直接断电。[p. 558-559]

### 14.3 Reset-only 与清理代价

ACT-SHAVE 可单独 reset；共享 L2 不能独立 reset，要随 NCE spine/interface 处理。[p. 580]

reset-only 不会自动清除 IL1/DL1 data RAM、IRF、VRF 等状态；HAS 列出 1824 个 32-bit 写入单元需清理。文档给出单 SHAVE 58713 cycles / 约 108 us @543 MHz 的流程估计，不是当前机器实测。[p. 581, p. 587-588]

该页的 12-SHAVE/6-tile 累计数据是遗留配置，不应作为 PTL 总延迟。HAS 推荐 SHAVE power-down/up 代替 reset-only；p. 586 明确说 reset-only 仅作参考，固件不应实现该序列作为推荐流程。

## 15. 性能：能确定什么，不能确定什么

### 15.1 可引用的数据

| 项目 | HAS 给出的信息 | 不能据此推出 |
| --- | --- | --- |
| SIMD 宽度 | 512 bit [p. 333] | 所有 FP16 指令都是每周期 32 个结果 |
| CMX LSU 端口 | 每核两个 32 B/cc [p. 133] | 任意地址模式都能持续达到合计 64 B/cc |
| CMX 读延迟 | 无冲突预期 8 cycles [p. 133, p. 148] | 任意运行负载下都是 8-cycle load-to-use |
| LSU reorder 容量 | 每端口最多 16 outstanding reads [p. 143] | 16 个 DDR miss 或 16 个 GPU warp |
| L2 outstanding | 全共享实例读写合计 32 [p. 433] | 每 SHAVE 独享 32 或读写各 32 |
| SOC_AXI_2 延迟 | 128 VPU cycles 目标与假设性估算 [p. 434] | 实机 DDR 固定 latency/GB/s |
| L2 实验数据 | 专门小节为空 [p. 615] | 已有可引用的 PTL 实测结果 |

如果两条 LSU 都能满载，则端口量级上限是 `64 B/cycle × f`，但实际 read/write 组合共享这些端口，还受 NoC、bank、指令排程和竞争影响。f 必须取相关路径实际时钟，不应随意代入 DPU 时钟。

### 15.2 优化方向（工程推论）

1. 数据尽量落在本地 CMX，避免 bulk tensor 依赖 far-data DDR cache miss。
2. 让内层循环连续向量化，降低索引、分支和不规则 gather 的占比。
3. 提前加载并交错独立计算，覆盖 CMX load latency，同时限制寄存器压力。
4. 按真实 port-sharing 规则安排 VLIW，不能只看单元空闲。
5. 控制展开与融合后代码大小，避免 4 KB IL1 成为新瓶颈。
6. 多 SHAVE/多 tile 同时考虑 scratch 隔离、barrier 和共享 L2/DDR 竞争。
7. 先看关键路径的 cycles、stall 和搬运，再决定是否写 ASM；仅靠理论 SIMD 宽度判断没有意义。

这些不是本次性能测试结果，本次没有编译或运行 kernel benchmark。

## 16. Profiling 与调试

### 16.1 每核与共享计数器

每个 SHAVE 有 8 个 program-execution performance counters，用于 stall 等统计，支持配对获得 64-bit view；L2 另有 8 个共享统计 counter。[p. 617, p. 357-358]

compiler-scheduled profiling 要做三件事：在 CMX 分配记录区、配置各任务 profiling 地址与使能、插入周期性 DMA 将记录搬到 DRAM。runtime-generated tasks 的 profiling 小节仍带 OPEN 问题。[p. 618]

### 16.2 时间戳不是核心 cycles

AON timer 是分布式 64-bit 时间基准，默认每两个 `perf_clk` ticks 增加一次，divisor 可配置。SHAVE 经 NCE timer/control bus 获取同步的时间值。[p. 618-620]

**开发含义：**profiling timestamp delta 和 SHAVE core cycle count 不应直接混用；DVFS、timer divisor、同步与测量开销需要记录。给 benchmark 报告时应说明时钟和计时口径。

### 16.3 DCU 硬件能力

- 每核一个 DCU，提供 start/stop、IRQ 控制、浮点 trap 控制、寄存器访问与取指缓冲控制。
- 2 个硬件 instruction breakpoints、1 个 software instruction breakpoint、2 个 hardware data breakpoints。
- 支持单步/多步、异步 halt 和 execution counters。
- `SVU_IH[15:0]` 保存最近 16 条指令，`SVU_BTH` 记录最近 16 次 branch 的 taken 状态。
- 没有直接的 SHAVE 硬件 trace streaming，需 debugger 或 RISC-V 轮询这些历史寄存器。

来源：[p. 617, p. 671, p. 681-683]。因此“SHAVE 硬件没有 debugger”不准确；能否在当前板卡使用依赖 debug authorization、固件、驱动与工具。

断点观察到的执行位置不是统一的精确 source-level 边界：HAS 列出 SW BP 为 Execute-1，instruction BP 为 Execute+0；data BP 根据访问和比较方式可能在 Execute+2/+7；halt-on-branch 为 Execute+1。排查内存破坏时要考虑流水已经前进。[p. 683]

### 16.4 DFx 范围

debug/cross-trigger 由 SoC 授权控制，DCU 访问不应被当作量产 kernel 的任意能力。DFx 章节同时讨论其他代际 latch-array debug；NPU5 列出的 latch arrays 在 `nce_scl`，不能把 NPU6/7 的 `nce_shave` 描述反推成 PTL 特性。[p. 649-658, p. 665-671]

## 17. 错误、hang 与恢复

CMX 路径发生不可纠正 ECC、context violation、非法访问或 power-gated tile 访问时，LSU2NSIP 将 error 传给 SHAVE。SHAVE halt execution，但继续接收 pending responses；该路径的恢复要求 NCE subsystem reset。[p. 142-143]

DDR/L2 错误要确认 AERR/UERR、halt-on-interrupt 和 ERR#061 配置，否则可能出现“错误已发生但核心没按预期停止”。受影响范围可能大于单 kernel 或单 SHAVE。[p. 359, p. 709]

**排查顺序（工程建议）：**

1. 检查地址是否落在合法 local/virtual/window/page 范围，以及 context 是否正确。
2. 检查是否错误地读取远端 CMX 或 multicast 地址。
3. 查看 barrier producer/consumer 和 FIFO 是否满足进度条件。
4. 查看 core halt/status、DCU 最近指令与 L2/CMX 错误来源。
5. 确认是否受 power state、preemption 或已知 errata 影响。
6. 在证实输入、指针、同步与错误状态后，再怀疑 ISA/硬件问题。

## 18. 直接相关与相邻 errata

以下是对 SHAVE 开发、调度或协同执行有直接意义的项目，不是全体 DPU errata 的替代清单。是否适用于某 stepping，需结合原文 affected/fixed 字段及固件版本确认。

| Erratum | 影响 | HAS 中的处理要求/边界 | 来源 |
| --- | --- | --- | --- |
| ERR#032, 18038474056 | SHAVE barrier decrement 因 byte-enable 涉及未分配 barrier 而误报 context violation，exit code 0x1F | HAS 表示正常 SHAVE compiler 生成的写法不会触发，不需普通 mission-mode workaround；手写访问仍需核查；NPU6 修复 | p. 701 |
| ERR#061, 18043299791 | L2 收到系统内存错误后可能保留 valid 状态和旧数据，存在隔离风险 | `SL2_ICR.AERR=1`、`SL2_IHR.AERR=1`、`SL2_CTRL.QOE=1`；恢复可能影响并行 context | p. 709 |
| ERR#065, 18042574402 | barrier IRQ 过载影响其他 context 的进度 | 限制每 inference completion IRQ 的 workaround 不适用于 multi-tenancy，后者需其他方案；NPU7 修复 | p. 710 |
| ERR#067, 18044200741 | legacy barrier FIFO 零计数项后紧跟非零项可能死锁 | preemption restore 流程将 PCOUNT=0 调整为 1，由 runtime 补偿 PHIT；不要直接修改普通图语义 | p. 710-711 |
| ERR#060, 18043204508 | legacy barrier producer=0 时可能不产生 IRQ | PHIT 状态不受影响，HAS 说现有 SW use case 不受影响 | p. 708 |
| ERR#012, 18034753935 | VPU4/VPU5 BF16 denormal 默认行为不同 | 指定 DPU MPE/PPE 控制位；不能覆盖 SHAVE 转换的独立规则 | p. 696 |
| ERR#013, 18035819384 | DMA LA 从 HALTED 恢复后后续 halt 可能死锁 | abort active job 并 context flush，不支持直接 resume；NPU6 修复 | p. 696-697 |
| ERR#010, 18032052817 | CMX 连续 scrub 且无中间访问可能死锁 | 固件按 bank 执行指定写/读 workaround；NPU6 修复 | p. 695 |
| ERR#029, 18039428591 | D0i2idle→D0active 的电源域 DFx reset 问题 | PTL A0 受影响，B0+ 不受影响；须 FW/KMD workaround | p. 700 |
| ERR#035, 18039233178 | BitC 前一错误 descriptor 的状态影响下一任务 | HAS 要求 application context switch/preemption 时 engine reset；NPU6 修复 | p. 702 |
| ERR#064, 18043444473 | DPU 门控前未返回 credit，导致接口锁死 | compiler 配置 `noc_clk_en=1`，FW 在 process switch 做 CMX tile reset；NPU7 修复 | p. 709 |
| ERR#001, 18031255628 | DPU profiling 可能写错 CMX 地址，进而影响 SHAVE 数据 | 按 HAS 使用 HWP_WLOAD_ID 对应地址方案 | p. 690 |
| ERR#006, 18031391267 | DMA decompression/weight preparation 的 JWDONE_TIME 过早 | end-to-end latency 可用 JFINISH_TIME；NPU6 修复 | p. 693 |

## 19. 文档矛盾、遗留项与知识库修正

| 问题 | 本次处理 |
| --- | --- |
| 通用配置写“最多四个 SHAVE/tile”，PTL 表写两个 | 产品画像采用 PTL 专用表 p. 106，不拿通用上限当 PTL |
| p. 604-606 地址图画到 tile 5，p. 581 复位例子按 12 SHAVE 累计 | 标为遗留/跨平台图例，不作为 PTL 资源依据 |
| p. 155 标题为 Cross LSU Port Ordering，正文却是 APB broadcaster | 不据此声称自动保证跨 LSU 顺序；p. 187 只证明 LSU response reorder 机制 |
| p. 149-150 性能表有 remote SHAVE read 列，但 p. 147/175 明确仅远端写 | 不认为表格列名赋予 remote-read 能力，遵循显式功能限制 |
| p. 336 window B 例子却标 OFFSET_WIN_D | 不抄示例字段名，遵循 p. 335 window 映射表 |
| p. 340 中断章节混旧 TRF 名称与反馈记录 | 以 TRF 表及外部 PTL register map 核对，不直接复制旧名 |
| p. 153 注释把清 GPO[7] 写成 GPI[7] 等 | 标为示例文字问题，按 p. 154 流程及寄存器定义核查 |
| p. 350 L2 最小 partition 粒度仍是模板占位符 | 16 KB 只作为图例，不宣称为已验证最小值 |
| p. 163 将错误使能提到 ISR，p. 359/709 用 ICR/IHR | 实现错误处理按专门错误章节、errata 与 register map 核查 |
| p. 634 受控 DMA push 与 p. 644 “不能控制 DMA”概述有张力 | 区分受控 job submission 与任意管理寄存器访问 |
| p. 615 L2 性能小节无结果 | 不补造吞吐/延迟，不借用相邻 CPU/MMIO 数据 |
| 旧知识库的 DDR outstanding=12 或固定 DMA 比例 | 本版 p. 433 明确共享 L2 合计 32；不沿用旧数字 |
| 泛化说“没有 debugger” | 硬件有 DCU 断点/单步/trace；软件可用性另查 |
| 把 BF16/FP8 “支持”当作所有执行单元统一语义 | 分别确认 DPU、SHAVE load/store、SHAVE arithmetic |

## 20. 面向开发的检查表

### Kernel 作者

- [ ] 确认实际可用 SHAVE 数、工作分片和 kernel 参数协议，而非硬编码全芯片资源。
- [ ] 数据主要在本地 CMX，跨 tile 访问不做非法 remote read。
- [ ] 区分宿主地址、SHAVE 32-bit 指针、48-bit window/page 和 virtual CMX 地址。
- [ ] 向量尾部、对齐、byte enable、stride 和 scratch 大小正确。
- [ ] VLIW bundle 合法，寄存器/条件码/carry/异常状态的生命周期明确。
- [ ] 指令展开不过度膨胀 IL1 工作集；软件流水覆盖实际 load latency。
- [ ] FP16/BF16/FP8 的转换、denormal、NaN、clamp 和精度阈值已验证。
- [ ] 长循环可抢占；不滥用 protected instruction 或不可中断 repeat。
- [ ] 受控 DMA 使用获得授权的 LA，并满足 completion/preemption 协议。

### Compiler / Runtime / Firmware

- [ ] 使用 job 虚拟资源，检查 tile/FIFO/barrier 分配与释放。
- [ ] producer/consumer 计数、barrier 重用和 FIFO 模式与执行计划一致。
- [ ] 输出可见性、DMA 完成与 consumer 启动顺序正确。
- [ ] cache 策略、partition、初始 invalidate 与 context retirement 由正确权限方完成。
- [ ] L2 AERR/UERR/QOE、同 context halt、错误恢复满足 errata。
- [ ] power/reset 按推荐序列，不能依赖 reset 自动清掉全部 kernel 状态。
- [ ] profiling 区域不与 tensor/scratch 冲突，并及时搬回 DRAM。
- [ ] 区分 core cycles、固定时间基准和实机频率，再比较性能。

## 21. 继续深入所需资料

本 PDF 没有提供下列完整信息，本次不作补造：

1. **SHAVE512 ISA**：完整指令、编码、latency/throughput、寄存器 hazard、LSU conversion 细节、FP8 算术能力。[p. 348]
2. **PTL Register Map**：cache policy、DCU counter event、IRQ、窗口、DMA job-push、clock/reset 等寄存器的最终定义。[p. 606]
3. **SAS / runtime ABI**：kernel entry 参数、stack、调用约定、preemption save/restore、context retirement、cache 交接协议。[p. 559]
4. **板卡 profiling**：实际频率、cycles、LSU stalls、L1/L2 miss、DMA overlap、并发和端到端模型效果。[p. 617-620]
5. **Stepping/HSD 与固件版本**：确认 errata 修复和 workaround 的实际启用情况。

PDF 中提取出的关键外部入口，尚未访问：

- [SHAVE512 ISA](https://docs.intel.com/documents/MovidiusExternal/vpu5/PTL/has/SHAVE512_ISA.html)
- [PTL VPU5 Register Map](https://docs.intel.com/documents/MovidiusExternal/vpu5/PTL/has/VPU5_Register_Map.html)
- [HAS 引用的 Context Retirement Flow](https://docs.intel.com/documents/MovidiusExternal/vpu4/Common/SW/VPU4SAS.html#context-retirement-flow)

## 22. 提取产物与复现

- [extract_has.py](extract_has.py)：提取脚本。
- [pages.json](pages.json)：按 PDF 页码索引的原始文本列表，Python 下标为页码减一。
- [has.txt](has.txt)：含 `===== PDF PAGE N =====` 的可检索全文。
- [evidence-index.md](evidence-index.md)：207 个关键词命中页的导航。
- [shave_hits.json](shave_hits.json)：完整匹配行。
- [address-map-ocr.json](address-map-ocr.json)：p. 604-606 的本地 OCR、坐标与置信度；十六进制可能识别错误，不能直接用于编程。
- [external-links.json](external-links.json)：相关页中的外部引用，不表示已阅读链接内容。
- [provenance.json](provenance.json)：来源哈希、PDF metadata、版本和页数。

运行环境为用户的 micromamba `qwen35`，安装了 `pymupdf` 与 `rapidocr_onnxruntime`；未修改系统 Python，PDF/OCR 均在本地处理。

```bash
micromamba run -n qwen35 python temp/npu5-shave/extract_has.py --ocr
```

**总结：PTL ACT-SHAVE 是围绕本地 CMX 优化的少量 VLIW 向量 DSP 核。发挥性能的关键不是模仿 GPU 启动大量线程，而是选好计算分工、安排向量/访存流水、控制寄存器与代码工作集，并通过 compiler/runtime 把 DPU、DMA、barrier 和资源隔离组织好。**