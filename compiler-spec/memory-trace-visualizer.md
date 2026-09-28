# Compiler CMX Memory Trace 使用指南

本文介绍如何生成并使用 NPU Compiler 的 CMX Memory Trace，以及如何分析 buffer 生命周期、内存碎片、spill 和 DMA/DPU 流水。

## 1. 工具位置

可视化工具位于 VPUX Plugin 仓库：

```text
scripts/malloc_trace_vis/malloc_trace_vis.html
```

它是单文件浏览器应用，不需要安装 Python 包或启动后端服务。输入文件是 compiler 生成的 `mallocTraceRaw_*.json`。

## 2. 生成 Memory Trace

Memory Trace 由 `feasible-allocation` 阶段生成，仅在 Debug/Developer build 中可用。编译模型前设置：

```bash
export IE_NPU_ENABLE_SCHEDULE_STATISTICS=1
```

然后正常执行 `compile_tool`。输出文件会生成在当前工作目录，因此建议先进入专门的结果目录：

```bash
mkdir -p /path/to/trace-output
cd /path/to/trace-output

export IE_NPU_ENABLE_SCHEDULE_STATISTICS=1
/path/to/compile_tool \
    -m /path/to/model.xml \
    -o model.blob \
    -d NPU.5010 \
    -log_level LOG_INFO \
    -c compiler.conf
```

Qwen 2B 示例：

```bash
cd /home/pengqian/vpux/scripts/qwen35
./qwen-2B.sh
```

该脚本已经设置 `IE_NPU_ENABLE_SCHEDULE_STATISTICS=1`。

### 避免混用旧文件

重新编译会覆盖同名 trace。运行前应移动或备份旧结果：

```bash
cd /home/pengqian/vpux/scripts/qwen35

stamp=$(date +%Y%m%d-%H%M%S)
backup="trace-$stamp"
mkdir "$backup"
mv mallocTraceRaw_*.json compileTimeScheduleTrace.json "$backup"/ 2>/dev/null || true

./qwen-2B.sh
```

完成后使用 `stat` 确认文件时间，避免分析残留输出：

```bash
stat -c '%y %s %n' mallocTraceRaw_*.json
```

## 3. 输出文件说明

常见输出如下：

```text
mallocTraceRaw_main?t_Func.json
mallocTraceRaw_main?t_Func_part1.json
mallocTraceRaw_main?t_Func_part2.json
...
compileTimeScheduleTrace.json
```

### `mallocTraceRaw_*_partN.json`

这是 CMX Memory Trace，也是 visualizer 的输入。模型经过 outlining 后，每个函数生成一个独立 JSON；`partN` 不是同一时间线的第 N 段，不应拼接。

### 不带 `partN` 的主文件

通常只包含顶层函数调用，没有 CMX buffer，一般不用于内存分析。

### `compileTimeScheduleTrace.json`

这是全模型的 compiler schedule simulation，不是 CMX Memory Trace。它使用 Google Trace Event Format，可在 [Perfetto](https://ui.perfetto.dev/) 中打开，用于查看整体 latency、DMA/DPU/SW 时间及 overlap。

## 4. 查找算子所在的 Part

当输出包含很多 part 时，根据 `op_loc` 搜索：

```bash
cd /path/to/trace-output
grep -l 'layers.6.mlp.down_proj' mallocTraceRaw_*.json
```

文件名中的 `?` 是普通字符，但 shell 中最好给路径加引号：

```bash
'mallocTraceRaw_main?t_Func_part33.json'
```

Qwen 2B 当前模型的示例映射：

| Trace | 主要内容 |
|---|---|
| `part6` | Layer 3 MLP/down_proj |
| `part15` | Layer 4 MLP/down_proj |
| `part24` | Layer 5 MLP/down_proj |
| `part33` | Layer 6 MLP/down_proj |

该映射取决于模型和 compiler pipeline，不能作为固定 ABI。每次应以 `grep` 结果为准。

## 5. 打开 Visualizer

### 本地桌面环境

直接在浏览器中打开：

```text
/path/to/vpux-plugin/scripts/malloc_trace_vis/malloc_trace_vis.html
```

点击顶部文件区域，或将 `mallocTraceRaw_*.json` 拖入页面。

### Windows 本地使用（推荐，无需端口转发）

`malloc_trace_vis.html` 是自包含文件。将它下载到 Windows 一次，以后双击即可运行；工具更新时只需替换这个 HTML。

Windows 文件名不能包含 `?`，因此先在开发机上复制目标 trace 并改成 Windows 合法名称：

```bash
cd /home/pengqian/vpux/scripts/qwen35

cp 'mallocTraceRaw_main?t_Func_part33.json' updated-part33.json
cp \
  'profile-baseline-20260924-124201/mallocTraceRaw_main?t_Func_part33.json' \
  baseline-part33.json
```

在 Windows PowerShell 中，通过 `scp` 下载 HTML 和 JSON：

```powershell
scp pengqian@<开发机地址>:/home/pengqian/vpux/repo/applications.ai.vpu-accelerators.vpux-plugin/scripts/malloc_trace_vis/malloc_trace_vis.html $HOME\Downloads\

scp pengqian@<开发机地址>:/home/pengqian/vpux/scripts/qwen35/updated-part33.json $HOME\Downloads\
```

如果 Windows SSH config 已配置与 VS Code 相同的主机别名，例如 `vpudev`，可以直接使用：

```powershell
scp vpudev:/home/pengqian/vpux/scripts/qwen35/updated-part33.json $HOME\Downloads\
```

也可以使用 WinSCP 下载文件。之后的使用流程是：

1. 在 Windows 双击 `malloc_trace_vis.html`；
2. 将 JSON 拖到页面顶部的 `Drop mallocTraceRaw.json here` 区域；
3. 或点击该区域，通过文件选择器选择 JSON。

JSON 不是通过 Windows 的“打开方式”与 HTML 关联。必须先打开 HTML，再在页面中加载 JSON。比较两个结果时，可打开两个浏览器标签页，分别加载 baseline 和 updated JSON。

### VS Code Remote 环境

浏览器可能无法直接访问远端 `file://` 路径，可以启动临时只读 HTTP 服务：

```bash
cd /home/pengqian/vpux
micromamba run -n qwen35 python -m http.server 8767 --bind 127.0.0.1
```

然后通过 VS Code 端口转发打开：

```text
http://127.0.0.1:8767/repo/applications.ai.vpu-accelerators.vpux-plugin/scripts/malloc_trace_vis/malloc_trace_vis.html
```

浏览器文件选择器运行在本地机器时，可能不能直接选择远端 Linux 文件。这种情况下可将 trace 下载到本地后选择，或使用能访问远端文件系统的浏览器环境。

## 6. 页面布局与操作

### CMX Memory State

左侧显示当前 cycle 的 CMX 快照：

- `new`：刚分配的区域；
- `active`：当前存活的 buffer；
- `freed`：刚释放的区域；
- `free space`：当前未占用区域；
- `spill-write` / `spill-read`：spill 写出和读回。

拖动顶部 cycle slider 可观察内存随时间变化。鼠标滚轮缩放，拖动平移。

### Allocation Timeline

横轴是 cycle，纵轴是 CMX 地址。每个矩形表示一个 buffer 的地址范围和生命周期。

可搜索：

- buffer ID；
- buffer name；
- 关联 operation 的 `op_loc`。

### Execution Timeline

显示 DPU、SHAVE 和 DMA executor/lane 上的 operation。可按 operation ID 或 `op_loc` 搜索，例如：

```text
layers.6.mlp.down_proj/ov_ext::linear/MatMul
```

点击 operation 会高亮其输入输出 buffer；点击 buffer 会高亮所有 reader/writer。按 `Escape` 清除固定选择。

## 7. JSON 字段

### Buffer

```json
{
  "id": 18,
  "lo": 1098240,
  "hi": 1491456,
  "name": "optional IR value name"
}
```

- `id`：本 trace 内的 buffer ID；
- `lo`：CMX 起始地址，包含；
- `hi`：CMX 结束地址，不包含；
- buffer 大小为 `hi - lo`；
- `name`：可选的 IR 名称。

### Operation

```json
{
  "op_id": 61,
  "op_loc": "human-readable IR location",
  "cycle_begin": 681997,
  "cycle_end": 691679,
  "executor": "DPU",
  "executor_mask": 1,
  "inputs": [17, 18, 16],
  "outputs": [20],
  "spill_id": 0
}
```

- `cycle_begin` / `cycle_end`：模拟调度周期；
- `executor`：`DPU`、`SHAVE_ACT`、`DMA_NN_CH_0`、`DMA_NN_CH_1` 等；
- `executor_mask`：executor lane 位掩码，`1` 为 lane 0，`2` 为 lane 1，`3` 为 lane 0 和 lane 1；
- `inputs` / `outputs`：关联 buffer ID；
- `spill_id`：可选，同一 ID 的 spill-write/spill-read 构成一对。

Buffer 生命周期由所有引用它的 operation 推导：

```text
[min(cycle_begin), max(cycle_end))
```

## 8. 常见分析方法

### 8.1 判断内存碎片

在目标 cycle 查看所有 active buffer，将地址排序后计算相邻区间的空洞。

需要区分：

- 总空闲内存是否足够；
- 最大连续空闲区是否足够。

例如第二个 weight tile 需要 393,216 B。即使总空闲为 550 KiB，若最大的单个空洞只有 240 KiB，仍然无法分配。

### 8.2 判断 Ping-pong/Double Buffering

不能只看不同的 buffer ID，必须同时满足：

1. 相邻 tile 使用两个不同的物理地址区间；
2. 两个 buffer 的生命周期发生重叠；
3. 地址按 A/B/A/B 交替复用；
4. 下一片 DMA 与当前片 DPU 重叠；
5. DMA 通常在 lane 0/lane 1 之间交替。

典型形态：

```text
DMA lane 0: load weight A0 -------- load weight A2
DMA lane 1:    load weight B1 -------- load weight B3
DPU:                       tile 0 | tile 1 | tile 2
```

如果所有 weight tile 都使用同一物理地址，只能等当前 DPU 完成后再覆盖该地址，属于单缓冲。

### 8.3 分析 DMA/DPU 流水

重点比较：

- DPU task 的启动间隔；
- DPU 自身 duration；
- DMA 是否位于相邻 DPU task 之间；
- DMA-DPU overlap；
- DMA lane 是否均衡；
- DPU 是否存在长 idle gap。

若 DPU duration 不变，但启动间隔显著缩短，通常说明收益来自数据搬运与计算重叠，而非计算本身变快。

### 8.4 分析 Spill

Visualizer 会使用 `spill_id` 关联 spill-write 和 spill-read。点击带虚线边框的 buffer，可跳转到 partner。

判断 spill 是否值得时，应比较：

```text
spill 额外开销 vs. spill 腾出 CMX 后消除的 DMA/DPU stall
```

少量 spill 可能换来更有效的 double buffering，因此不能仅以 spill 数量判断退化。

## 9. 新旧结果对比

生成新结果前先保存 baseline：

```bash
mkdir -p baseline
cp -a mallocTraceRaw_*.json compileTimeScheduleTrace.json model.blob baseline/
```

修改 compiler 并重新编译模型后，分别打开两个 visualizer 页面：

```text
baseline/mallocTraceRaw_...partN.json
mallocTraceRaw_...partN.json
```

在两个页面搜索同一个 `op_loc`，比较：

1. operation 和 buffer 数量；
2. trace 总 cycles；
3. 目标 operation 的起止 cycle；
4. buffer 地址、大小和生命周期；
5. DMA lane 使用；
6. DMA-DPU overlap；
7. spill 数量和开销；
8. 最大连续空闲区和内存碎片。

不要仅比较 JSON 文本哈希。JSON 数组或对象输出顺序可能变化，应按 `op_id` 和 buffer `id` 做语义比较。

## 10. 查看整体 Schedule

将 `compileTimeScheduleTrace.json` 拖入 [Perfetto](https://ui.perfetto.dev/)，或直接查看其中的 `taskStatistics`：

- `total duration`；
- `DMA duration`；
- `DPU duration`；
- `SW duration`；
- `DMA-DPU overlap`；
- `DMA duration without overlaps`；
- `total idle`。

Compiler schedule trace 是基于 VPUNN cost model 的模拟结果，适合定位调度变化，但不等同于真实硬件 latency。最终性能结论应使用 NPU 硬件 profiling 验证。

## 11. 常见问题

### 页面显示 `No trace loaded`

需要在页面顶部选择 `mallocTraceRaw_*.json`，不能选择 `compileTimeScheduleTrace.json`。

### 文件很多，不知道打开哪个

使用 `grep -l '<op_loc substring>' mallocTraceRaw_*.json` 定位 part。

### 修改代码后结果没有变化

检查：

1. compiler/plugin 是否重新构建；
2. `compile_tool` 是否加载了新 build；
3. trace 时间戳是否为本次运行；
4. strategy cache 是否真正命中目标层；
5. 是否误读了旧目录中的 JSON；
6. 浏览器是否重新加载了文件，而不是保留内存中的旧 trace。

### Trace 中 cycle 和硬件实测不一致

这是预期风险。Trace 中 operation cost 来自 VPUNN，部分 DMA、SHAVE 或特殊 layer 的 cost 可能不准确。Memory Trace 仍可用于分析地址、生命周期、依赖和调度结构，但 latency 应以硬件数据为准。