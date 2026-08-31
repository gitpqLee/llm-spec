# NPU GDN Asynchronous Optimization Proposal

We suggest optimizing NPU GDN in stages, starting with the lowest-risk changes.

## 1. Intra-Chunk DPU/SHAVE Overlap

Currently, SHAVE waits immediately after each DPU MatMul. We can split the synchronous operation into `run -> independent work -> wait`.

```text
Current:
DPU QK^T -> wait -> SHAVE compute R -> fold decay

Proposed:
DPU:    |------ QK^T ------|
SHAVE:     |-- compute R --|
Sync:                       wait -> fold decay
```

`QK^T` and `R` use disjoint buffers, so they can execute concurrently. This requires no additional buffers and does not change the algorithm, making it the easiest first optimization.

Other similar opportunities include:

- Overlap the state-update DPU MatMul with SHAVE output writeback.
- Overlap the `KK^T` DPU MatMul with Q/H0 FP16 staging.

## 2. Inter-Chunk DMA Double Buffering

Inspired by FlashQLA's TMA and shared-memory pipeline, we can use DMA with two CMX input stages:

```text
Time       T0            T1                  T2
---------------------------------------------------------
DMA        load C0       load C1             load C2
Compute                  compute C0          compute C1
Buffer     Stage A       A compute/B load    B compute/A load
```

While processing chunk $i$, DMA prefetches chunk $i+1$, hiding DDR-to-CMX transfer latency.

However, GDN has a recurrent-state dependency:

$$
H_{i+1}=f(H_i,\mathrm{chunk}_i)
$$

The next chunk may prefetch its inputs and compute state-independent operations such as normalization, gates, $KK^T$, and $QK^T$. Operations such as $KH_0$, $QH_0$, solve, output, and state update must wait for the previous chunk to produce the new state.

## Recommended Order

```text
Phase 1: DPU QK^T || SHAVE compute R
Phase 2: Add other intra-chunk DPU/SHAVE overlaps
Phase 3: Add DMA + CMX A/B input buffering
Phase 4: Evaluate inter-chunk KK^T/QK^T precomputation
```

The first phase only moves the DPU wait point, so it has the lowest implementation risk. Inter-chunk DMA ping-pong is closer to FlashQLA's pipeline, but requires changes to CMX allocation, DMA descriptors, and potentially compiler/runtime interfaces.
