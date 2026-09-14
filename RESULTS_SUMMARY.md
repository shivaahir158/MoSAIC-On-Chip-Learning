# MoSAIC Experiment Results Summary

## GPU Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA RTX 500 Ada Generation Laptop GPU |
| SMs | 16 |
| Shared mem/block | 48 KB (default), 99 KB (optin) |
| Shared mem/SM | 100 KB |
| Registers/SM | 65,536 |
| L2 cache | 16 MB |
| Global memory | 3.7 GB |
| Memory bandwidth | 128 GB/s |
| Compute capability | 8.9 |

## 1. Baseline MLP Training (Steps 1-2)

| Metric | Value |
|--------|-------|
| Architecture | 1024 -> 512 -> 10, no bias, ReLU, CrossEntropyLoss, SGD lr=0.01 |
| Batch size | 64 |
| Runtime | 0.83-1.00 ms/iter |
| Final loss | ~1.6-1.9 |

## 2. Tiling Comparison (Step 3)

Tested on base case: (64x1024) @ (1024x512)

| Tile Size | Total Tasks | Max Parallelism | Avg Parallelism | Occupancy | Shared Mem Violations | Data Movement |
|-----------|-------------|-----------------|-----------------|-----------|----------------------|---------------|
| 32x32 | 3,301 | 1,040 | 186.7x | 87.4% | 0 | 34.5 MB |
| 64x64 | 444 | 128 | 39.1x | 68.2% | 0 | 18.2 MB |
| 128x128 | 128 | 32 | 9.2x | 50.1% | 64 | 14.1 MB |
| 256x256 | 42 | 8 | 3.6x | 74.0% | 20 | 12.0 MB |

Key finding: 64x64 is the best balance. No shared memory violations, good occupancy, manageable task count for CP-SAT.

128x128 tiles need 96 KB shared memory but the GPU block limit is 48 KB, so 64 out of 128 tasks exceed the hardware limit.

## 3. Scaling Experiment (Steps 4-7)

| Scale | Batch | Dims | Tasks | Depth | Max Par. | Avg Par. | Occupancy | Peak Mem | Power |
|-------|-------|------|-------|-------|----------|----------|-----------|----------|-------|
| 1X | 64 | 1024->512->10 | 128 | 12 | 32 | 10.5x | 50.1% | 20.7 MB | 11.97 W |
| 2X | 128 | 2048->1024->10 | 444 | 12 | 128 | 28.9x | 87.9% | 34.0 MB | 13.13 W |
| 3X | 192 | 3072->1536->10 | 1,901 | 13 | 588 | 81.2x | 57.6% | 56.0 MB | 14.06 W |

### Memory Hierarchy at Each Scale

| Scale | Global Read | Global Write | Total Data Mov. | Shared Mem | Registers |
|-------|------------|-------------|-----------------|------------|-----------|
| 1X | 8.0 MB | 5.8 MB | 13.8 MB | 5.3 MB | 7.3 MB |
| 2X | 43.7 MB | 27.1 MB | 70.9 MB | 33.2 MB | 36.6 MB |
| 3X | 160.7 MB | 105.3 MB | 266.0 MB | 119.7 MB | 143.4 MB |

### Energy at Each Scale

| Scale | Baseline Runtime | Avg Power | Energy per 10 iters |
|-------|-----------------|-----------|---------------------|
| 1X | 0.83 ms/iter | 11.97 W | 598.5 mJ |
| 2X | 1.29 ms/iter | 13.13 W | 656.5 mJ |
| 3X | 1.33 ms/iter | 14.06 W | 2,230.3 mJ |

## 4. GPU Memory Hierarchy Detail (Step 7)

Sample task breakdown for a 64x128 @ 128x128 matmul tile:

| Memory Level | Bytes | Fits? |
|-------------|-------|-------|
| Registers | 73,728 B (72 regs/thread x 256 threads) | Yes |
| Shared memory | 98,304 B (96 KB/block) | NO (limit 48 KB) |
| L2 cache | 131,072 B | Yes (16 MB total) |
| Global read | 98,304 B | Yes |
| Global write | 32,768 B | Yes |
| Occupancy | 16.7% | Low due to shared mem pressure |

This is why 128x128 tiles are problematic on this GPU. At 64x64 tile size, shared memory per tile drops to ~32 KB, fitting within the 48 KB limit with room for multiple blocks per SM.

## 5. DAG Structure (Steps 4-6)

Base case with 64x64 tiles:

| Property | Value |
|----------|-------|
| Total tasks | 444 |
| Total edges | 1,722 |
| DAG depth | 12 levels |
| Task types | 280 matmul, 9 accum, 8 relu, 8 grad_relu, 1 loss, 1 grad_loss, 136 weight_update, 1 input |

### Motifs Detected

| Motif | Count |
|-------|-------|
| Fork-join | 9 |
| Fan-out (fanout >= 3) | 18 |
| Fan-in (indegree >= 3) | 153 |
| Chain | 0 |

### DAG Level Widths (parallelism at each depth)

```
Level  0:    1 task    input
Level  1:  128 tasks   L1 forward tiles (all independent)
Level  2:    8 tasks   L1 accumulations
Level  3:    8 tasks   ReLU blocks
Level  4:    8 tasks   L2 forward tiles
Level  5:    1 task    L2 accumulation
Level  6:    1 task    loss (serialization point)
Level  7:    1 task    grad_loss (serialization point)
Level  8:   16 tasks   dW2 + dA1 tiles
Level  9:   16 tasks   dReLU + dW2 weight updates
Level 10:  128 tasks   dW1 tiles (all independent)
Level 11:  128 tasks   weight updates (all independent)
```

## 6. Scheduling Comparison (Steps 8-12, 15)

2 CUDA streams (dual processor), 444 tasks, 64x64 tiles:

| Method | Makespan (us) | Gap vs CP-SAT | GPU Utilization | Energy (mJ) | Runtime Overhead |
|--------|--------------|---------------|-----------------|-------------|------------------|
| PyTorch baseline | 1,003 | N/A | ~100% | N/A | None |
| HEFT | 4,917 | 4.9% | 96.4% | 0.0709 | O(V log V), instant |
| CP-SAT (optimal) | 4,688 | 0.0% | 101.1% | 0.0676 | 30 sec solve time |
| MoSAIC (learned) | 4,912 | 4.8% | 96.5% | 0.0708 | O(V), online |

### Learned Priority Weights (MoSAIC)

```
H(v; theta) = theta^T * phi(v)

theta* = [1.518, -0.773, -0.237, -0.485, 0.082]
          rank_u  depth   fanout  indegree comm_cost
```

Interpretation:
- Upward rank (critical path urgency) is the dominant positive factor
- Negative depth weight means prefer tasks closer to the root (earlier readiness)
- Negative fanout weight means avoid prematurely expanding parallel branches
- This matches the ESWEEK paper finding for BA-like motifs

## 7. Runtime Feedback (Step 14)

Simulated 1.5x GPU thermal throttle:

| Condition | Makespan (us) |
|-----------|--------------|
| Normal | 4,912 |
| Throttled (1.5x) | 7,304 |
| Adapted theta | 7,298 |

The learned theta adapts slightly under changed conditions, though improvement is small (0.1%) because the priority structure remains similar under uniform slowdown.

## 8. Key Takeaways

1. Tile size is a first-class design parameter. 64x64 fits the GPU shared memory, 128x128 does not.
2. Scaling from 1X to 3X increases parallelism from 10x to 81x but also increases data movement from 14 MB to 266 MB.
3. MoSAIC achieves near-optimal scheduling (4.8% gap) at O(V) cost, while CP-SAT takes 30 seconds.
4. The memory hierarchy must be modeled: 50% of tasks at 128x128 tile size violate shared memory limits.
5. Power increases from 12W to 14W as workload scales, with energy growing from 599 mJ to 2,230 mJ.
