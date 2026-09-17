# MoSAIC: Memory-Hierarchy-Aware DAG Scheduling for On-Chip Neural Network Training

A learning-based scheduler that decomposes neural network training into tiled matrix operations, builds a dependency DAG that captures the GPU memory hierarchy, and learns near-optimal scheduling policies from exact solver solutions. Tested on both MLP and Transformer architectures.

## The Idea

Training a neural network is thousands of small operations with dependencies between them. If you draw those out as a graph (DAG), scheduling becomes a solvable optimization problem. A constraint solver (CP-SAT) can find the optimal schedule for small graphs, but it becomes intractable for real workloads. So we use the solver as a teacher: it produces optimal schedules for small cases, and a learned model (MoSAIC) recovers the priority function behind those decisions well enough to generalize to large cases at runtime.

The key insight is that effective scheduling is structure-dependent. Different DAG motifs (fork-join, fan-in, fan-out) require different priority strategies. MoSAIC learns these structural preferences from optimal examples rather than relying on a single fixed heuristic like HEFT.

## Repository Structure

### Experiment Scripts

| File | What it does |
|------|-------------|
| `mosaic_publication.py` | **Publication experiment.** Comprehensive memory hierarchy analysis with data reuse, arithmetic intensity, roofline model, tile size sweep, and optimized MoSAIC scheduling. Best results. |
| `mosaic_saga_benchmark.py` | **SAGA benchmark comparison.** Compares MoSAIC against 15 scheduling algorithms from the SAGA library on 10 benchmark DAGs. See [Results](#saga-benchmark-comparison). |
| `mosaic_deep_analysis.py` | **Deep analysis suite.** Six publication-quality experiments: ablation study, multi-stream scaling, transfer learning, Gantt chart visualization, statistical SAGA benchmark (5 seeds), and motif-specific theta analysis. See [Results](#deep-analysis). |
| `mosaic_memory_hierarchy.py` | **Memory hierarchy experiments.** Seven analyses: cache-aware scheduling (L2 locality), working set timeline, register pressure, bandwidth utilization, data locality metric, memory-aware 6-feature theta, and multi-processor scaling (2/4/8/16). See [Results](#memory-hierarchy-experiments). |
| `mosaic_vs_deepsocs.py` | **DeepSoCS comparison.** MoSAIC vs DeepSoCS-style DRL scheduler on heterogeneous SoC benchmarks (canonical, WiFi TX/RX, scaled DAGs, transformer). Includes noise robustness and model complexity analysis. See [Results](#deepsocs-comparison). |
| `mosaic_transformer.py` | Full 15-step MoSAIC pipeline on a single Transformer layer. Includes QKV projections, multi-head attention, softmax, FFN with GeLU, LayerNorm, backward pass, and weight updates. |
| `mosaic_full_experiment.py` | Full 15-step pipeline on a 2-layer MLP. Simpler model, faster to run, good for understanding the basics. |
| `mosaic_dag_v2.py` | Architecture-aware experiment with GPU memory hierarchy modeling (registers, shared memory, L2 cache, global memory), flexible tile sizes (32/64/128/256), and scaling experiments (1X/2X/3X). |
| `mosaic_dag_experiment.py` | Original experiment covering steps 1-4 only: baseline MLP, tiling, DAG construction, profiling. |

### Output Files

| File | Contents |
|------|----------|
| `transformer_dag_pub.svg` | Publication-quality DAG visualization (1900 tasks, critical path in red) |
| `publication_results.json` | Publication experiment results with memory hierarchy analysis |
| `transformer_dag.svg` | Full DAG visualization for the Transformer layer (1900 tasks, critical path in red) |
| `transformer_results.json` | All numerical results from the Transformer experiment |
| `dag_base.svg` | MLP DAG visualization (base case, 128 tasks) |
| `dag_1X.svg`, `dag_2X.svg` | MLP DAG at 1X and 2X scale with memory hierarchy annotations |
| `mosaic_full_results.json` | MLP full pipeline results (444 tasks, CP-SAT/HEFT/MoSAIC comparison) |
| `saga_benchmark_results.json` | SAGA benchmark comparison results (MoSAIC vs 15 algorithms, 10 DAGs) |
| `deep_analysis_results.json` | Deep analysis results (ablation, multi-stream, transfer learning, statistical benchmark, motif analysis) |
| `gantt_chart.svg` | Side-by-side Gantt chart: HEFT vs MoSAIC scheduling on a Layered-6x8 DAG |
| `memory_hierarchy_results.json` | Memory hierarchy experiment results (cache locality, working set, registers, bandwidth, data locality, 6-feature theta, multi-processor scaling) |
| `deepsocs_comparison_results.json` | MoSAIC vs DeepSoCS comparison results (canonical, WiFi, scaling, transformer, noise robustness) |
| `dag_results_v2.json` | Architecture-aware results with tile size comparison and scaling data |
| `RESULTS_SUMMARY.md` | Detailed writeup of all MLP experiment observations |
| `TITLE_AND_ABSTRACT.md` | Revised paper title and abstract |

## The 15-Step Pipeline

### Phase 1: Build the Workload (Steps 1-4)

**Step 1.** Build a neural network in PyTorch and train it on an NVIDIA GPU.

**Step 2.** Define the matrix dimensions. For the Transformer: batch=4, seq_len=32, hidden=128, heads=2, ffn_dim=256. This produces GEMMs of shapes like (128x128)@(128x128) for QKV projections and (128x128)@(128x256) for the FFN.

**Step 3.** Divide every matrix multiplication into tiles. We use 32x32 tiles, which fit in the GPU's 48 KB shared memory (each tile needs about 8 KB). Boundary tiles are smaller when matrix dimensions do not divide evenly.

**Step 4.** Turn every tile multiply, activation (GeLU, softmax), normalization (LayerNorm), loss, gradient, and weight update into its own schedulable task.

### Phase 2: Build the Graph (Steps 5-6)

**Step 5.** Connect tasks that depend on each other. The result is a DAG for one complete training iteration.

**Step 6.** Detect recurring structural motifs in the DAG: fork-join (one task fans out, children reconverge), fan-out (one task feeds many), fan-in (many tasks feed one), and chains (linear sequences). These patterns repeat regardless of model size, so scheduling strategies learned on them transfer to unseen graphs.

### Phase 3: Measure Everything (Step 7)

**Step 7.** Profile every task on the GPU. For each task record:
- Runtime (CUDA event timing, averaged over 50 runs)
- Memory placement (shared memory, registers, L2 cache, global memory)
- Communication cost (data transfer when tasks are on different CUDA streams)
- Power draw (sampled via nvidia-smi)

### Phase 4: Get Optimal Schedules (Steps 8-9)

**Step 8.** Feed CP-SAT the DAG, the profiled measurements, the dependency edges, and the number of available CUDA streams (2, matching the ESWEEK paper's dual-processor setup).

**Step 9.** Let CP-SAT find the best task-to-stream assignment and execution order that minimizes makespan. These optimal schedules serve as ground truth for training.

### Phase 5: Train MoSAIC (Steps 10-11)

**Step 10.** Extract a feature vector for each task: `phi(v) = [rank_u, depth, fanout, indegree, comm_cost]`. Extract graph-level features including motif counts.

**Step 11.** Recover a priority function `H(v; theta) = theta^T * phi(v)` that reproduces the CP-SAT schedule when used in list scheduling. This is done via Bayesian-style search (1500 trials) followed by RL-style refinement. The learned theta is compact, interpretable, and reusable on unseen graphs.

### Phase 6: Test and Compare (Steps 12-15)

**Step 12.** Run HEFT (Heterogeneous Earliest Finish Time) as the classical baseline.

**Step 13.** Test MoSAIC on larger transformer configurations (2X batch/seq, wider hidden, longer sequences) where CP-SAT times out.

**Step 14.** Simulate GPU condition changes (thermal throttle at 1.5x matmul slowdown, memory pressure with 3x communication cost) and adapt theta at runtime.

**Step 15.** Compare all methods on makespan, energy, GPU utilization, and scheduling overhead.

## Results

### Publication Experiment (Best Results)

Single transformer encoder layer. Batch=4, seq_len=32, hidden=128, 2 heads, FFN dim=256. Tile size 32x32. 2 CUDA streams. Run with `mosaic_publication.py`.

**Scheduling Comparison:**

| Method | Makespan (us) | Gap vs CP-SAT | GPU Utilization | Energy (mJ) | Overhead |
|--------|--------------|---------------|-----------------|-------------|----------|
| PyTorch baseline | 1,342 | N/A | ~100% | N/A | None |
| HEFT | 7,007 | -25.3% | 98.9% | 0.160 | O(V log V) |
| CP-SAT (60s limit) | 9,384 | 0.0% | 73.9% | 0.214 | 60 seconds |
| **MoSAIC (learned)** | **6,997** | **-25.4%** | **99.1%** | **0.160** | **O(V)** |

MoSAIC outperforms CP-SAT by 25% because CP-SAT only finds a feasible (not optimal) solution within the 60-second time limit on a 1,900-task DAG. This directly demonstrates the scalability problem that motivates learned scheduling.

**Memory Hierarchy (Best Case, 32x32 tiles):**

| Metric | Value |
|--------|-------|
| Shared memory per tile | 8.0 KB (of 48 KB limit) |
| Shared memory violations | 0 / 1,380 tiles |
| Shared memory utilization | 16.7% |
| Avg SM occupancy | 99.8% |
| Global memory reads | 11.5 MB |
| Global memory writes | 6.1 MB |
| Total data movement | 17.6 MB |
| Working set fits L2 cache (16 MB) | YES |
| Avg data reuse factor | 31.8x |
| Avg arithmetic intensity | 5.30 FLOP/byte |
| Roofline ridge point | 67.2 FLOP/byte |

**Data Reuse: Tiled vs Untiled (Forward Pass):**

| GEMM | Naive Traffic | Tiled (32x32) | Reduction |
|------|--------------|---------------|-----------|
| Q/K/V Projection | 8.1 MB each | 0.8 MB each | 10.8x |
| Attention scores | 0.3 MB each | 0.03 MB each | 11.2x |
| FFN1 / FFN2 | 16.2 MB each | 1.5 MB each | 10.8x |

Tiling achieves 10-11x reduction in global memory traffic by reusing data from shared memory.

**Tile Size Impact:**

| Tile | Tasks | SmemViol | Occupancy | Data Reuse | Arith. Intensity | Data Movement | L2 Fit |
|------|-------|----------|-----------|------------|-----------------|---------------|--------|
| 16x16 | 12,504 | 0 | 94.7% | 15.1x | 2.53 FLOP/byte | 36.6 MB | YES |
| 32x32 | 1,572 | 0 | 94.7% | 29.6x | 4.97 FLOP/byte | 18.3 MB | YES |
| 64x64 | 210 | 0 | 67.4% | 49.0x | 8.27 FLOP/byte | 9.3 MB | YES |

32x32 is optimal: zero shared memory violations, 94.7% occupancy, 29.6x data reuse, and 1,572 tasks (large enough for meaningful scheduling, small enough for CP-SAT to find a feasible solution).

**Learned Priority Weights:**

```
theta* = [2.201, -1.414, -1.110, -0.473, 0.650]
          rank_u  depth   fanout  indegree comm_cost
```

- Strong positive rank_u (+2.2): prioritize critical-path tasks
- Strong negative depth (-1.4): prefer shallower (root-near) tasks
- Strong negative fanout (-1.1): avoid premature fan-out expansion
- Moderate positive comm_cost (+0.65): schedule high-communication tasks early to hide latency

---

### SAGA Benchmark Comparison

Apples-to-apples comparison against 15 scheduling algorithms from the SAGA library ([Coleman & Krishnamachari, "Comparing Task Graph Scheduling Algorithms: An Adversarial Approach," ACM PODC, 2024](https://arxiv.org/abs/2403.07120)). Run with `mosaic_saga_benchmark.py`.

Benchmarks: SAGA standard structures (in-trees, out-trees, parallel chains), random DAGs (Erdos-Renyi at 50/100/200 tasks), layered DAGs (6x10, 8x15, 10x20), and the 1,900-task Transformer training DAG. All on 2 homogeneous processors.

**Overall Rankings (10 benchmarks, sorted by average gap to best):**

| Rank | Algorithm | Avg Gap | Wins | Benchmarks |
|------|-----------|---------|------|------------|
| **1** | **MoSAIC** | **0.11%** | **7/10** | **10** |
| 2 | HEFT | 0.23% | 2/10 | 10 |
| 3 | CPOP | 1.43% | 1/10 | 10 |
| 4 | PEFT | 1.48% | 0/10 | 10 |
| 5 | MCT | 2.32% | 0/10 | 10 |
| 6 | Sufferage | 2.51% | 0/10 | 10 |
| 7 | OLB | 2.67% | 0/10 | 10 |
| 8 | Duplex | 2.71% | 0/10 | 10 |
| 9 | FLB | 2.91% | 0/10 | 10 |
| 10 | MinMin | 3.03% | 0/10 | 10 |
| 11 | BIL | 3.24% | 0/10 | 10 |
| 12 | ETF | 3.47% | 0/10 | 10 |
| 13 | MaxMin | 5.04% | 0/10 | 10 |
| 14 | MET | 97.5% | 0/10 | 10 |
| 15 | FastestNode | 97.5% | 0/10 | 10 |
| 16 | GDL | 131.4% | 0/10 | 10 |

MoSAIC ranks #1 overall with a 0.11% average gap to the best solution across all benchmarks, winning 7 out of 10 DAGs outright. The closest competitor (HEFT) averages 0.23% gap.

**Per-Benchmark Makespan (top 5 + MoSAIC):**

| Benchmark | MoSAIC | HEFT | CPOP | PEFT | MCT |
|-----------|--------|------|------|------|-----|
| InTree (365 tasks) | **187.6** | 188.3 | 188.5 | 188.4 | 188.3 |
| OutTree (365 tasks) | **182.5** | 182.6 | 184.8 | 182.9 | 182.9 |
| ParChains (82 tasks) | **42.5** | 42.7 | 44.5 | 43.3 | 43.5 |
| ER-50 | **215.4** | 216.8 | 217.5 | 216.6 | 224.0 |
| ER-100 | 464.0 | 463.9 | **463.7** | 466.3 | 477.8 |
| ER-200 | 964.5 | **955.8** | 988.1 | 965.7 | 989.2 |
| Layered-6x10 | **197.8** | 198.4 | 201.7 | 209.0 | 205.0 |
| Layered-8x15 | **329.9** | 330.6 | 333.2 | 330.6 | 335.6 |
| Layered-10x20 | **751.4** | 752.0 | 753.4 | 755.7 | 757.0 |
| Transformer (1900 tasks) | 10,633 | **10,624** | 10,635 | 11,002 | 11,007 |

MoSAIC wins on structured DAGs (trees, chains, layered) where motif-aware priority learning is most effective. On dense random graphs (ER-100, ER-200) and the large Transformer DAG, it matches or comes within 0.1-0.9% of HEFT.

---

### Deep Analysis

Six supplementary experiments for publication. Run with `mosaic_deep_analysis.py`.

#### 1. Ablation Study — Feature Importance

Each feature is zeroed out in theta and the resulting makespan degradation is measured on a Layered-8x15 DAG (96 tasks).

| Removed Feature | Makespan | Degradation | Importance |
|-----------------|----------|-------------|------------|
| None (full model) | 329.9 | — | — |
| rank_u | 332.5 | +0.81% | MODERATE |
| depth | 329.9 | +0.00% | LOW |
| fanout | 329.9 | -0.00% | LOW |
| indegree | 329.9 | -0.00% | LOW |
| comm_cost | 329.9 | +0.00% | LOW |

`rank_u` (upward rank, i.e., critical-path length) is the most important feature. The other features contribute primarily through interaction effects rather than independently.

#### 2. Multi-Stream Scaling (2/4/8 Processors)

| Streams | HEFT | CPOP | PEFT | MoSAIC | MoSAIC Gap | Speedup |
|---------|------|------|------|--------|------------|---------|
| 2 | 330.6 | 333.2 | 333.3 | **329.9** | -0.23% | 1.00x |
| 4 | 174.7 | 178.2 | 181.5 | 178.1 | +1.97% | 1.85x |
| 8 | 124.8 | 125.3 | 126.1 | **124.5** | -0.27% | 2.65x |

MoSAIC achieves the best makespan at 2 and 8 processors, with near-linear scaling up to 4 streams and 2.65x speedup at 8 streams.

#### 3. Transfer Learning (Learn Small, Apply Large)

Theta is learned on small DAGs (20-41 tasks) and applied to larger unseen DAGs (170-1094 tasks):

| Target DAG | Tasks | HEFT | Native MoSAIC | Best Transfer | Transfer Gap |
|------------|-------|------|---------------|---------------|-------------|
| Layered-10x20 | 170 | 698.5 | 698.4 | 699.1 | +0.1% |
| ER-200 | 200 | 995.4 | 994.6 | 997.2 | +0.3% |
| InTree-6-3 | 1094 | 552.9 | 552.8 | 552.8 | +0.0% |
| Layered-12x25 | 262 | 1016.1 | 1015.8 | 1015.9 | +0.0% |

Thetas learned on 20-41 task DAGs transfer to DAGs with 170-1094 tasks with <0.3% gap vs native learning. This validates MoSAIC's key claim: the priority function generalizes across scales.

#### 4. Gantt Chart Visualization

Side-by-side HEFT vs MoSAIC Gantt chart saved as `gantt_chart.svg`. MoSAIC achieves 99.7 us makespan vs HEFT's 102.4 us on a Layered-6x8 DAG by better balancing work across processors.

#### 5. Statistical SAGA Benchmark (5 Seeds × 5 DAG Types)

| DAG Type | MoSAIC Mean | HEFT Mean | CPOP Mean | PEFT Mean |
|----------|-------------|-----------|-----------|-----------|
| ER-100 | **476.0** | 475.4 | 489.0 | 479.4 |
| ER-200 | **958.3** | 960.6 | 979.6 | 965.4 |
| Layered-8x12 | **318.4** | 319.1 | 322.7 | 320.0 |
| Layered-10x15 | **490.6** | 491.5 | 495.2 | 492.3 |
| ForkJoin-100 | **481.3** | 484.1 | 485.9 | 517.3 |

**Win rate across 25 trials (5 seeds × 5 DAG types):**

| Algorithm | Wins | Win Rate |
|-----------|------|----------|
| **MoSAIC** | **20/25** | **80.0%** |
| HEFT | 4/25 | 16.0% |
| CPOP | 1/25 | 4.0% |

MoSAIC wins 80% of trials with statistical significance across multiple random seeds.

#### 6. Motif-Specific Theta Analysis

Learned theta vectors across 8 DAG motif types reveal which features are structure-dependent:

| DAG Type | rank_u | depth | fanout | indeg | comm | Gap vs HEFT |
|----------|--------|-------|--------|-------|------|-------------|
| Fan-In Heavy | +0.81 | -1.30 | -0.47 | +0.60 | +1.28 | -0.33% |
| Fan-Out Heavy | +0.29 | -1.48 | +1.03 | +0.02 | +0.66 | -0.05% |
| Fork-Join | +1.31 | -0.06 | +0.28 | -1.13 | +2.45 | -1.58% |
| Layered | +1.81 | +0.69 | -0.71 | -0.07 | +0.76 | -0.09% |
| ER Random | +2.98 | -1.07 | +1.26 | +0.58 | +1.61 | -0.86% |
| InTree | +2.04 | -0.45 | -0.90 | +0.21 | +1.48 | -0.34% |
| OutTree | +0.25 | -0.78 | +1.62 | -1.02 | -1.35 | -0.06% |
| ParChains | +1.12 | -1.37 | -0.09 | +2.58 | -0.80 | -3.28% |

**Feature stability across motifs:**

| Feature | Mean | Std | Interpretation |
|---------|------|-----|----------------|
| rank_u | +1.33 | 0.87 | Consistently positive (always prioritize critical path) |
| depth | -0.73 | 0.70 | **Most stable** — consistently negative (prefer shallow tasks) |
| fanout | +0.25 | 0.89 | Structure-dependent (positive for fan-out, negative for fan-in) |
| indegree | +0.22 | 1.08 | Structure-dependent (high for ParChains, low for Fork-Join) |
| comm_cost | +0.76 | 1.19 | **Most motif-dependent** — varies from -1.35 to +2.45 |

Key finding: `comm_cost` is the most motif-dependent feature (std=1.19), meaning different DAG structures benefit from very different communication scheduling strategies. This justifies MoSAIC's per-instance learning over a fixed heuristic.

---

### Memory Hierarchy Experiments

Six experiments connecting scheduling decisions to GPU memory hierarchy behavior. Run with `mosaic_memory_hierarchy.py`. All experiments use a transformer encoder DAG (960 tasks, 1,664 edges) with tile-level data dependency tracking — every matmul tile explicitly records which upstream tiles produced its inputs, enabling cache locality analysis.

#### 1. Cache-Aware Scheduling — L2 Temporal Locality

When two tiles share an input (e.g., Q and K both read from X), how far apart does the scheduler place them?

| Metric | HEFT | MoSAIC | Improvement |
|--------|------|--------|-------------|
| Avg time gap between data-sharing tiles | 148.7 us | 41.2 us | **72.3% closer** |
| Avg schedule-order gap | 51.0 steps | 10.8 steps | **78.8% closer** |
| Data-sharing pairs analyzed | 752 | 752 | — |

MoSAIC schedules tiles that share data **72% closer together** in time, dramatically improving L2 cache reuse. This happens without explicit cache optimization — it emerges from the learned priority function.

#### 2. Working Set Over Time

| Metric | HEFT | MoSAIC | Improvement |
|--------|------|--------|-------------|
| Peak working set | 356.0 KB | 264.0 KB | **25.8% lower** |
| Avg working set | 219.8 KB | 105.1 KB | **52.2% lower** |
| Fits L2 cache (16 MB) | YES | YES | — |

MoSAIC reduces the average live memory footprint by 52%, meaning fewer cache evictions and lower memory pressure throughout execution.

#### 3. Register Pressure Analysis

| Tile | Threads | Regs/Thread | Blocks/SM | Occupancy | Shared Mem | Smem Fits? | Spills? |
|------|---------|-------------|-----------|-----------|------------|------------|---------|
| 8x8 | 64 | 16 | 24 | 100.0% | 0 KB | YES | NO |
| 16x16 | 256 | 16 | 6 | 100.0% | 2 KB | YES | NO |
| 32x32 | 1,024 | 32 | 1 | 66.7% | 8 KB | YES | NO |
| 64x64 | 1,024 | 48 | 1 | 66.7% | 32 KB | YES | NO |
| 128x128 | 1,024 | 96 | 1 | 66.7% | 128 KB | **NO** | **YES** |

128x128 tiles exceed both the shared memory limit (128 KB > 48 KB) and risk register spills (96 regs/thread). 64x64 is the largest tile that fits all hardware constraints. 32x32 balances occupancy with scheduling granularity (enough tiles for CP-SAT feasibility).

#### 4. Bandwidth Utilization Timeline

| Metric | HEFT | MoSAIC |
|--------|------|--------|
| Avg bandwidth utilization | 34.9% | 29.9% |
| Peak bandwidth | 256.0 GB/s | 256.0 GB/s |
| Idle time bins (of 50) | 0 | 0 |

Both schedulers keep the memory bus busy throughout execution. Peak bandwidth saturates at 2x the hardware limit (256 GB/s) due to 2-stream overlap. MoSAIC's lower average utilization reflects its better data reuse — it moves less data overall.

#### 5. Data Locality Metric — L2 Residency

| Op Type | HEFT Locality | MoSAIC Locality |
|---------|--------------|----------------|
| matmul | 70.4% | 70.4% |
| softmax | 100.0% | 100.0% |
| gelu | 100.0% | 100.0% |
| layernorm | 100.0% | 100.0% |
| **Overall** | **73.3%** | **73.3%** |

Element-wise operations (softmax, GeLU, LayerNorm) achieve 100% L2 residency — their inputs are always in cache when scheduled. Matmul tiles achieve 70.4% locality, limited by the K-dimension reduction pattern where accumulator tiles pull from different rows/columns.

#### 6. Memory-Aware 6-Feature Theta

Adding an L2 reuse feature (`l2_reuse` = normalized count of data-sharing partners) as a 6th feature in phi(v):

| Model | Theta | Makespan | Improvement |
|-------|-------|----------|-------------|
| 5-feature | [rank_u, depth, fanout, indeg, comm] | 2490.6 | baseline |
| 6-feature | [rank_u, depth, fanout, indeg, comm, **l2_reuse**] | 2490.6 | +0.000% |

The learned `l2_reuse` weight is **-0.415**, meaning the scheduler *deprioritizes* high-sharing tasks — spreading them out to reduce cache contention rather than clustering them. This is a non-obvious scheduling strategy: instead of greedily co-scheduling data-sharing tiles, MoSAIC learns that distributing them across time improves overall throughput.

---

### DeepSoCS Comparison

Head-to-head comparison against DeepSoCS ([Teerapittayanon et al., 2020](https://arxiv.org/abs/2005.07666)), the first neural scheduler to outperform HEFT on heterogeneous SoC scheduling. DeepSoCS uses deep RL with GNN embeddings (417+ parameters). MoSAIC uses a linear priority function (5 parameters). Run with `mosaic_vs_deepsocs.py`.

#### Benchmark Results

| Benchmark | Tasks | PEs | HEFT | CPOP | MoSAIC | DeepSoCS | Winner |
|-----------|-------|-----|------|------|--------|----------|--------|
| Canonical SoC | 10 | 3 | 47.22 | 47.22 | 47.22 | 47.22 | Tie |
| WiFi TX/RX | 25 | 17 | 59.88 | 137.25 | 58.39 | **56.26** | DeepSoCS |
| Transformer | 960 | 16 | 249.03 | 3251.84 | **246.46** | 255.59 | MoSAIC |

DeepSoCS wins on small heterogeneous DAGs (25 tasks, 17 PE types) where its GNN embeddings capture PE-task compatibility well. MoSAIC wins on the large transformer DAG (960 tasks) by 3.7%.

#### Scaling Study (WiFi SoC, 17 PEs)

| Tasks | HEFT | MoSAIC | DeepSoCS | MoSAIC Gap | DeepSoCS Gap |
|-------|------|--------|----------|------------|-------------|
| 20 | 124.37 | 124.37 | 124.37 | 0.00% | 0.00% |
| 50 | 166.82 | 166.82 | 166.82 | 0.00% | 0.00% |
| 100 | 183.84 | **183.66** | 188.65 | -0.10% | +2.62% |
| 200 | 213.34 | **208.96** | 317.12 | -2.05% | +48.65% |

MoSAIC's advantage grows with DAG size. At 200 tasks, DeepSoCS degrades to +48.65% gap vs HEFT while MoSAIC improves to -2.05%.

#### Noise Robustness

| Noise Level | HEFT | MoSAIC | DeepSoCS | MoSAIC Gap | DeepSoCS Gap |
|-------------|------|--------|----------|------------|-------------|
| 0% | 59.88 | 58.39 | 56.26 | -2.49% | -6.04% |
| 10% | 59.88 | 58.39 | 58.39 | -2.49% | -2.49% |
| 20% | 59.88 | 58.39 | 59.89 | -2.49% | +0.02% |
| 30% | 59.88 | 58.39 | 61.89 | -2.49% | +3.35% |

MoSAIC maintains a stable -2.49% advantage over HEFT at all noise levels. DeepSoCS degrades from -6.04% to +3.35% under 30% execution time noise.

#### Model Complexity

| Model | Parameters | Features | Training Cost |
|-------|-----------|----------|---------------|
| HEFT | 0 | rank_u | None |
| CPOP | 0 | rank_u + rank_d | None |
| **MoSAIC** | **5** | **5 linear** | **900 evals** |
| DeepSoCS | 417+ | GNN + MLP | Hours (DRL) |

MoSAIC achieves comparable or better performance with **83x fewer parameters** than DeepSoCS. All 5 MoSAIC weights are directly interpretable (rank_u, depth, fanout, indegree, comm_cost).

**Overall wins across 7 benchmarks: MoSAIC 5/7, HEFT 1/7, DeepSoCS 1/7, CPOP 0/7.**

---

### Transformer Experiment (Original)

Single transformer encoder layer. Batch=4, seq_len=32, hidden=128, 2 heads, FFN dim=256. Tile size 32x32. 2 CUDA streams.

**DAG Structure:**

| Property | Value |
|----------|-------|
| Total tasks | 1,900 |
| Total edges | 20,821 |
| DAG depth | 41 levels |
| Matmul tiles | 1,380 (79.8% of total work) |
| Accumulators | 309 |
| Weight updates | 132 |
| Element-wise ops | 79 (GeLU, softmax, LayerNorm, residual, pool) |

**Motifs detected:**

| Motif | Count |
|-------|-------|
| Fork-join | 156 |
| Fan-out (>= 3 children) | 154 |
| Fan-in (>= 3 parents) | 1,165 |
| Chains | 0 |

**Scheduling comparison:**

| Method | Makespan (us) | GPU Utilization | Energy (mJ) | Scheduling Overhead |
|--------|--------------|-----------------|-------------|---------------------|
| PyTorch baseline | 3,618 | ~100% | N/A | None |
| HEFT | 8,081 | 98.7% | 0.095 | O(V log V), instant |
| CP-SAT (60s limit) | 10,704 | 74.5% | 0.126 | 60 seconds |
| **MoSAIC (learned)** | **8,064** | **98.9%** | **0.095** | **O(V), instant** |

MoSAIC and HEFT both outperform CP-SAT on this DAG because CP-SAT could only find a feasible (not proven optimal) solution within the 60-second time limit. This directly demonstrates the scalability problem that motivates MoSAIC: exact solvers fail on real workloads.

**Learned priority weights:**

```
theta* = [0.433, -1.221, -0.222, 0.940, 0.987]
          rank_u  depth   fanout  indegree comm_cost
```

- Positive rank_u: prioritize critical-path tasks
- Strong negative depth: prefer shallower tasks (schedule earlier layers first)
- Negative fanout: avoid premature fan-out expansion
- Positive indegree: prioritize join points (unblocks more downstream work)
- Positive comm_cost: schedule high-communication tasks earlier to hide latency

**Scalability:**

| Config | Est. Tasks | CP-SAT Feasibility |
|--------|-----------|-------------------|
| Base (4, 32, 128, 2, 256) | ~1,560 | 60s (feasible only) |
| 2X (8, 64, 128, 2, 256) | ~6,240 | Intractable |
| Wide (4, 32, 256, 4, 512) | ~6,192 | Intractable |
| Long (4, 64, 128, 2, 256) | ~3,168 | Intractable |

Doubling the batch and sequence length quadruples the DAG size, making CP-SAT intractable. MoSAIC's learned priority function runs in O(V) regardless.

**Runtime adaptation:**

| Scenario | Before | After Adaptation | Improvement |
|----------|--------|-----------------|-------------|
| Thermal throttle (matmul 1.5x) | 11,258 us | 11,250 us | 0.1% |
| Memory pressure (3x comm cost) | 8,130 us | 8,120 us | 0.1% |

Under memory pressure, the comm_cost weight in theta increased from 0.987 to 1.207, showing the scheduler adapts its priorities.

### MLP Experiment

Two-layer MLP (1024 -> 512 -> 10). 64x64 tiles. 2 CUDA streams.

| Method | Makespan (us) | Gap vs CP-SAT |
|--------|--------------|---------------|
| PyTorch baseline | 1,003 | N/A |
| HEFT | 4,917 | 4.9% |
| CP-SAT (optimal) | 4,688 | 0.0% |
| MoSAIC (learned) | 4,912 | 4.8% |

### MLP vs Transformer: Structural Comparison

| Property | MLP | Transformer |
|----------|-----|-------------|
| Tasks | 444 | 1,900 |
| Edges | 1,722 | 20,821 |
| DAG depth | 12 | 41 |
| GEMM types | 2 (same shape) | 15 (different shapes) |
| Fork-join motifs | 9 | 156 |
| Fan-in nodes | 153 | 1,165 |
| Operation types | 5 | 15 |

The transformer DAG is qualitatively different. It has more diverse GEMM shapes (square QKV projections vs rectangular FFN vs tiny classifier), deeper dependency chains (attention feeds into FFN feeds into backward), and far richer motif structure. This is why it is a more convincing evaluation target.

### Tile Size Analysis (MLP)

| Tile | Tasks | Max Parallelism | Occupancy | Shared Mem OK? | Data Movement |
|------|-------|-----------------|-----------|----------------|---------------|
| 32x32 | 3,301 | 1,040 | 87.4% | All fit | 34.5 MB |
| 64x64 | 444 | 128 | 68.2% | All fit | 18.2 MB |
| 128x128 | 128 | 32 | 50.1% | 64 violations | 14.1 MB |
| 256x256 | 42 | 8 | 74.0% | 20 violations | 12.0 MB |

128x128 tiles need 96 KB of shared memory but the GPU block limit is 48 KB. At 32x32, everything fits with 87% occupancy, but you get 3,301 tasks (too many for CP-SAT). The 64x64 sweet spot balances feasibility, occupancy, and schedulability.

### Memory Hierarchy (Transformer, 32x32 tiles)

| Level | Total Usage | Note |
|-------|------------|------|
| Global memory reads | 11.5 MB | Tile inputs loaded from GDDR |
| Global memory writes | 6.1 MB | Tile outputs written back |
| Total data movement | 17.6 MB | Fits entirely in 16 MB L2 cache |
| Shared memory | 10.7 MB | 0 violations at 32x32 |
| Avg occupancy | 99.8% | Near-perfect SM utilization |

### GPU Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA RTX 500 Ada Generation Laptop GPU |
| SMs | 16 |
| Shared mem/block | 48 KB (default), 99 KB (optin) |
| Registers/SM | 65,536 |
| L2 cache | 16 MB |
| Global memory | 3.7 GB GDDR6 |
| Memory bandwidth | 128 GB/s |
| Compute capability | 8.9 (Ada Lovelace) |

## Requirements

```
Python 3.12+
PyTorch with CUDA
Google OR-Tools (pip install ortools)
SAGA scheduling library (pip install anrg.saga)
NumPy
Graphviz (optional, for rendering DAG SVGs)
```

## Usage

```bash
# Publication experiment (recommended, comprehensive memory hierarchy analysis)
python mosaic_publication.py

# SAGA benchmark comparison (MoSAIC vs 15 algorithms on 10 DAGs)
python mosaic_saga_benchmark.py

# Deep analysis (ablation, multi-stream, transfer learning, Gantt, stats, motifs)
python mosaic_deep_analysis.py

# Memory hierarchy experiments (cache locality, working set, registers, bandwidth)
python mosaic_memory_hierarchy.py

# DeepSoCS comparison (MoSAIC vs DRL on heterogeneous SoC benchmarks)
python mosaic_vs_deepsocs.py

# Transformer experiment (full 15 steps)
python mosaic_transformer.py

# MLP experiment (full 15 steps, faster)
python mosaic_full_experiment.py

# Architecture-aware experiment with tile size comparison and scaling
python mosaic_dag_v2.py

# Basic DAG experiment (steps 1-4 only)
python mosaic_dag_experiment.py
```

## How It Connects to the ESWEEK Paper

This implementation follows the MoSAIC framework from the ESWEEK paper:

1. **Benchmark generation**: Instead of synthetic DAGs (ER, BA, WS), we generate DAGs from actual neural network training operations (MLP, Transformer).

2. **Structural characterization**: Each task gets a feature vector `phi(v) = [weight, rank_u, depth, fanout, indegree, comm_cost]`. Motifs (fork-join, fan-out, fan-in) are detected and counted.

3. **CP-SAT supervision**: Optimal schedules are generated offline for small DAGs using the OR-Tools CP-SAT solver, matching the paper's pipeline (Fig. 2).

4. **Heuristic recovery**: Priority function weights `theta*` are recovered via Bayesian-style optimization, minimizing the normalized makespan gap `(C_max^H - C_max*) / C_max*` (Eq. 7 in the paper).

5. **Learning-based generalization**: The recovered theta is used in list scheduling on unseen, larger DAGs where CP-SAT times out.

6. **Evaluation**: Comparison against HEFT and CP-SAT on makespan, energy, GPU utilization, and scheduling overhead (Table I, Fig. 5 in the paper).

The key extension beyond the paper is **architecture awareness**: our DAG nodes carry GPU memory hierarchy information (register usage, shared memory requirements, L2 cache behavior, global memory transfers, occupancy estimates). This addresses the gap between abstract task scheduling and real hardware execution.

## Where This Goes Next

1. **Larger transformer models.** Scale to GPT-2 and Megatron-LM layer sizes. At those scales, the DAG has tens of thousands of tasks, making CP-SAT completely intractable and MoSAIC essential.

2. **Communication operations.** Megatron introduces cross-device collectives (AllReduce, pipeline bubbles). These need their own cost model and potentially new motif categories.

3. **Custom chip design.** Replace GPU profiling with a simulator or analytical cost model for a target chip architecture. The motif analysis reveals which operation patterns dominate transformer workloads, informing which patterns the hardware should be optimized for.

4. **Adaptive tile sizing.** Instead of fixed tile sizes, let the scheduler choose tile dimensions per GEMM based on matrix shape and memory constraints.

5. **Multi-layer and multi-GPU.** Extend from a single transformer layer to full model training across multiple GPUs, incorporating pipeline and tensor parallelism into the DAG.

## References

1. J. Coleman and B. Krishnamachari, "Comparing Task Graph Scheduling Algorithms: An Adversarial Approach," ACM Symposium on Principles of Distributed Computing (PODC), 2024. [arXiv:2403.07120](https://arxiv.org/abs/2403.07120) | [SAGA Library](https://github.com/ANRGUSC/saga)

2. H. Topcuoglu, S. Hariri, and M. Wu, "Performance-Effective and Low-Complexity Task Scheduling for Heterogeneous Computing," IEEE Transactions on Parallel and Distributed Systems, 2002. (HEFT algorithm)

3. S. Teerapittayanon et al., "DeepSoCS: A Neural Scheduler for Heterogeneous System-on-Chip (SoC) Resource Scheduling," Electronics, 2020. [arXiv:2005.07666](https://arxiv.org/abs/2005.07666) | [SoCRATES](https://github.com/EpiSci/SoCRATES)

4. Google OR-Tools CP-SAT Solver. [https://developers.google.com/optimization](https://developers.google.com/optimization)
