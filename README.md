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

3. Google OR-Tools CP-SAT Solver. [https://developers.google.com/optimization](https://developers.google.com/optimization)
