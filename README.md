# MoSAIC: Memory-Hierarchy-Aware DAG Scheduling for On-Chip Neural Network Training

A learning-based scheduler that decomposes neural network training into tiled matrix operations, builds a dependency DAG that captures the GPU memory hierarchy, and learns near-optimal scheduling policies from exact solver solutions. Tested on both MLP and Transformer architectures.

## The Idea

Training a neural network is thousands of small operations with dependencies between them. If you draw those out as a graph (DAG), scheduling becomes a solvable optimization problem. A constraint solver (CP-SAT) can find the optimal schedule for small graphs, but it becomes intractable for real workloads. So we use the solver as a teacher: it produces optimal schedules for small cases, and a learned model (MoSAIC) recovers the priority function behind those decisions well enough to generalize to large cases at runtime.

The key insight is that effective scheduling is structure-dependent. Different DAG motifs (fork-join, fan-in, fan-out) require different priority strategies. MoSAIC learns these structural preferences from optimal examples rather than relying on a single fixed heuristic like HEFT.

## Repository Structure

### Experiment Scripts

| File | What it does |
|------|-------------|
| `mosaic_transformer.py` | **Main experiment.** Full 15-step MoSAIC pipeline on a single Transformer layer. Includes QKV projections, multi-head attention, softmax, FFN with GeLU, LayerNorm, backward pass, and weight updates. |
| `mosaic_full_experiment.py` | Full 15-step pipeline on a 2-layer MLP. Simpler model, faster to run, good for understanding the basics. |
| `mosaic_dag_v2.py` | Architecture-aware experiment with GPU memory hierarchy modeling (registers, shared memory, L2 cache, global memory), flexible tile sizes (32/64/128/256), and scaling experiments (1X/2X/3X). |
| `mosaic_dag_experiment.py` | Original experiment covering steps 1-4 only: baseline MLP, tiling, DAG construction, profiling. |

### Output Files

| File | Contents |
|------|----------|
| `transformer_dag.svg` | Full DAG visualization for the Transformer layer (1900 tasks, critical path in red) |
| `transformer_results.json` | All numerical results from the Transformer experiment |
| `dag_base.svg` | MLP DAG visualization (base case, 128 tasks) |
| `dag_1X.svg`, `dag_2X.svg` | MLP DAG at 1X and 2X scale with memory hierarchy annotations |
| `mosaic_full_results.json` | MLP full pipeline results (444 tasks, CP-SAT/HEFT/MoSAIC comparison) |
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

### Transformer Experiment

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
Python 3.8+
PyTorch with CUDA
Google OR-Tools (pip install ortools)
NumPy
Graphviz (optional, for rendering DAG SVGs)
```

## Usage

```bash
# Transformer experiment (recommended, full 15 steps)
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
