# MoSAIC: Learning to Schedule Kernel Operations for Custom Chip Design

A learning-based scheduler that decomposes neural network training into tiled matrix operations, builds a DAG of dependencies, and learns near-optimal scheduling policies from exact solver solutions.

## The Idea

Training a neural network is really thousands of small operations with dependencies between them. If you draw those out as a graph, scheduling becomes a solvable optimization problem. A solver (CP-SAT) can find the optimal schedule for small graphs, but it gets too slow for real workloads. So we use the solver as a teacher: it produces optimal schedules for small cases, and a learned model (MoSAIC) copies its reasoning well enough to work on large cases at runtime.

## What This Repo Contains

### Scripts

| File | Description |
|------|-------------|
| `mosaic_dag_experiment.py` | Steps 1-4: baseline MLP, tiling, DAG construction, profiling |
| `mosaic_dag_v2.py` | Architecture-aware version with GPU memory hierarchy, flexible tile sizes, scaling experiments, power monitoring |
| `mosaic_full_experiment.py` | All 15 steps end to end: DAG construction, motif detection, CP-SAT scheduling, HEFT baseline, MoSAIC learned scheduling, comparison |

### Output Files

| File | Description |
|------|-------------|
| `dag_base.svg` | DAG visualization for base case (64x64 tiles) |
| `dag_1X.svg`, `dag_2X.svg` | DAG visualizations at 1X and 2X scale |
| `*.dot` | Graphviz source files for all DAGs |
| `dag_results.json` | Results from v1 experiment |
| `dag_results_v2.json` | Results from architecture-aware experiment |
| `mosaic_full_results.json` | Results from the full 15-step experiment |

## The 15 Steps

1. Build a small MLP in PyTorch and train it on an NVIDIA GPU
2. Use matrices of 64x1024, 1024x512, and 512x10 for the two MLP layers
3. Divide the matrix multiplications into tiles (64x64 or 128x128). Boundary tiles can be smaller
4. Treat every tile multiplication, activation, loss calculation, gradient calculation, and weight update as a separate task
5. Connect dependent tasks to create a DAG for one complete training step
6. Find repeated motifs in the DAG: fan-out, fan-in, chains, and fork-join structures
7. Profile every DAG task on the NVIDIA GPU to measure runtime, energy, power, memory, and data transfer cost
8. Give the small DAG, task measurements, dependencies, and available CUDA streams to CP-SAT
9. Let CP-SAT find the best task order and stream assignment for minimizing training time and energy
10. Use the CP-SAT schedules as expert examples to teach MoSAIC how different DAG features and motifs should be scheduled
11. MoSAIC learns which ready tasks to prioritize, especially critical tasks inside fork-join and fan-in motifs
12. Run the original PyTorch workload as the baseline and run HEFT as the traditional scheduling baseline
13. Run the MoSAIC scheduler on larger and unseen matrix sizes where CP-SAT would be too slow for runtime use
14. Use runtime feedback to let MoSAIC adjust its scheduling priorities or reward function when GPU conditions change
15. Compare PyTorch, HEFT, CP-SAT, and MoSAIC using training time, energy, GPU utilization, scheduling overhead, loss, and accuracy

## Results

Tested on NVIDIA RTX 500 Ada with 64x64 tiles, 2 CUDA streams, 444 tasks in the DAG.

| Method | Makespan (us) | Gap vs CP-SAT | GPU Utilization |
|--------|--------------|---------------|-----------------|
| PyTorch baseline | 1003 | N/A | ~100% |
| HEFT | 4917 | 4.9% | 96.4% |
| CP-SAT (optimal) | 4688 | 0.0% | 101.1% |
| MoSAIC (learned) | 4912 | 4.8% | 96.5% |

MoSAIC learns a priority function `H(v; theta) = theta^T * phi(v)` where `phi(v) = [rank_u, depth, fanout, indegree, comm_cost]`. The recovered weights show that upward rank (critical path urgency) dominates scheduling decisions, consistent with the findings in the ESWEEK paper.

## Requirements

- Python 3.8+
- PyTorch with CUDA
- Google OR-Tools (`pip install ortools`)
- NumPy
- Graphviz (optional, for rendering DAG SVGs)

## Usage

```bash
# Run the full 15-step experiment
python mosaic_full_experiment.py

# Run just steps 1-4 with architecture-aware memory hierarchy
python mosaic_dag_v2.py

# Run the basic DAG experiment
python mosaic_dag_experiment.py
```

## Where This Goes Next

The MLP is a testbed. The real target is LLM Megatron-style operations to inform the design of a custom chip. Motif discovery is the bridge: if you can identify which operation patterns dominate a transformer workload, you know which patterns the hardware should be built around.
