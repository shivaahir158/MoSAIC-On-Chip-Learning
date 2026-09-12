"""
MoSAIC On-Chip Learning: DAG Experiment (Steps 1-4)
=====================================================
Step 1: Build a small MLP in PyTorch, train on NVIDIA GPU
Step 2: Use matrices 64x1024, 1024x512, 512x10 for two MLP layers
Step 3: Divide matrix multiplications into 128x128 tiles (boundary tiles smaller)
Step 4: Treat every tile matmul, activation, loss, gradient, weight update as separate task

Outputs:
  - DAG visualization with tiling, timing, parallelism, memory usage
  - Scaling from 64x64 to full size
  - Profiled execution results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json
import os
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ============================================================
# STEP 1 & 2: Build MLP with specified matrix dimensions
# ============================================================

class SimpleMLP(nn.Module):
    """Two-layer MLP: 1024 -> 512 -> 10 (input dim = 1024)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512, bias=False)  # Weight: 1024x512
        self.fc2 = nn.Linear(512, 10, bias=False)     # Weight: 512x10

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_baseline_mlp(device, epochs=5):
    """Step 1: Train the MLP normally as a baseline and capture timing."""
    model = SimpleMLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Batch of 64 samples, 1024 features -> matrices are 64x1024
    X = torch.randn(64, 1024, device=device)
    y = torch.randint(0, 10, (64,), device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"[Baseline] {epochs} epochs in {elapsed*1000:.2f} ms, final loss={loss.item():.4f}")
    return model, elapsed


# ============================================================
# STEP 3: Tiling logic
# ============================================================

TILE_SIZE = 128

@dataclass
class Tile:
    """Represents one tile of a matrix multiplication."""
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    k_start: int = 0
    k_end: int = 0

    @property
    def shape(self):
        return (self.row_end - self.row_start, self.col_end - self.col_start)

    @property
    def k_size(self):
        return self.k_end - self.k_start

    def memory_bytes(self):
        """Memory for input tiles + output tile (float32)."""
        m = self.row_end - self.row_start
        n = self.col_end - self.col_start
        k = self.k_end - self.k_start
        # A_tile: m x k, B_tile: k x n, C_tile: m x n
        return (m * k + k * n + m * n) * 4


def compute_tiles(M, N, K, tile_size=TILE_SIZE):
    """Divide an MxN = (MxK) @ (KxN) matmul into tiles."""
    tiles = []
    for i in range(0, M, tile_size):
        for j in range(0, N, tile_size):
            for k in range(0, K, tile_size):
                t = Tile(
                    row_start=i, row_end=min(i + tile_size, M),
                    col_start=j, col_end=min(j + tile_size, N),
                    k_start=k,  k_end=min(k + tile_size, K),
                )
                tiles.append(t)
    return tiles


def describe_tiling(M, N, K, tile_size=TILE_SIZE):
    """Print tiling breakdown for a matrix multiplication."""
    tiles = compute_tiles(M, N, K, tile_size)
    n_row = math.ceil(M / tile_size)
    n_col = math.ceil(N / tile_size)
    n_k   = math.ceil(K / tile_size)

    total_mem = sum(t.memory_bytes() for t in tiles)
    boundary = [t for t in tiles if t.shape[0] < tile_size or t.shape[1] < tile_size or t.k_size < tile_size]

    print(f"  MatMul ({M}x{K}) @ ({K}x{N}) -> ({M}x{N})")
    print(f"    Tile grid: {n_row} x {n_col} x {n_k} = {len(tiles)} tiles")
    print(f"    Full tiles: {len(tiles) - len(boundary)}, Boundary tiles: {len(boundary)}")
    print(f"    Total tile memory: {total_mem / 1024:.1f} KB")

    return tiles


# ============================================================
# STEP 4: DAG Construction - every operation is a task node
# ============================================================

@dataclass
class DAGTask:
    """A single task node in the computation DAG."""
    task_id: str
    task_type: str          # matmul_tile, accumulate, activation, loss, grad_loss,
                            # grad_tile, weight_update, etc.
    layer: Optional[int]    # which MLP layer (1 or 2)
    description: str
    dependencies: List[str] = field(default_factory=list)
    memory_bytes: int = 0
    runtime_us: float = 0.0  # microseconds, filled after profiling
    tile_info: Optional[Dict] = None  # tile coordinates
    parallelism_group: str = ""  # tasks in same group can run in parallel


class MoSAICDAG:
    """Builds the full DAG for one training iteration of the tiled MLP."""

    def __init__(self, batch_size=64, tile_size=TILE_SIZE):
        self.batch_size = batch_size
        self.tile_size = tile_size
        self.tasks: OrderedDict[str, DAGTask] = OrderedDict()
        self.task_counter = 0

        # Matrix dimensions from Step 2
        # Forward: X(64x1024) @ W1(1024x512) -> H1(64x512)
        #          ReLU(H1) -> A1(64x512)
        #          A1(64x512) @ W2(512x10) -> H2(64x10)
        # Loss: CrossEntropy(H2, y)
        # Backward: dL/dH2(64x10)
        #           dL/dW2 = A1^T(512x64) @ dH2(64x10) -> (512x10)
        #           dL/dA1 = dH2(64x10) @ W2^T(10x512) -> (64x512)
        #           dL/dH1 = dL/dA1 * relu'(H1) -> (64x512)
        #           dL/dW1 = X^T(1024x64) @ dL/dH1(64x512) -> (1024x512)
        # Weight update: W1 -= lr * dW1, W2 -= lr * dW2

        self._build_forward()
        self._build_loss()
        self._build_backward()
        self._build_weight_update()

    def _add_task(self, task_type, layer, desc, deps=None, mem=0, tile_info=None, par_group=""):
        self.task_counter += 1
        tid = f"T{self.task_counter:04d}"
        t = DAGTask(
            task_id=tid, task_type=task_type, layer=layer,
            description=desc, dependencies=deps or [],
            memory_bytes=mem, tile_info=tile_info,
            parallelism_group=par_group,
        )
        self.tasks[tid] = t
        return tid

    def _build_tiled_matmul(self, M, N, K, name_prefix, layer, input_deps, par_group_prefix):
        """Create tile tasks for one matrix multiplication, return output accumulation task IDs."""
        tiles = compute_tiles(M, N, K, self.tile_size)

        # Group tiles by output block (i, j) - tiles along k must be accumulated
        output_blocks = defaultdict(list)  # (i_block, j_block) -> list of tile task IDs

        for idx, tile in enumerate(tiles):
            i_blk = tile.row_start // self.tile_size
            j_blk = tile.col_start // self.tile_size
            k_blk = tile.k_start // self.tile_size

            tid = self._add_task(
                task_type="matmul_tile",
                layer=layer,
                desc=f"{name_prefix}_tile_r{i_blk}_c{j_blk}_k{k_blk} "
                     f"[{tile.row_end-tile.row_start}x{tile.k_size}]@[{tile.k_size}x{tile.col_end-tile.col_start}]",
                deps=input_deps,
                mem=tile.memory_bytes(),
                tile_info={
                    "row": (tile.row_start, tile.row_end),
                    "col": (tile.col_start, tile.col_end),
                    "k": (tile.k_start, tile.k_end),
                    "shape": tile.shape,
                },
                par_group=f"{par_group_prefix}_tiles",
            )
            output_blocks[(i_blk, j_blk)].append(tid)

        # Accumulation tasks: sum partial products along k dimension
        accum_ids = []
        for (i_blk, j_blk), tile_ids in sorted(output_blocks.items()):
            if len(tile_ids) == 1:
                accum_ids.append(tile_ids[0])  # no accumulation needed
            else:
                aid = self._add_task(
                    task_type="accumulate",
                    layer=layer,
                    desc=f"{name_prefix}_accum_r{i_blk}_c{j_blk} (sum {len(tile_ids)} partials)",
                    deps=tile_ids,
                    mem=(M // math.ceil(M / self.tile_size)) * (N // math.ceil(N / self.tile_size)) * 4,
                    par_group=f"{par_group_prefix}_accum",
                )
                accum_ids.append(aid)

        return accum_ids, tiles

    def _build_forward(self):
        """Forward pass: tiled matmuls + activations."""
        # Input task
        self.input_id = self._add_task("input", None, "Load input X (64x1024)", mem=64*1024*4)

        # Layer 1: X(64x1024) @ W1(1024x512) -> H1(64x512)
        print("\n--- Layer 1 Forward Tiling ---")
        self.layer1_fwd_ids, self.layer1_fwd_tiles = self._build_tiled_matmul(
            64, 512, 1024, "L1_fwd", 1, [self.input_id], "L1_fwd"
        )

        # ReLU activation on each output block
        self.relu_ids = []
        for i, acc_id in enumerate(self.layer1_fwd_ids):
            rid = self._add_task(
                "activation", 1, f"ReLU_block_{i}",
                deps=[acc_id], mem=self.tile_size * self.tile_size * 4,
                par_group="L1_relu",
            )
            self.relu_ids.append(rid)

        # Layer 2: A1(64x512) @ W2(512x10) -> H2(64x10)
        print("\n--- Layer 2 Forward Tiling ---")
        self.layer2_fwd_ids, self.layer2_fwd_tiles = self._build_tiled_matmul(
            64, 10, 512, "L2_fwd", 2, self.relu_ids, "L2_fwd"
        )

    def _build_loss(self):
        """Loss computation."""
        self.loss_id = self._add_task(
            "loss", None, "CrossEntropyLoss(H2, y)",
            deps=self.layer2_fwd_ids, mem=64*10*4 + 64*4,
        )

    def _build_backward(self):
        """Backward pass: gradient computations as tiled matmuls."""
        # dL/dH2 (64x10) - gradient of loss w.r.t. output
        self.grad_h2_id = self._add_task(
            "grad_loss", None, "Grad_dL_dH2 (64x10)",
            deps=[self.loss_id], mem=64*10*4,
        )

        # dL/dW2 = A1^T(512x64) @ dH2(64x10) -> (512x10)
        print("\n--- dL/dW2 Backward Tiling ---")
        self.grad_w2_ids, _ = self._build_tiled_matmul(
            512, 10, 64, "dW2", 2, [self.grad_h2_id] + self.relu_ids, "bwd_dW2"
        )

        # dL/dA1 = dH2(64x10) @ W2^T(10x512) -> (64x512)
        print("\n--- dL/dA1 Backward Tiling ---")
        self.grad_a1_ids, _ = self._build_tiled_matmul(
            64, 512, 10, "dA1", 2, [self.grad_h2_id], "bwd_dA1"
        )

        # dL/dH1 = dL/dA1 * relu'(H1) (element-wise, per block)
        self.grad_h1_ids = []
        for i, (ga1_id, relu_id) in enumerate(zip(self.grad_a1_ids, self.relu_ids)):
            gid = self._add_task(
                "grad_activation", 1, f"dReLU_block_{i} (element-wise)",
                deps=[ga1_id, relu_id], mem=self.tile_size * self.tile_size * 4,
                par_group="bwd_drelu",
            )
            self.grad_h1_ids.append(gid)

        # dL/dW1 = X^T(1024x64) @ dH1(64x512) -> (1024x512)
        print("\n--- dL/dW1 Backward Tiling ---")
        self.grad_w1_ids, _ = self._build_tiled_matmul(
            1024, 512, 64, "dW1", 1, [self.input_id] + self.grad_h1_ids, "bwd_dW1"
        )

    def _build_weight_update(self):
        """Weight updates: W -= lr * dW, one task per weight tile."""
        self.wu_ids = []
        for i, gw2_id in enumerate(self.grad_w2_ids):
            wid = self._add_task(
                "weight_update", 2, f"W2_update_block_{i}",
                deps=[gw2_id], mem=512*10*4,
                par_group="wu_W2",
            )
            self.wu_ids.append(wid)

        for i, gw1_id in enumerate(self.grad_w1_ids):
            wid = self._add_task(
                "weight_update", 1, f"W1_update_block_{i}",
                deps=[gw1_id], mem=self.tile_size * self.tile_size * 4,
                par_group="wu_W1",
            )
            self.wu_ids.append(wid)


# ============================================================
# PROFILING: Time each task type on the GPU
# ============================================================

def profile_task(task: DAGTask, device):
    """Run the actual GPU operation for a task and measure time."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if task.task_type == "matmul_tile":
        ti = task.tile_info
        m = ti["row"][1] - ti["row"][0]
        n = ti["col"][1] - ti["col"][0]
        k = ti["k"][1] - ti["k"][0]
        A = torch.randn(m, k, device=device)
        B = torch.randn(k, n, device=device)
        # Warmup
        _ = A @ B
        torch.cuda.synchronize()
        start.record()
        for _ in range(100):
            C = A @ B
        end.record()
        torch.cuda.synchronize()
        task.runtime_us = start.elapsed_time(end) * 1000 / 100  # ms->us, avg

    elif task.task_type == "activation":
        A = torch.randn(64, 512, device=device)
        torch.cuda.synchronize()
        start.record()
        for _ in range(100):
            _ = F.relu(A)
        end.record()
        torch.cuda.synchronize()
        task.runtime_us = start.elapsed_time(end) * 1000 / 100

    elif task.task_type in ("loss", "grad_loss"):
        logits = torch.randn(64, 10, device=device)
        y = torch.randint(0, 10, (64,), device=device)
        torch.cuda.synchronize()
        start.record()
        for _ in range(100):
            _ = F.cross_entropy(logits, y)
        end.record()
        torch.cuda.synchronize()
        task.runtime_us = start.elapsed_time(end) * 1000 / 100

    elif task.task_type == "accumulate":
        A = torch.randn(128, 128, device=device)
        B = torch.randn(128, 128, device=device)
        torch.cuda.synchronize()
        start.record()
        for _ in range(100):
            _ = A + B
        end.record()
        torch.cuda.synchronize()
        task.runtime_us = start.elapsed_time(end) * 1000 / 100

    elif task.task_type == "grad_activation":
        A = torch.randn(64, 512, device=device)
        mask = (A > 0).float()
        torch.cuda.synchronize()
        start.record()
        for _ in range(100):
            _ = A * mask
        end.record()
        torch.cuda.synchronize()
        task.runtime_us = start.elapsed_time(end) * 1000 / 100

    elif task.task_type == "weight_update":
        W = torch.randn(128, 128, device=device)
        dW = torch.randn(128, 128, device=device)
        torch.cuda.synchronize()
        start.record()
        for _ in range(100):
            W.sub_(dW, alpha=0.01)
        end.record()
        torch.cuda.synchronize()
        task.runtime_us = start.elapsed_time(end) * 1000 / 100

    elif task.task_type == "input":
        task.runtime_us = 0.1  # negligible

    return task.runtime_us


# ============================================================
# DAG ANALYSIS: Parallelism, Critical Path, Memory
# ============================================================

def analyze_dag(dag: MoSAICDAG):
    """Compute DAG metrics."""
    tasks = dag.tasks

    # --- Parallelism groups ---
    groups = defaultdict(list)
    for t in tasks.values():
        if t.parallelism_group:
            groups[t.parallelism_group].append(t.task_id)

    # --- Critical path (longest path by runtime) ---
    # Topological order (tasks are already in insertion order = topo order)
    dist = {}  # task_id -> longest path time to reach this task
    pred = {}  # predecessor on critical path
    for tid, task in tasks.items():
        if not task.dependencies:
            dist[tid] = task.runtime_us
            pred[tid] = None
        else:
            best = max((dist.get(d, 0), d) for d in task.dependencies)
            dist[tid] = best[0] + task.runtime_us
            pred[tid] = best[1]

    # Find the end of the critical path
    cp_end = max(dist, key=lambda t: dist[t])
    critical_path = []
    cur = cp_end
    while cur:
        critical_path.append(cur)
        cur = pred[cur]
    critical_path.reverse()

    # --- Depth levels (for parallelism width) ---
    depth = {}
    for tid, task in tasks.items():
        if not task.dependencies:
            depth[tid] = 0
        else:
            depth[tid] = max(depth[d] for d in task.dependencies) + 1

    max_depth = max(depth.values())
    level_widths = defaultdict(int)
    for tid, d in depth.items():
        level_widths[d] += 1

    # --- Task type summary ---
    type_counts = defaultdict(int)
    type_times = defaultdict(float)
    type_mem = defaultdict(int)
    for t in tasks.values():
        type_counts[t.task_type] += 1
        type_times[t.task_type] += t.runtime_us
        type_mem[t.task_type] += t.memory_bytes

    total_work = sum(t.runtime_us for t in tasks.values())
    total_mem = sum(t.memory_bytes for t in tasks.values())
    cp_time = dist[cp_end]
    max_parallelism = max(level_widths.values())
    avg_parallelism = total_work / cp_time if cp_time > 0 else 1

    return {
        "total_tasks": len(tasks),
        "total_work_us": total_work,
        "critical_path_us": cp_time,
        "critical_path_tasks": critical_path,
        "critical_path_len": len(critical_path),
        "max_parallelism": max_parallelism,
        "avg_parallelism": avg_parallelism,
        "max_depth": max_depth,
        "total_memory_bytes": total_mem,
        "type_counts": dict(type_counts),
        "type_times_us": dict(type_times),
        "type_memory_bytes": dict(type_mem),
        "parallelism_groups": {k: len(v) for k, v in groups.items()},
        "level_widths": dict(level_widths),
        "depth_map": depth,
    }


# ============================================================
# SCALING EXPERIMENT: 64x64 -> full size
# ============================================================

def scaling_experiment(device):
    """Show how tiling scales from small to full matrix sizes."""
    print("\n" + "="*70)
    print("SCALING EXPERIMENT: Tiling from small to full size")
    print("="*70)

    configs = [
        ("Tiny (64x64)",    64,   64,   64),
        ("Small (64x256)",  64,  256,  256),
        ("Medium (64x512)", 64,  512,  512),
        ("Full Layer 1",    64,  512, 1024),
        ("Full Layer 2",    64,   10,  512),
    ]

    results = []
    for name, M, N, K in configs:
        tiles = compute_tiles(M, N, K)
        total_mem = sum(t.memory_bytes() for t in tiles)
        boundary = sum(1 for t in tiles if t.shape[0] < TILE_SIZE or t.shape[1] < TILE_SIZE or t.k_size < TILE_SIZE)

        # Profile one tile
        m = min(M, TILE_SIZE)
        n = min(N, TILE_SIZE)
        k = min(K, TILE_SIZE)
        A = torch.randn(m, k, device=device)
        B = torch.randn(k, n, device=device)
        _ = A @ B; torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(200):
            _ = A @ B
        end.record()
        torch.cuda.synchronize()
        tile_time_us = start.elapsed_time(end) * 1000 / 200

        max_parallel = math.ceil(M / TILE_SIZE) * math.ceil(N / TILE_SIZE)

        r = {
            "name": name,
            "MxN_KxN": f"({M}x{K})@({K}x{N})",
            "total_tiles": len(tiles),
            "full_tiles": len(tiles) - boundary,
            "boundary_tiles": boundary,
            "tile_memory_KB": total_mem / 1024,
            "tile_time_us": tile_time_us,
            "max_parallel_output_blocks": max_parallel,
            "est_serial_us": tile_time_us * len(tiles),
            "est_parallel_us": tile_time_us * math.ceil(len(tiles) / max_parallel) if max_parallel > 0 else tile_time_us * len(tiles),
        }
        results.append(r)

        print(f"\n  {name}: {r['MxN_KxN']}")
        print(f"    Tiles: {r['total_tiles']} ({r['full_tiles']} full, {r['boundary_tiles']} boundary)")
        print(f"    Tile memory: {r['tile_memory_KB']:.1f} KB")
        print(f"    One tile: {r['tile_time_us']:.1f} us")
        print(f"    Max parallel blocks: {r['max_parallel_output_blocks']}")
        print(f"    Est. serial: {r['est_serial_us']:.1f} us | parallel: {r['est_parallel_us']:.1f} us")

    return results


# ============================================================
# VISUALIZATION: Generate DOT graph
# ============================================================

def generate_dag_dot(dag: MoSAICDAG, analysis: dict, filename="mosaic_dag"):
    """Generate a Graphviz DOT file for the DAG."""
    depth_map = analysis["depth_map"]
    cp_set = set(analysis["critical_path_tasks"])

    colors = {
        "input": "#4CAF50",
        "matmul_tile": "#2196F3",
        "accumulate": "#FF9800",
        "activation": "#9C27B0",
        "loss": "#F44336",
        "grad_loss": "#F44336",
        "grad_activation": "#E91E63",
        "weight_update": "#00BCD4",
    }

    lines = [
        'digraph MoSAIC_DAG {',
        '  rankdir=TB;',
        '  node [shape=box, style="filled,rounded", fontsize=9, fontname="Helvetica"];',
        '  edge [color="#666666", arrowsize=0.6];',
        f'  label="MoSAIC DAG: {analysis["total_tasks"]} tasks | '
        f'Critical Path: {analysis["critical_path_us"]:.0f} us ({analysis["critical_path_len"]} tasks) | '
        f'Max Parallelism: {analysis["max_parallelism"]} | '
        f'Memory: {analysis["total_memory_bytes"]/1024:.0f} KB";',
        '  labelloc=t; fontsize=14;',
        '',
    ]

    # Group nodes by depth level using subgraphs
    levels = defaultdict(list)
    for tid, d in depth_map.items():
        levels[d].append(tid)

    for d in sorted(levels.keys()):
        lines.append(f'  subgraph cluster_level_{d} {{')
        lines.append(f'    style=invis;')
        lines.append(f'    rank=same;')
        for tid in levels[d]:
            task = dag.tasks[tid]
            color = colors.get(task.task_type, "#9E9E9E")
            border = "red" if tid in cp_set else "#333333"
            penwidth = "3" if tid in cp_set else "1"
            label = (f"{tid}\\n{task.task_type}\\n"
                     f"{task.description[:30]}\\n"
                     f"{task.runtime_us:.1f}us | {task.memory_bytes/1024:.1f}KB")
            lines.append(f'    {tid} [label="{label}", fillcolor="{color}", '
                         f'color="{border}", penwidth={penwidth}, fontcolor="white"];')
        lines.append('  }')
        lines.append('')

    # Edges
    for tid, task in dag.tasks.items():
        for dep in task.dependencies:
            style = "bold" if (dep in cp_set and tid in cp_set) else "solid"
            color = "red" if (dep in cp_set and tid in cp_set) else "#666666"
            lines.append(f'  {dep} -> {tid} [style={style}, color="{color}"];')

    lines.append('}')

    dot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{filename}.dot")
    with open(dot_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nDAG DOT file written to: {dot_path}")

    # Try to render
    try:
        import subprocess
        svg_path = dot_path.replace('.dot', '.svg')
        subprocess.run(['dot', '-Tsvg', dot_path, '-o', svg_path], check=True, timeout=30)
        print(f"DAG SVG rendered to: {svg_path}")
    except Exception as e:
        print(f"(Could not render SVG: {e}. Install graphviz to render.)")

    return dot_path


# ============================================================
# TEXT VISUALIZATION: ASCII DAG summary
# ============================================================

def print_dag_summary(dag: MoSAICDAG, analysis: dict):
    """Print a comprehensive text summary of the DAG."""
    print("\n" + "="*70)
    print("DAG SUMMARY")
    print("="*70)

    print(f"\nTotal tasks: {analysis['total_tasks']}")
    print(f"DAG depth (levels): {analysis['max_depth'] + 1}")
    print(f"Critical path: {analysis['critical_path_len']} tasks, {analysis['critical_path_us']:.1f} us")
    print(f"Total work: {analysis['total_work_us']:.1f} us")
    print(f"Max parallelism (widest level): {analysis['max_parallelism']}")
    print(f"Average parallelism: {analysis['avg_parallelism']:.1f}x")
    print(f"Total memory across all tasks: {analysis['total_memory_bytes']/1024:.1f} KB")

    print(f"\n--- Task Type Breakdown ---")
    print(f"{'Type':<20} {'Count':>6} {'Total Time (us)':>15} {'Memory (KB)':>12}")
    print("-" * 55)
    for tt in sorted(analysis['type_counts'].keys()):
        print(f"{tt:<20} {analysis['type_counts'][tt]:>6} "
              f"{analysis['type_times_us'].get(tt, 0):>15.1f} "
              f"{analysis['type_memory_bytes'].get(tt, 0)/1024:>12.1f}")

    print(f"\n--- Parallelism Groups (tasks that CAN run concurrently) ---")
    for grp, cnt in sorted(analysis['parallelism_groups'].items()):
        print(f"  {grp}: {cnt} parallel tasks")

    print(f"\n--- DAG Level Widths (parallelism at each depth) ---")
    widths = analysis['level_widths']
    max_w = max(widths.values())
    for d in sorted(widths.keys()):
        bar = "#" * int(widths[d] / max_w * 50)
        print(f"  Level {d:>3}: {widths[d]:>4} tasks  {bar}")

    print(f"\n--- Critical Path (longest execution chain) ---")
    for tid in analysis['critical_path_tasks'][:15]:
        t = dag.tasks[tid]
        print(f"  {tid}: {t.task_type:<18} {t.runtime_us:>8.1f} us  {t.description[:40]}")
    if len(analysis['critical_path_tasks']) > 15:
        print(f"  ... and {len(analysis['critical_path_tasks']) - 15} more tasks")


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Tile size: {TILE_SIZE}x{TILE_SIZE}")

    # Step 1: Baseline training
    print("\n" + "="*70)
    print("STEP 1 & 2: Baseline MLP Training")
    print("="*70)
    model, baseline_time = train_baseline_mlp(device)

    # Step 3: Show tiling breakdown
    print("\n" + "="*70)
    print("STEP 3: Tiling Breakdown (128x128 tiles)")
    print("="*70)
    print("\nForward Layer 1: X(64x1024) @ W1(1024x512)")
    describe_tiling(64, 512, 1024)
    print("\nForward Layer 2: A1(64x512) @ W2(512x10)")
    describe_tiling(64, 10, 512)
    print("\nBackward dW2: A1^T(512x64) @ dH2(64x10)")
    describe_tiling(512, 10, 64)
    print("\nBackward dA1: dH2(64x10) @ W2^T(10x512)")
    describe_tiling(64, 512, 10)
    print("\nBackward dW1: X^T(1024x64) @ dH1(64x512)")
    describe_tiling(1024, 512, 64)

    # Step 4: Build the full DAG
    print("\n" + "="*70)
    print("STEP 4: Building DAG (every tile op = separate task)")
    print("="*70)
    dag = MoSAICDAG(batch_size=64, tile_size=TILE_SIZE)

    # Profile all tasks
    print("\nProfiling all tasks on GPU...")
    task_types_profiled = set()
    for tid, task in dag.tasks.items():
        profile_task(task, device)
        if task.task_type not in task_types_profiled:
            task_types_profiled.add(task.task_type)

    # Analyze
    analysis = analyze_dag(dag)
    print_dag_summary(dag, analysis)

    # Generate visualization
    generate_dag_dot(dag, analysis)

    # Scaling experiment
    scaling_results = scaling_experiment(device)

    # Final comparison
    print("\n" + "="*70)
    print("COMPARISON: Baseline vs Tiled DAG")
    print("="*70)
    print(f"  Baseline (PyTorch, 5 epochs):   {baseline_time*1000:.2f} ms total")
    print(f"  DAG total work (1 iteration):   {analysis['total_work_us']/1000:.2f} ms")
    print(f"  DAG critical path (1 iter):     {analysis['critical_path_us']/1000:.2f} ms")
    print(f"  Speedup potential (work/CP):     {analysis['avg_parallelism']:.1f}x with {analysis['max_parallelism']} max parallel tasks")
    print(f"  Total DAG memory footprint:     {analysis['total_memory_bytes']/1024:.1f} KB")

    # Save results to JSON
    results = {
        "baseline_time_ms": baseline_time * 1000,
        "dag_total_tasks": analysis["total_tasks"],
        "dag_total_work_us": analysis["total_work_us"],
        "dag_critical_path_us": analysis["critical_path_us"],
        "dag_critical_path_len": analysis["critical_path_len"],
        "dag_max_parallelism": analysis["max_parallelism"],
        "dag_avg_parallelism": analysis["avg_parallelism"],
        "dag_total_memory_KB": analysis["total_memory_bytes"] / 1024,
        "task_type_counts": analysis["type_counts"],
        "task_type_times_us": analysis["type_times_us"],
        "parallelism_groups": analysis["parallelism_groups"],
        "scaling_results": scaling_results,
    }
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dag_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
