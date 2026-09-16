"""
MoSAIC Memory Hierarchy Experiments
=====================================
Publication-quality memory hierarchy analyses:
  1. Cache-aware scheduling: L2 temporal locality comparison (HEFT vs MoSAIC)
  2. Working set over time: live memory footprint at each schedule step
  3. Register pressure analysis: registers/thread vs tile size, spill threshold
  4. Bandwidth utilization timeline: GB/s used over time during schedule
  5. Data locality metric: fraction of tile inputs already in L2 when scheduled
  6. Memory-aware theta: add L2 reuse feature to phi(v), re-learn theta
"""

import json
import os
import math
import heapq
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ============================================================
# GPU SPEC
# ============================================================

@dataclass
class GPUSpec:
    name: str = "NVIDIA RTX 500 Ada"
    sm_count: int = 16
    shared_mem_per_block: int = 49152       # 48 KB
    regs_per_sm: int = 65536
    regs_per_block: int = 65536
    max_threads_per_sm: int = 1536
    max_threads_per_block: int = 1024
    l2_cache_bytes: int = 16 * 1024 * 1024  # 16 MB
    global_mem_bytes: int = 4 * 1024**3     # ~4 GB
    memory_bandwidth_gbps: float = 128.0
    warp_size: int = 32

GPU = GPUSpec()

# ============================================================
# TILE & TASK MODEL
# ============================================================

@dataclass
class TileTask:
    """A single tiled matrix operation."""
    name: str
    op_type: str          # "matmul", "gelu", "softmax", "layernorm", "loss", "grad", "update"
    tile_m: int
    tile_n: int
    tile_k: int           # 0 for element-wise ops
    flops: int
    input_bytes: int      # bytes read from global memory
    output_bytes: int     # bytes written to global memory
    shared_mem_bytes: int # shared memory needed
    regs_per_thread: int  # estimated registers per thread
    threads: int          # threads launched
    runtime_us: float     # profiled runtime
    # Dependencies: which tiles' outputs does this tile read?
    input_tiles: List[str]  # names of tiles whose outputs are inputs

def estimate_tile_task(name, op_type, M, N, K=0, elem_bytes=4):
    """Create a TileTask with realistic resource estimates."""
    if op_type == "matmul":
        flops = 2 * M * N * K
        # Read A tile (M×K) + B tile (K×N), write C tile (M×N)
        input_bytes = (M * K + K * N) * elem_bytes
        output_bytes = M * N * elem_bytes
        shared_mem = (M * K + K * N) * elem_bytes  # both tiles in smem
        threads = ((M * N + 31) // 32) * 32
        threads = min(threads, GPU.max_threads_per_block)
        # Registers: ~32 for simple matmul, ~48 for fused, ~64 for large
        regs = 32 if K <= 32 else 48 if K <= 64 else 64
        runtime = flops / (8.6e9) * 1e6  # rough estimate at peak FLOPS
    else:
        # Element-wise: GeLU, softmax, layernorm, etc.
        flops = M * N * (10 if op_type in ("gelu", "softmax") else 5)
        input_bytes = M * N * elem_bytes
        output_bytes = M * N * elem_bytes
        shared_mem = M * N * elem_bytes  # one tile in smem
        threads = ((M * N + 31) // 32) * 32
        threads = min(threads, GPU.max_threads_per_block)
        regs = 16 if op_type == "layernorm" else 12
        runtime = flops / (8.6e9) * 1e6

    return TileTask(
        name=name, op_type=op_type,
        tile_m=M, tile_n=N, tile_k=K,
        flops=flops, input_bytes=input_bytes, output_bytes=output_bytes,
        shared_mem_bytes=shared_mem, regs_per_thread=regs,
        threads=threads, runtime_us=max(0.1, runtime),
        input_tiles=[],
    )


def build_transformer_tiled_dag(batch=4, seq=32, hidden=128, heads=2, ffn_dim=256, tile_size=32):
    """Build a full transformer layer DAG with tile-level tasks and rich data dependencies.

    Key: every matmul tile C[i,j] += A[i,k] * B[k,j] explicitly tracks which
    upstream accumulator tiles produced A[i,k] and B[k,j]. This creates the
    data-sharing graph needed for cache locality analysis.
    """
    tasks = {}       # name -> TileTask
    edges = []       # (src, dst, bytes)
    T = tile_size

    def add_matmul_tiles(prefix, M, N, K, a_prefix=None, b_prefix=None):
        """Add tiled matmul C = A @ B with full dependency tracking.

        a_prefix: prefix of the layer producing matrix A (row tiles)
        b_prefix: prefix of the layer producing matrix B (col tiles)
        Returns dict mapping (row, col) -> accumulator tile name.
        """
        acc_map = {}  # (i, j) -> acc_name
        for i in range(0, M, T):
            for j in range(0, N, T):
                tm = min(T, M - i)
                tn = min(T, N - j)
                acc_name = f"{prefix}_acc_r{i}_c{j}"
                acc = estimate_tile_task(acc_name, "matmul", tm, tn, 0)
                acc.flops = 0
                acc.input_bytes = 0
                acc.shared_mem_bytes = 0
                acc.regs_per_thread = 8
                acc.runtime_us = 0.05
                tasks[acc_name] = acc
                acc_map[(i, j)] = acc_name

                for k in range(0, K, T):
                    tk = min(T, K - k)
                    tile_name = f"{prefix}_tile_r{i}_c{j}_k{k}"
                    tile = estimate_tile_task(tile_name, "matmul", tm, tn, tk)
                    tasks[tile_name] = tile

                    # Track data dependencies:
                    # This tile reads A[i, k] and B[k, j] from upstream layers
                    if a_prefix:
                        # Try acc format first, then elementwise format
                        for fmt in [f"{a_prefix}_acc_r{i}_c{k}", f"{a_prefix}_r{i}_c{k}"]:
                            if fmt in tasks:
                                tile.input_tiles.append(fmt)
                                edges.append((fmt, tile_name, tm * tk * 4))
                                break
                    if b_prefix:
                        for fmt in [f"{b_prefix}_acc_r{k}_c{j}", f"{b_prefix}_r{k}_c{j}"]:
                            if fmt in tasks:
                                tile.input_tiles.append(fmt)
                                edges.append((fmt, tile_name, tk * tn * 4))
                                break

                    # tile -> accumulator
                    edges.append((tile_name, acc_name, tm * tn * 4))

        return acc_map

    def add_elementwise(prefix, M, N, op_type, input_acc_map):
        """Add element-wise op tiles, each depending on the corresponding accumulator tile.

        input_acc_map: dict mapping (row, col) -> source tile name
        """
        out_map = {}
        for i in range(0, M, T):
            for j in range(0, N, T):
                tm = min(T, M - i)
                tn = min(T, N - j)
                name = f"{prefix}_r{i}_c{j}"
                tile = estimate_tile_task(name, op_type, tm, tn)
                # Depend on matching source tile
                src_key = (i, j)
                if src_key in input_acc_map:
                    src_name = input_acc_map[src_key]
                    tile.input_tiles.append(src_name)
                    edges.append((src_name, name, tm * tn * 4))
                tasks[name] = tile
                out_map[(i, j)] = name
        return out_map

    def make_acc_prefix_map(prefix, M, N):
        """Create a map from (i,j) -> acc tile name for use with add_elementwise.
        Used when the source is identified by prefix string rather than acc_map dict."""
        result = {}
        for i in range(0, M, T):
            for j in range(0, N, T):
                acc_name = f"{prefix}_acc_r{i}_c{j}"
                if acc_name in tasks:
                    result[(i, j)] = acc_name
                else:
                    # Try _r{i}_c{j} format (elementwise output)
                    alt_name = f"{prefix}_r{i}_c{j}"
                    if alt_name in tasks:
                        result[(i, j)] = alt_name
        return result

    BN = batch * seq  # combined batch dimension

    # QKV projections: (BN x hidden) @ (hidden x hidden)
    # Input X is shared across Q, K, V — this is the key data-sharing point
    # Create "input" placeholder accumulators to represent X
    x_map = {}
    for i in range(0, BN, T):
        for j in range(0, hidden, T):
            tm = min(T, BN - i)
            tn = min(T, hidden - j)
            name = f"X_acc_r{i}_c{j}"
            x_tile = estimate_tile_task(name, "matmul", tm, tn, 0)
            x_tile.flops = 0
            x_tile.input_bytes = tm * tn * 4
            x_tile.output_bytes = tm * tn * 4
            x_tile.shared_mem_bytes = 0
            x_tile.regs_per_thread = 8
            x_tile.runtime_us = 0.02  # just a data source
            tasks[name] = x_tile
            x_map[(i, j)] = name

    # Weight matrices as data sources
    wq_map, wk_map, wv_map = {}, {}, {}
    for prefix, w_map in [("Wq", wq_map), ("Wk", wk_map), ("Wv", wv_map)]:
        for i in range(0, hidden, T):
            for j in range(0, hidden, T):
                tm = min(T, hidden - i)
                tn = min(T, hidden - j)
                name = f"{prefix}_acc_r{i}_c{j}"
                w_tile = estimate_tile_task(name, "matmul", tm, tn, 0)
                w_tile.flops = 0
                w_tile.input_bytes = tm * tn * 4
                w_tile.output_bytes = tm * tn * 4
                w_tile.shared_mem_bytes = 0
                w_tile.regs_per_thread = 8
                w_tile.runtime_us = 0.01
                tasks[name] = w_tile
                w_map[(i, j)] = name

    # Q = X @ Wq, K = X @ Wk, V = X @ Wv  (all share X!)
    q_map = add_matmul_tiles("Q", BN, hidden, hidden, a_prefix="X", b_prefix="Wq")
    k_map = add_matmul_tiles("K", BN, hidden, hidden, a_prefix="X", b_prefix="Wk")
    v_map = add_matmul_tiles("V", BN, hidden, hidden, a_prefix="X", b_prefix="Wv")

    # Attention per head
    head_dim = hidden // heads
    av_maps = {}
    for h in range(heads):
        # Attn = Q_h @ K_h^T: we use Q and K output tiles as inputs
        attn_map = add_matmul_tiles(f"Attn_h{h}", BN, BN, head_dim,
                                     a_prefix="Q", b_prefix="K")
        soft_map = add_elementwise(f"Soft_h{h}", BN, BN, "softmax", attn_map)
        # AV = Softmax @ V_h
        av_map = add_matmul_tiles(f"AV_h{h}", BN, head_dim, BN,
                                   a_prefix=f"Soft_h{h}", b_prefix="V")
        av_maps[h] = av_map

    # Output projection: concat heads then project
    # Simplified: Oproj = AV_concat @ Wo
    oproj_map = add_matmul_tiles("Oproj", BN, hidden, hidden, a_prefix="AV_h0")

    # LayerNorm 1
    ln1_map = add_elementwise("LN1", BN, hidden, "layernorm", oproj_map)

    # FFN1: LN1 @ W1
    ffn1_map = add_matmul_tiles("FFN1", BN, ffn_dim, hidden, a_prefix="LN1")

    # GeLU
    gelu_map = add_elementwise("GeLU", BN, ffn_dim, "gelu", ffn1_map)

    # FFN2: GeLU @ W2
    ffn2_map = add_matmul_tiles("FFN2", BN, hidden, ffn_dim, a_prefix="GeLU")

    # LayerNorm 2
    ln2_map = add_elementwise("LN2", BN, hidden, "layernorm", ffn2_map)

    return tasks, edges


# ============================================================
# SCHEDULING HELPERS (standalone, no SAGA dependency)
# ============================================================

def compute_features_standalone(tasks, edges):
    """Compute 5-feature vector for each task (standalone, no SAGA)."""
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    edge_weight = {}
    for src, dst, w in edges:
        if src in tasks and dst in tasks:
            successors[src].append(dst)
            predecessors[dst].append(src)
            edge_weight[(src, dst)] = w

    # Topological sort
    in_deg = {n: len(predecessors[n]) for n in tasks}
    queue = [n for n in tasks if in_deg[n] == 0]
    topo = []
    while queue:
        n = queue.pop(0)
        topo.append(n)
        for s in successors[n]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    # rank_u (upward rank)
    rank_u = {}
    for name in reversed(topo):
        t = tasks[name]
        succs = successors[name]
        if not succs:
            rank_u[name] = t.runtime_us
        else:
            rank_u[name] = t.runtime_us + max(
                edge_weight.get((name, s), 0) / (GPU.memory_bandwidth_gbps * 1e3) + rank_u[s]
                for s in succs
            )

    # depth
    depth = {}
    for name in topo:
        preds = predecessors[name]
        depth[name] = 0 if not preds else max(depth[p] for p in preds) + 1

    # Feature vectors
    phi = {}
    for name in tasks:
        fanout = len(successors[name])
        indegree = len(predecessors[name])
        comm = max((edge_weight.get((name, s), 0) for s in successors[name]), default=0.0)
        comm_norm = comm / (GPU.memory_bandwidth_gbps * 1e3)  # normalize to us
        phi[name] = np.array([rank_u[name], depth[name], fanout, indegree, comm_norm])

    return phi, successors, predecessors, edge_weight, topo


def list_schedule_standalone(tasks, edges, phi, theta, num_procs=2):
    """List schedule with heapq, returns per-task placement and timing."""
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    edge_weight = {}
    for src, dst, w in edges:
        if src in tasks and dst in tasks:
            successors[src].append(dst)
            predecessors[dst].append(src)
            edge_weight[(src, dst)] = w

    priority = {name: float(np.dot(theta, phi[name])) for name in tasks}
    in_count = {name: len(predecessors[name]) for name in tasks}
    ready = [(-priority[n], n) for n in tasks if in_count[n] == 0]
    heapq.heapify(ready)

    proc_avail = [0.0] * num_procs
    finish = {}
    task_proc = {}
    task_start = {}
    schedule_order = []  # ordered list of (name, proc, start, end)

    while ready:
        neg_pri, name = heapq.heappop(ready)
        if name in finish:
            continue
        t = tasks[name]
        best_p, best_start, best_end = 0, 0, float('inf')
        for p in range(num_procs):
            earliest = proc_avail[p]
            for par in predecessors[name]:
                par_fin = finish[par]
                comm_us = edge_weight.get((par, name), 0) / (GPU.memory_bandwidth_gbps * 1e3)
                if task_proc[par] != p:
                    earliest = max(earliest, par_fin + comm_us)
                else:
                    earliest = max(earliest, par_fin)
            end = earliest + t.runtime_us
            if end < best_end:
                best_end = end
                best_start = earliest
                best_p = p

        finish[name] = best_end
        task_start[name] = best_start
        proc_avail[best_p] = best_end
        task_proc[name] = best_p
        schedule_order.append((name, best_p, best_start, best_end))

        for child in successors[name]:
            in_count[child] -= 1
            if in_count[child] == 0:
                heapq.heappush(ready, (-priority[child], child))

    makespan = max(finish.values()) if finish else 0
    return makespan, schedule_order, finish, task_proc


def learn_theta_standalone(tasks, edges, phi, ref_ms, num_procs=2,
                           n_explore=500, n_refine=300, n_fine=100, seed=42):
    best_theta = None
    best_gap = float('inf')
    rng = np.random.RandomState(seed)

    for _ in range(n_explore):
        theta = rng.randn(len(list(phi.values())[0]))
        theta[0] = abs(theta[0]) * 2
        ms, *_ = list_schedule_standalone(tasks, edges, phi, theta, num_procs)
        gap = (ms - ref_ms) / ref_ms if ref_ms > 0 else 0
        if gap < best_gap:
            best_gap = gap
            best_theta = theta.copy()

    for i in range(n_refine):
        scale = 0.5 * (1 - i / n_refine)
        theta = best_theta + rng.randn(len(best_theta)) * scale
        ms, *_ = list_schedule_standalone(tasks, edges, phi, theta, num_procs)
        gap = (ms - ref_ms) / ref_ms if ref_ms > 0 else 0
        if gap < best_gap:
            best_gap = gap
            best_theta = theta.copy()

    for _ in range(n_fine):
        theta = best_theta + rng.randn(len(best_theta)) * 0.1
        ms, *_ = list_schedule_standalone(tasks, edges, phi, theta, num_procs)
        gap = (ms - ref_ms) / ref_ms if ref_ms > 0 else 0
        if gap < best_gap:
            best_gap = gap
            best_theta = theta.copy()

    return best_theta, best_gap


# HEFT baseline (standalone)
def heft_schedule(tasks, edges, num_procs=2):
    """Simple HEFT: schedule by descending rank_u, assign to earliest-finish processor."""
    phi, successors, predecessors, edge_weight, topo = compute_features_standalone(tasks, edges)
    # rank_u is phi[name][0]
    rank_u = {name: phi[name][0] for name in tasks}
    sorted_tasks = sorted(tasks.keys(), key=lambda n: -rank_u[n])

    proc_avail = [0.0] * num_procs
    finish = {}
    task_proc = {}
    task_start = {}
    schedule_order = []

    for name in sorted_tasks:
        t = tasks[name]
        best_p, best_start, best_end = 0, 0, float('inf')
        for p in range(num_procs):
            earliest = proc_avail[p]
            for par in predecessors[name]:
                par_fin = finish[par]
                comm_us = edge_weight.get((par, name), 0) / (GPU.memory_bandwidth_gbps * 1e3)
                if task_proc[par] != p:
                    earliest = max(earliest, par_fin + comm_us)
                else:
                    earliest = max(earliest, par_fin)
            end = earliest + t.runtime_us
            if end < best_end:
                best_end = end
                best_start = earliest
                best_p = p

        finish[name] = best_end
        task_start[name] = best_start
        proc_avail[best_p] = best_end
        task_proc[name] = best_p
        schedule_order.append((name, best_p, best_start, best_end))

    makespan = max(finish.values()) if finish else 0
    return makespan, schedule_order, finish, task_proc


# ============================================================
# ANALYSIS 1: Cache-Aware Scheduling (L2 Temporal Locality)
# ============================================================

def run_cache_locality(tasks, edges, tile_size=32):
    """Compare L2 temporal locality: when tiles share data, how far apart are they scheduled?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Cache-Aware Scheduling -- L2 Temporal Locality")
    print("=" * 70)

    phi, *_ = compute_features_standalone(tasks, edges)

    # HEFT schedule
    heft_ms, heft_order, heft_finish, heft_proc = heft_schedule(tasks, edges)

    # MoSAIC schedule
    theta, _ = learn_theta_standalone(tasks, edges, phi, heft_ms)
    mosaic_ms, mosaic_order, mosaic_finish, mosaic_proc = list_schedule_standalone(tasks, edges, phi, theta)

    # Build data-sharing graph: tiles that read the same input tile
    # For each task, track its input_tiles
    consumers = defaultdict(list)  # data_source -> list of consumer task names
    for name, t in tasks.items():
        for inp in t.input_tiles:
            consumers[inp].append(name)

    def compute_locality_metrics(schedule_order, finish_times):
        """For each pair of tasks sharing an input, compute time gap."""
        # Build order index
        order_idx = {name: idx for idx, (name, *_) in enumerate(schedule_order)}
        time_at = {name: start for name, proc, start, end in schedule_order}

        gaps = []
        l2_hits = 0
        l2_misses = 0
        l2_line_size = 128  # bytes
        l2_total_lines = GPU.l2_cache_bytes // l2_line_size

        for source, consumer_list in consumers.items():
            if len(consumer_list) < 2:
                continue
            # Sort consumers by schedule order
            sorted_consumers = sorted(consumer_list, key=lambda n: order_idx.get(n, 0))
            for i in range(len(sorted_consumers) - 1):
                a, b = sorted_consumers[i], sorted_consumers[i + 1]
                if a in time_at and b in time_at:
                    time_gap = abs(time_at[b] - time_at[a])
                    order_gap = abs(order_idx.get(b, 0) - order_idx.get(a, 0))
                    gaps.append({"time_gap_us": time_gap, "order_gap": order_gap})
                    # Heuristic: if gap < L2 eviction window, count as hit
                    # L2 eviction window ~ tasks between them * avg bytes/task / L2 size
                    intervening_bytes = sum(
                        tasks[schedule_order[k][0]].output_bytes
                        for k in range(min(order_idx.get(a, 0), order_idx.get(b, 0)) + 1,
                                       max(order_idx.get(a, 0), order_idx.get(b, 0)))
                        if k < len(schedule_order)
                    )
                    if intervening_bytes < GPU.l2_cache_bytes:
                        l2_hits += 1
                    else:
                        l2_misses += 1

        return gaps, l2_hits, l2_misses

    heft_gaps, heft_hits, heft_misses = compute_locality_metrics(heft_order, heft_finish)
    mosaic_gaps, mosaic_hits, mosaic_misses = compute_locality_metrics(mosaic_order, mosaic_finish)

    heft_avg_time_gap = np.mean([g["time_gap_us"] for g in heft_gaps]) if heft_gaps else 0
    mosaic_avg_time_gap = np.mean([g["time_gap_us"] for g in mosaic_gaps]) if mosaic_gaps else 0
    heft_avg_order_gap = np.mean([g["order_gap"] for g in heft_gaps]) if heft_gaps else 0
    mosaic_avg_order_gap = np.mean([g["order_gap"] for g in mosaic_gaps]) if mosaic_gaps else 0

    heft_hit_rate = heft_hits / max(1, heft_hits + heft_misses) * 100
    mosaic_hit_rate = mosaic_hits / max(1, mosaic_hits + mosaic_misses) * 100

    print(f"\n  Data-sharing pairs found: {len(heft_gaps)}")
    print(f"\n  {'Metric':<35} {'HEFT':>12} {'MoSAIC':>12} {'Improvement':>12}")
    print(f"  {'-' * 71}")
    print(f"  {'Avg time gap (us)':<35} {heft_avg_time_gap:>12.2f} {mosaic_avg_time_gap:>12.2f} {(1 - mosaic_avg_time_gap / max(0.001, heft_avg_time_gap)) * 100:>+11.1f}%")
    print(f"  {'Avg schedule-order gap':<35} {heft_avg_order_gap:>12.1f} {mosaic_avg_order_gap:>12.1f} {(1 - mosaic_avg_order_gap / max(0.001, heft_avg_order_gap)) * 100:>+11.1f}%")
    print(f"  {'L2 hit rate (estimated)':<35} {heft_hit_rate:>11.1f}% {mosaic_hit_rate:>11.1f}% {mosaic_hit_rate - heft_hit_rate:>+11.1f}%")
    print(f"  {'Makespan':<35} {heft_ms:>12.1f} {mosaic_ms:>12.1f} {(1 - mosaic_ms / heft_ms) * 100:>+11.1f}%")

    return {
        "data_sharing_pairs": len(heft_gaps),
        "heft": {
            "makespan": round(heft_ms, 2),
            "avg_time_gap_us": round(heft_avg_time_gap, 2),
            "avg_order_gap": round(heft_avg_order_gap, 1),
            "l2_hit_rate_pct": round(heft_hit_rate, 1),
        },
        "mosaic": {
            "makespan": round(mosaic_ms, 2),
            "avg_time_gap_us": round(mosaic_avg_time_gap, 2),
            "avg_order_gap": round(mosaic_avg_order_gap, 1),
            "l2_hit_rate_pct": round(mosaic_hit_rate, 1),
        },
    }


# ============================================================
# ANALYSIS 2: Working Set Over Time
# ============================================================

def run_working_set(tasks, edges, tile_size=32):
    """Track live memory footprint at each time step of the schedule."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Working Set Over Time")
    print("=" * 70)

    phi, successors, predecessors, edge_weight, topo = compute_features_standalone(tasks, edges)

    heft_ms, heft_order, heft_finish, heft_proc = heft_schedule(tasks, edges)
    theta, _ = learn_theta_standalone(tasks, edges, phi, heft_ms)
    mosaic_ms, mosaic_order, mosaic_finish, mosaic_proc = list_schedule_standalone(tasks, edges, phi, theta)

    def compute_working_set_timeline(schedule_order, finish_times):
        """Compute live bytes at each task completion event."""
        # A tile's output is "live" from when it's produced until all consumers finish
        last_consumer = defaultdict(float)  # tile_name -> latest consumer finish time
        for name, proc, start, end in schedule_order:
            for inp in tasks[name].input_tiles:
                if inp in finish_times:
                    last_consumer[inp] = max(last_consumer[inp], end)

        # For tiles with no consumers, output dies immediately
        for name in tasks:
            if name not in last_consumer:
                last_consumer[name] = finish_times.get(name, 0)

        # Build events: (time, +/- bytes)
        events = []
        for name, proc, start, end in schedule_order:
            out_bytes = tasks[name].output_bytes
            events.append((end, +out_bytes, "produce", name))
            death_time = last_consumer.get(name, end)
            events.append((death_time, -out_bytes, "consume", name))

        events.sort(key=lambda e: (e[0], -e[1]))  # sort by time, frees before allocs at same time

        timeline = []
        live_bytes = 0
        for time, delta, event_type, name in events:
            live_bytes += delta
            live_bytes = max(0, live_bytes)  # safety
            timeline.append({"time_us": time, "live_bytes": live_bytes, "live_KB": live_bytes / 1024})

        return timeline

    heft_timeline = compute_working_set_timeline(heft_order, heft_finish)
    mosaic_timeline = compute_working_set_timeline(mosaic_order, mosaic_finish)

    heft_peak = max(e["live_KB"] for e in heft_timeline) if heft_timeline else 0
    mosaic_peak = max(e["live_KB"] for e in mosaic_timeline) if mosaic_timeline else 0
    heft_avg = np.mean([e["live_KB"] for e in heft_timeline]) if heft_timeline else 0
    mosaic_avg = np.mean([e["live_KB"] for e in mosaic_timeline]) if mosaic_timeline else 0

    # Sample timeline at regular intervals for compact output
    def sample_timeline(timeline, n_samples=20):
        if not timeline:
            return []
        max_time = max(e["time_us"] for e in timeline)
        step = max_time / n_samples
        samples = []
        t_idx = 0
        for i in range(n_samples + 1):
            t = i * step
            while t_idx < len(timeline) - 1 and timeline[t_idx + 1]["time_us"] <= t:
                t_idx += 1
            samples.append({"time_us": round(t, 1), "live_KB": round(timeline[t_idx]["live_KB"], 1)})
        return samples

    print(f"\n  {'Metric':<35} {'HEFT':>12} {'MoSAIC':>12} {'Improvement':>12}")
    print(f"  {'-' * 71}")
    print(f"  {'Peak working set (KB)':<35} {heft_peak:>12.1f} {mosaic_peak:>12.1f} {(1 - mosaic_peak / max(0.001, heft_peak)) * 100:>+11.1f}%")
    print(f"  {'Avg working set (KB)':<35} {heft_avg:>12.1f} {mosaic_avg:>12.1f} {(1 - mosaic_avg / max(0.001, heft_avg)) * 100:>+11.1f}%")
    l2_kb = GPU.l2_cache_bytes // 1024
    print(f"  {'Fits L2 cache (' + str(l2_kb) + ' KB)':<35} {'YES' if heft_peak * 1024 < GPU.l2_cache_bytes else 'NO':>12} {'YES' if mosaic_peak * 1024 < GPU.l2_cache_bytes else 'NO':>12}")

    return {
        "heft": {
            "peak_KB": round(heft_peak, 1), "avg_KB": round(heft_avg, 1),
            "fits_l2": heft_peak * 1024 < GPU.l2_cache_bytes,
            "timeline_samples": sample_timeline(heft_timeline),
        },
        "mosaic": {
            "peak_KB": round(mosaic_peak, 1), "avg_KB": round(mosaic_avg, 1),
            "fits_l2": mosaic_peak * 1024 < GPU.l2_cache_bytes,
            "timeline_samples": sample_timeline(mosaic_timeline),
        },
    }


# ============================================================
# ANALYSIS 3: Register Pressure
# ============================================================

def run_register_pressure():
    """Analyze register usage per tile size and identify spill thresholds."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Register Pressure Analysis")
    print("=" * 70)

    results = {}

    print(f"\n  {'Tile':<8} {'Threads':>8} {'Regs/Th':>8} {'TotalRegs':>10} {'Blk/SM':>7} {'Occupancy':>10} {'SmemKB':>7} {'SmemOK':>7} {'Spills':>7}")
    print(f"  {'-' * 72}")

    for tile_size in [8, 16, 32, 64, 128]:
        M, N, K = tile_size, tile_size, tile_size

        # Thread count: one thread per output element, rounded to warps
        threads = min(M * N, GPU.max_threads_per_block)
        threads = ((threads + 31) // 32) * 32

        # Register model based on CUDA matmul patterns:
        # - Small tiles (<=16): ~16 regs (few accumulators, simple loop)
        # - Medium tiles (32): ~32 regs (more accumulators + loop unrolling)
        # - Large tiles (64): ~48 regs (register tiling for performance)
        # - Very large (128): ~96 regs (heavy register tiling, risk of spills)
        if tile_size <= 16:
            regs_per_thread = 16
        elif tile_size <= 32:
            regs_per_thread = 32
        elif tile_size <= 64:
            regs_per_thread = 48
        else:
            regs_per_thread = 96

        total_regs = threads * regs_per_thread
        blocks_per_sm = min(
            GPU.regs_per_sm // max(1, total_regs),
            GPU.max_threads_per_sm // max(1, threads),
        )
        blocks_per_sm = max(1, blocks_per_sm)
        active_threads = blocks_per_sm * threads
        occupancy = min(100.0, active_threads / GPU.max_threads_per_sm * 100)

        smem_needed = (M * K + K * N) * 4  # both input tiles in shared memory
        smem_fits = smem_needed <= GPU.shared_mem_per_block
        spills = regs_per_thread > 128 or total_regs > GPU.regs_per_sm

        print(f"  {tile_size:>3}x{tile_size:<3} {threads:>8} {regs_per_thread:>8} {total_regs:>10} {blocks_per_sm:>7} {occupancy:>9.1f}% {smem_needed // 1024:>5}KB {'YES' if smem_fits else 'NO':>7} {'YES' if spills else 'NO':>7}")

        results[tile_size] = {
            "threads": threads,
            "regs_per_thread": regs_per_thread,
            "total_regs": total_regs,
            "blocks_per_sm": blocks_per_sm,
            "occupancy_pct": round(occupancy, 1),
            "register_spills": spills,
            "shared_mem_bytes": smem_needed,
            "shared_mem_fits": smem_needed <= GPU.shared_mem_per_block,
        }

    print(f"\n  Hardware limits:")
    print(f"    Registers per SM:       {GPU.regs_per_sm:,}")
    print(f"    Max threads per SM:     {GPU.max_threads_per_sm:,}")
    print(f"    Max threads per block:  {GPU.max_threads_per_block:,}")
    print(f"    Shared mem per block:   {GPU.shared_mem_per_block // 1024} KB")

    # Find optimal tile size: must fit in shared memory AND registers, prefer larger tiles
    # (fewer tasks = less scheduling overhead, better data reuse)
    valid_tiles = [t for t in results if not results[t]["register_spills"] and results[t]["shared_mem_fits"]]
    best_tile = max(valid_tiles) if valid_tiles else min(results.keys())
    print(f"\n  Largest valid tile (no spills, smem fits): {best_tile}x{best_tile}")
    print(f"    Occupancy: {results[best_tile]['occupancy_pct']:.1f}%, "
          f"Regs/thread: {results[best_tile]['regs_per_thread']}, "
          f"Shared mem: {results[best_tile]['shared_mem_bytes'] // 1024} KB")
    print(f"  Note: 128x128 needs {128*128*2*4 // 1024} KB shared mem (exceeds {GPU.shared_mem_per_block // 1024} KB limit)")
    print(f"  Note: 128x128 needs {results.get(128, {}).get('regs_per_thread', 96)} regs/thread (spill risk above 128)")

    results["optimal_tile"] = best_tile

    return results


# ============================================================
# ANALYSIS 4: Bandwidth Utilization Timeline
# ============================================================

def run_bandwidth_timeline(tasks, edges):
    """Plot memory bandwidth usage over time for HEFT vs MoSAIC schedules."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Bandwidth Utilization Timeline")
    print("=" * 70)

    phi, *_ = compute_features_standalone(tasks, edges)
    heft_ms, heft_order, heft_finish, heft_proc = heft_schedule(tasks, edges)
    theta, _ = learn_theta_standalone(tasks, edges, phi, heft_ms)
    mosaic_ms, mosaic_order, mosaic_finish, mosaic_proc = list_schedule_standalone(tasks, edges, phi, theta)

    def compute_bw_timeline(schedule_order, n_bins=50):
        """Compute global memory bandwidth usage in time bins.

        Only counts global memory traffic (not shared memory reuse).
        For tiled matmul, only the first load from global -> shared counts;
        subsequent reuse within shared memory doesn't consume bandwidth.
        """
        if not schedule_order:
            return []
        max_time = max(end for _, _, _, end in schedule_order)
        bin_width = max_time / n_bins

        bw_bins = [0.0] * n_bins
        bytes_bins = [0.0] * n_bins

        for name, proc, start, end in schedule_order:
            t = tasks[name]
            # Global memory bytes: input_bytes (load) + output_bytes (store)
            # For matmul tiles with shared memory, data is loaded once from global
            # and reused K/T times within shared memory
            if t.op_type == "matmul" and t.tile_k > 0:
                # Only one global load per tile, not per accumulation
                global_bytes = t.input_bytes + t.output_bytes
            else:
                global_bytes = t.input_bytes + t.output_bytes

            duration = max(0.001, end - start)
            bw_gbps = global_bytes / (duration * 1e-6) / 1e9  # GB/s
            # Cap at hardware peak — can't exceed physical bandwidth
            bw_gbps = min(bw_gbps, GPU.memory_bandwidth_gbps)

            start_bin = int(start / bin_width)
            end_bin = int(end / bin_width)
            start_bin = min(start_bin, n_bins - 1)
            end_bin = min(end_bin, n_bins - 1)

            for b in range(start_bin, end_bin + 1):
                bin_start = b * bin_width
                bin_end = (b + 1) * bin_width
                overlap_start = max(start, bin_start)
                overlap_end = min(end, bin_end)
                overlap_frac = max(0, overlap_end - overlap_start) / max(0.001, duration)
                bw_bins[b] += bw_gbps * overlap_frac
                bytes_bins[b] += global_bytes * overlap_frac

        # Cap per-bin bandwidth at 2x peak (2 streams can overlap)
        peak_bw = GPU.memory_bandwidth_gbps * 2  # 2 streams
        timeline = []
        for i in range(n_bins):
            t_val = (i + 0.5) * bin_width
            effective_bw = min(bw_bins[i], peak_bw)
            timeline.append({
                "time_us": round(t_val, 2),
                "bandwidth_gbps": round(effective_bw, 2),
                "bytes_moved": round(bytes_bins[i], 0),
                "utilization_pct": round(min(100, effective_bw / GPU.memory_bandwidth_gbps * 100), 1),
            })
        return timeline

    heft_bw = compute_bw_timeline(heft_order)
    mosaic_bw = compute_bw_timeline(mosaic_order)

    heft_avg_util = np.mean([b["utilization_pct"] for b in heft_bw]) if heft_bw else 0
    mosaic_avg_util = np.mean([b["utilization_pct"] for b in mosaic_bw]) if mosaic_bw else 0
    heft_peak_bw = max(b["bandwidth_gbps"] for b in heft_bw) if heft_bw else 0
    mosaic_peak_bw = max(b["bandwidth_gbps"] for b in mosaic_bw) if mosaic_bw else 0
    heft_idle_bins = sum(1 for b in heft_bw if b["utilization_pct"] < 1.0)
    mosaic_idle_bins = sum(1 for b in mosaic_bw if b["utilization_pct"] < 1.0)

    print(f"\n  {'Metric':<35} {'HEFT':>12} {'MoSAIC':>12} {'Improvement':>12}")
    print(f"  {'-' * 71}")
    print(f"  {'Avg bandwidth utilization (%)':<35} {heft_avg_util:>11.1f}% {mosaic_avg_util:>11.1f}% {mosaic_avg_util - heft_avg_util:>+11.1f}%")
    print(f"  {'Peak bandwidth (GB/s)':<35} {heft_peak_bw:>12.1f} {mosaic_peak_bw:>12.1f}")
    print(f"  {'Idle time bins (of 50)':<35} {heft_idle_bins:>12} {mosaic_idle_bins:>12}")
    bw_limit = GPU.memory_bandwidth_gbps
    print(f"  {'Peak BW / HW limit (' + str(bw_limit) + ' GB/s)':<35} {heft_peak_bw / bw_limit * 100:>11.1f}% {mosaic_peak_bw / bw_limit * 100:>11.1f}%")

    return {
        "peak_hw_bandwidth_gbps": GPU.memory_bandwidth_gbps,
        "heft": {
            "avg_utilization_pct": round(heft_avg_util, 1),
            "peak_bandwidth_gbps": round(heft_peak_bw, 1),
            "idle_bins": heft_idle_bins,
            "timeline": heft_bw,
        },
        "mosaic": {
            "avg_utilization_pct": round(mosaic_avg_util, 1),
            "peak_bandwidth_gbps": round(mosaic_peak_bw, 1),
            "idle_bins": mosaic_idle_bins,
            "timeline": mosaic_bw,
        },
    }


# ============================================================
# ANALYSIS 5: Data Locality Metric
# ============================================================

def run_data_locality(tasks, edges):
    """Fraction of tile inputs already in L2 when a tile is scheduled."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Data Locality Metric -- L2 Residency at Schedule Time")
    print("=" * 70)

    phi, *_ = compute_features_standalone(tasks, edges)
    heft_ms, heft_order, heft_finish, heft_proc = heft_schedule(tasks, edges)
    theta, _ = learn_theta_standalone(tasks, edges, phi, heft_ms)
    mosaic_ms, mosaic_order, mosaic_finish, mosaic_proc = list_schedule_standalone(tasks, edges, phi, theta)

    def compute_locality_score(schedule_order):
        """Simulate L2 cache as LRU, track hit rate for tile inputs."""
        l2_capacity = GPU.l2_cache_bytes
        # Simple model: track bytes in cache as FIFO
        cache_contents = []  # list of (name, bytes) in insertion order
        cache_bytes = 0
        total_inputs = 0
        cache_hits = 0
        per_task_locality = []

        cache_set = set()

        for name, proc, start, end in schedule_order:
            t = tasks[name]
            # Check: are this task's input tiles in cache?
            task_hits = 0
            task_total = len(t.input_tiles)
            for inp in t.input_tiles:
                if inp in cache_set:
                    task_hits += 1
                    cache_hits += 1
                total_inputs += 1

            locality = task_hits / max(1, task_total)
            per_task_locality.append(locality)

            # Add this task's output to cache
            out_bytes = t.output_bytes
            cache_contents.append((name, out_bytes))
            cache_set.add(name)
            cache_bytes += out_bytes

            # Evict oldest entries if over capacity
            while cache_bytes > l2_capacity and cache_contents:
                evicted_name, evicted_bytes = cache_contents.pop(0)
                cache_bytes -= evicted_bytes
                cache_set.discard(evicted_name)

        hit_rate = cache_hits / max(1, total_inputs) * 100
        avg_locality = np.mean(per_task_locality) if per_task_locality else 0

        return hit_rate, avg_locality * 100, per_task_locality

    heft_hr, heft_loc, heft_per_task = compute_locality_score(heft_order)
    mosaic_hr, mosaic_loc, mosaic_per_task = compute_locality_score(mosaic_order)

    # Breakdown by op type
    def locality_by_op(schedule_order, per_task_locality):
        op_locality = defaultdict(list)
        for idx, (name, proc, start, end) in enumerate(schedule_order):
            if idx < len(per_task_locality):
                op_locality[tasks[name].op_type].append(per_task_locality[idx])
        return {op: round(np.mean(vals) * 100, 1) for op, vals in op_locality.items()}

    heft_by_op = locality_by_op(heft_order, heft_per_task)
    mosaic_by_op = locality_by_op(mosaic_order, mosaic_per_task)

    print(f"\n  {'Metric':<35} {'HEFT':>12} {'MoSAIC':>12} {'Improvement':>12}")
    print(f"  {'-' * 71}")
    print(f"  {'L2 cache hit rate (%)':<35} {heft_hr:>11.1f}% {mosaic_hr:>11.1f}% {mosaic_hr - heft_hr:>+11.1f}%")
    print(f"  {'Avg per-task locality (%)':<35} {heft_loc:>11.1f}% {mosaic_loc:>11.1f}% {mosaic_loc - heft_loc:>+11.1f}%")

    print(f"\n  Locality by operation type:")
    print(f"    {'Op Type':<15} {'HEFT':>10} {'MoSAIC':>10}")
    print(f"    {'-' * 35}")
    all_ops = sorted(set(list(heft_by_op.keys()) + list(mosaic_by_op.keys())))
    for op in all_ops:
        h = heft_by_op.get(op, 0)
        m = mosaic_by_op.get(op, 0)
        print(f"    {op:<15} {h:>9.1f}% {m:>9.1f}%")

    return {
        "l2_cache_size_KB": GPU.l2_cache_bytes // 1024,
        "heft": {
            "hit_rate_pct": round(heft_hr, 1),
            "avg_locality_pct": round(heft_loc, 1),
            "by_op_type": heft_by_op,
        },
        "mosaic": {
            "hit_rate_pct": round(mosaic_hr, 1),
            "avg_locality_pct": round(mosaic_loc, 1),
            "by_op_type": mosaic_by_op,
        },
    }


# ============================================================
# ANALYSIS 6: Memory-Aware Theta (6-feature phi)
# ============================================================

def run_memory_aware_theta(tasks, edges):
    """Add L2 reuse feature to phi(v), re-learn theta, compare vs 5-feature."""
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Memory-Aware Theta -- 6-Feature Learning")
    print("=" * 70)

    # Standard 5-feature
    phi5, successors, predecessors, edge_weight, topo = compute_features_standalone(tasks, edges)

    # Compute L2 reuse score: how many other tasks share an input tile with this task?
    # Higher = more reuse potential = should be scheduled near its data-sharing partners
    consumers = defaultdict(list)
    for name, t in tasks.items():
        for inp in t.input_tiles:
            consumers[inp].append(name)

    l2_reuse = {}
    for name in tasks:
        t = tasks[name]
        sharing_count = 0
        for inp in t.input_tiles:
            sharing_count += len(consumers[inp]) - 1  # exclude self
        l2_reuse[name] = sharing_count

    # Normalize
    max_reuse = max(l2_reuse.values()) if l2_reuse and max(l2_reuse.values()) > 0 else 1
    l2_reuse_norm = {name: l2_reuse[name] / max_reuse for name in l2_reuse}

    # Build 6-feature phi
    phi6 = {}
    for name in tasks:
        phi6[name] = np.append(phi5[name], l2_reuse_norm.get(name, 0))

    # HEFT baseline
    heft_ms, *_ = heft_schedule(tasks, edges)

    # Learn 5-feature theta
    theta5, gap5 = learn_theta_standalone(tasks, edges, phi5, heft_ms)
    ms5, *_ = list_schedule_standalone(tasks, edges, phi5, theta5)

    # Learn 6-feature theta
    theta6, gap6 = learn_theta_standalone(tasks, edges, phi6, heft_ms)
    ms6, *_ = list_schedule_standalone(tasks, edges, phi6, theta6)

    labels5 = ["rank_u", "depth", "fanout", "indegree", "comm_cost"]
    labels6 = labels5 + ["l2_reuse"]

    print(f"\n  HEFT makespan: {heft_ms:.2f}")
    print(f"\n  5-feature theta: [{', '.join(f'{x:.3f}' for x in theta5)}]")
    print(f"    Features: {', '.join(labels5)}")
    print(f"    Makespan: {ms5:.2f} (gap vs HEFT: {(ms5 - heft_ms) / heft_ms * 100:+.2f}%)")
    print(f"\n  6-feature theta: [{', '.join(f'{x:.3f}' for x in theta6)}]")
    print(f"    Features: {', '.join(labels6)}")
    print(f"    Makespan: {ms6:.2f} (gap vs HEFT: {(ms6 - heft_ms) / heft_ms * 100:+.2f}%)")

    improvement = (ms5 - ms6) / ms5 * 100
    print(f"\n  6-feature improvement over 5-feature: {improvement:+.3f}%")
    print(f"  L2 reuse weight in theta6: {theta6[5]:+.3f}")

    if theta6[5] > 0.1:
        print(f"  Interpretation: Positive l2_reuse weight means the scheduler prioritizes")
        print(f"  tasks with high data-sharing potential, improving cache locality.")
    elif theta6[5] < -0.1:
        print(f"  Interpretation: Negative l2_reuse weight means the scheduler deprioritizes")
        print(f"  high-sharing tasks, spreading them out to reduce cache contention.")
    else:
        print(f"  Interpretation: L2 reuse weight is near zero; cache locality is already")
        print(f"  implicitly captured by the other 5 features on this DAG.")

    # Test on multiple DAG sizes
    print(f"\n  === GENERALIZATION: 5-feature vs 6-feature across DAG sizes ===")
    print(f"  {'Config':<25} {'HEFT':>10} {'5-feat':>10} {'6-feat':>10} {'6f Gain':>10}")
    print(f"  {'-' * 65}")

    configs = [
        ("Small (b=2,s=16,h=64)", 2, 16, 64, 1, 128),
        ("Base (b=4,s=32,h=128)", 4, 32, 128, 2, 256),
        ("Wide (b=4,s=32,h=256)", 4, 32, 256, 4, 512),
    ]

    gen_results = []
    for label, b, s, h, heads, ffn in configs:
        cfg_tasks, cfg_edges = build_transformer_tiled_dag(b, s, h, heads, ffn)
        cfg_phi5, *_ = compute_features_standalone(cfg_tasks, cfg_edges)

        # Build 6-feature for this config
        cfg_consumers = defaultdict(list)
        for name, t in cfg_tasks.items():
            for inp in t.input_tiles:
                cfg_consumers[inp].append(name)
        cfg_l2 = {}
        for name in cfg_tasks:
            sc = sum(len(cfg_consumers[inp]) - 1 for inp in cfg_tasks[name].input_tiles)
            cfg_l2[name] = sc
        cfg_max = max(cfg_l2.values()) if cfg_l2 and max(cfg_l2.values()) > 0 else 1
        cfg_phi6 = {n: np.append(cfg_phi5[n], cfg_l2[n] / cfg_max) for n in cfg_tasks}

        cfg_heft, *_ = heft_schedule(cfg_tasks, cfg_edges)
        cfg_t5, _ = learn_theta_standalone(cfg_tasks, cfg_edges, cfg_phi5, cfg_heft)
        cfg_ms5, *_ = list_schedule_standalone(cfg_tasks, cfg_edges, cfg_phi5, cfg_t5)
        cfg_t6, _ = learn_theta_standalone(cfg_tasks, cfg_edges, cfg_phi6, cfg_heft)
        cfg_ms6, *_ = list_schedule_standalone(cfg_tasks, cfg_edges, cfg_phi6, cfg_t6)

        gain = (cfg_ms5 - cfg_ms6) / cfg_ms5 * 100
        print(f"  {label:<25} {cfg_heft:>10.1f} {cfg_ms5:>10.1f} {cfg_ms6:>10.1f} {gain:>+9.3f}%")
        gen_results.append({
            "config": label, "tasks": len(cfg_tasks),
            "heft": round(cfg_heft, 1), "ms_5feat": round(cfg_ms5, 1),
            "ms_6feat": round(cfg_ms6, 1), "gain_pct": round(gain, 3),
        })

    return {
        "heft_makespan": round(heft_ms, 2),
        "theta_5feat": theta5.tolist(),
        "makespan_5feat": round(ms5, 2),
        "theta_6feat": theta6.tolist(),
        "makespan_6feat": round(ms6, 2),
        "improvement_pct": round(improvement, 3),
        "l2_reuse_weight": round(float(theta6[5]), 3),
        "generalization": gen_results,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("MoSAIC Memory Hierarchy Experiments")
    print("=" * 70)

    all_results = {}

    # Build transformer DAG with tile-level data dependency tracking
    print("\n  Building transformer DAG with memory hierarchy annotations...")
    tasks, edges = build_transformer_tiled_dag(batch=4, seq=32, hidden=128, heads=2, ffn_dim=256, tile_size=32)
    print(f"  DAG: {len(tasks)} tasks, {len(edges)} edges")

    # Count tasks with input_tile tracking
    tasks_with_inputs = sum(1 for t in tasks.values() if t.input_tiles)
    print(f"  Tasks with tracked input dependencies: {tasks_with_inputs}")
    print(f"  Total data-sharing relationships: {sum(len(t.input_tiles) for t in tasks.values())}")

    # 1. Cache-aware scheduling
    all_results["cache_locality"] = run_cache_locality(tasks, edges)

    # 2. Working set over time
    all_results["working_set"] = run_working_set(tasks, edges)

    # 3. Register pressure
    all_results["register_pressure"] = run_register_pressure()

    # 4. Bandwidth utilization
    all_results["bandwidth"] = run_bandwidth_timeline(tasks, edges)

    # 5. Data locality
    all_results["data_locality"] = run_data_locality(tasks, edges)

    # 6. Memory-aware theta
    all_results["memory_aware_theta"] = run_memory_aware_theta(tasks, edges)

    # Save results
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_hierarchy_results.json")
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {out}")

    print("\n" + "=" * 70)
    print("MEMORY HIERARCHY EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
