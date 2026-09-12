"""
MoSAIC On-Chip Learning: Architecture-Aware DAG Experiment (v2)
================================================================
Addresses advisor feedback:
  1. GPU memory hierarchy modeled in DAG (registers, shared mem, L2 cache, global mem)
  2. Flexible/configurable tile sizes
  3. Scalability experiment (1X, 2X, 3X matrix sizes)
  4. Power and energy measurement per task and per scale

Steps implemented:
  1. Build small MLP in PyTorch, train on GPU
  2. Use matrices 64x1024, 1024x512, 512x10
  3. Divide matmuls into configurable tiles (boundary tiles handled)
  4. Every tile matmul, activation, loss, gradient, weight update = separate task
     with GPU memory hierarchy annotations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json
import os
import subprocess
import threading
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ============================================================
# GPU HARDWARE MODEL
# ============================================================

@dataclass
class GPUSpec:
    """Hardware specification of the target GPU."""
    name: str
    sm_count: int               # number of streaming multiprocessors
    max_threads_per_block: int
    max_threads_per_sm: int
    warp_size: int
    regs_per_sm: int            # total registers per SM
    regs_per_block: int
    shared_mem_per_block: int   # bytes (default)
    shared_mem_per_block_optin: int  # bytes (extended)
    shared_mem_per_sm: int      # bytes
    l2_cache_size: int          # bytes
    global_mem_bytes: int
    memory_bus_width: int       # bits
    memory_clock_mhz: int
    compute_capability: Tuple[int, int]

    @property
    def global_mem_bandwidth_gbps(self):
        """Theoretical peak global memory bandwidth in GB/s."""
        # bandwidth = bus_width * clock * 2 (DDR) / 8
        return (self.memory_bus_width * self.memory_clock_mhz * 2) / (8 * 1000)

    @property
    def regs_per_thread_max(self):
        """Max registers a single thread can use (hardware limit = 255)."""
        return 255

    def print_summary(self):
        print(f"\n{'='*70}")
        print(f"GPU HARDWARE SPECIFICATION: {self.name}")
        print(f"{'='*70}")
        print(f"  SMs: {self.sm_count}")
        print(f"  Max threads/block: {self.max_threads_per_block}")
        print(f"  Max threads/SM: {self.max_threads_per_sm}")
        print(f"  Warp size: {self.warp_size}")
        print(f"  Registers/SM: {self.regs_per_sm:,}")
        print(f"  Shared mem/block (default): {self.shared_mem_per_block:,} B ({self.shared_mem_per_block/1024:.0f} KB)")
        print(f"  Shared mem/block (optin):   {self.shared_mem_per_block_optin:,} B ({self.shared_mem_per_block_optin/1024:.0f} KB)")
        print(f"  Shared mem/SM: {self.shared_mem_per_sm:,} B ({self.shared_mem_per_sm/1024:.0f} KB)")
        print(f"  L2 cache: {self.l2_cache_size:,} B ({self.l2_cache_size/1024/1024:.0f} MB)")
        print(f"  Global memory: {self.global_mem_bytes/1024**3:.1f} GB")
        print(f"  Memory bandwidth: {self.global_mem_bandwidth_gbps:.1f} GB/s")
        print(f"  Compute capability: {self.compute_capability[0]}.{self.compute_capability[1]}")


def detect_gpu_spec(device) -> GPUSpec:
    """Auto-detect GPU hardware specs from PyTorch + nvidia-smi."""
    props = torch.cuda.get_device_properties(device)
    return GPUSpec(
        name=props.name,
        sm_count=props.multi_processor_count,
        max_threads_per_block=props.max_threads_per_block,
        max_threads_per_sm=getattr(props, 'max_threads_per_multi_processor', 1536),
        warp_size=props.warp_size,
        regs_per_sm=getattr(props, 'regs_per_multiprocessor', 65536),
        regs_per_block=getattr(props, 'regs_per_block', 65536),
        shared_mem_per_block=props.shared_memory_per_block,
        shared_mem_per_block_optin=getattr(props, 'shared_memory_per_block_optin', props.shared_memory_per_block),
        shared_mem_per_sm=getattr(props, 'shared_memory_per_multiprocessor', 102400),
        l2_cache_size=getattr(props, 'L2_cache_size', 16777216),
        global_mem_bytes=props.total_memory,
        memory_bus_width=getattr(props, 'memory_bus_width', 64),
        memory_clock_mhz=getattr(props, 'memory_clock_rate', 8001000) // 1000,
        compute_capability=(props.major, props.minor),
    )


# ============================================================
# MEMORY HIERARCHY MODEL PER TASK
# ============================================================

@dataclass
class MemoryPlacement:
    """Models where data lives in the GPU memory hierarchy for a task."""
    # Bytes in each level
    registers_bytes: int = 0        # data held in registers during computation
    shared_mem_bytes: int = 0       # data in shared memory (block-level)
    l2_cache_bytes: int = 0         # data expected to hit L2
    global_mem_read_bytes: int = 0  # data read from global (HBM/GDDR)
    global_mem_write_bytes: int = 0 # data written to global

    # Thread/block configuration
    threads_per_block: int = 0
    blocks_needed: int = 0
    regs_per_thread: int = 0

    # Derived
    shared_mem_per_block: int = 0
    occupancy_pct: float = 0.0      # theoretical occupancy

    @property
    def total_data_movement_bytes(self):
        return self.global_mem_read_bytes + self.global_mem_write_bytes

    @property
    def total_on_chip_bytes(self):
        return self.registers_bytes + self.shared_mem_bytes


def estimate_matmul_tile_memory(m, n, k, gpu: GPUSpec, tile_size: int) -> MemoryPlacement:
    """
    Estimate memory hierarchy usage for a tile GEMM: C(m,n) += A(m,k) @ B(k,n).

    Strategy modeled (similar to CUTLASS/cuBLAS tiled GEMM):
    - A_tile and B_tile are loaded from global memory into shared memory
    - Each thread computes a small sub-tile of C, accumulating in registers
    - C_tile is written back to global memory
    """
    mp = MemoryPlacement()
    elem_size = 4  # float32

    # --- Global memory transfers ---
    # Read A_tile(m x k) and B_tile(k x n) from global memory
    mp.global_mem_read_bytes = (m * k + k * n) * elem_size
    # Write C_tile(m x n) to global memory
    mp.global_mem_write_bytes = m * n * elem_size

    # --- Shared memory usage ---
    # Both A_tile and B_tile are staged in shared memory
    mp.shared_mem_bytes = (m * k + k * n) * elem_size
    # Per-block shared memory (A and B sub-tiles for one threadblock)
    mp.shared_mem_per_block = mp.shared_mem_bytes

    # --- Thread configuration ---
    # Use 16x16 thread blocks (common for matmul), each thread computes one element of C
    thread_tile_m = min(16, m)
    thread_tile_n = min(16, n)
    mp.threads_per_block = thread_tile_m * thread_tile_n
    mp.blocks_needed = max(1, (m * n) // mp.threads_per_block)

    # --- Register usage ---
    # Each thread accumulates one or more C elements (float accumulators)
    # Plus holds fragments of A and B rows/cols
    elements_per_thread = max(1, (m * n) // mp.threads_per_block)
    mp.regs_per_thread = min(
        elements_per_thread * 2 + 8,  # accumulators + A/B fragments + loop vars
        gpu.regs_per_thread_max
    )
    mp.registers_bytes = mp.regs_per_thread * mp.threads_per_block * 4  # 4 bytes per register

    # --- L2 cache ---
    # If total data fits in L2, expect cache hits on subsequent accesses
    total_data = mp.global_mem_read_bytes + mp.global_mem_write_bytes
    if total_data <= gpu.l2_cache_size:
        mp.l2_cache_bytes = total_data  # likely cached
    else:
        mp.l2_cache_bytes = gpu.l2_cache_size  # partial caching

    # --- Occupancy ---
    # Limited by: shared mem per SM, registers per SM, threads per SM
    blocks_by_shared = gpu.shared_mem_per_sm // max(mp.shared_mem_per_block, 1) if mp.shared_mem_per_block > 0 else 32
    blocks_by_regs = gpu.regs_per_sm // max(mp.regs_per_thread * mp.threads_per_block, 1)
    blocks_by_threads = gpu.max_threads_per_sm // max(mp.threads_per_block, 1)
    max_blocks_per_sm = min(blocks_by_shared, blocks_by_regs, blocks_by_threads, 32)
    active_threads = max_blocks_per_sm * mp.threads_per_block
    mp.occupancy_pct = min(100.0, (active_threads / gpu.max_threads_per_sm) * 100)

    return mp


def estimate_elementwise_memory(elements, gpu: GPUSpec) -> MemoryPlacement:
    """Memory model for element-wise ops (ReLU, gradient masking, weight update)."""
    mp = MemoryPlacement()
    elements = max(1, elements)
    elem_size = 4
    mp.global_mem_read_bytes = elements * elem_size
    mp.global_mem_write_bytes = elements * elem_size
    mp.threads_per_block = 256
    mp.blocks_needed = max(1, math.ceil(elements / mp.threads_per_block))
    mp.regs_per_thread = 8
    mp.registers_bytes = mp.regs_per_thread * mp.threads_per_block * 4
    mp.shared_mem_bytes = 0
    mp.shared_mem_per_block = 0
    total = mp.global_mem_read_bytes + mp.global_mem_write_bytes
    mp.l2_cache_bytes = min(total, gpu.l2_cache_size)
    active_threads = min(mp.blocks_needed, 32) * mp.threads_per_block
    mp.occupancy_pct = min(100.0, (active_threads / gpu.max_threads_per_sm) * 100)
    return mp


# ============================================================
# POWER / ENERGY MONITORING
# ============================================================

class PowerMonitor:
    """Sample GPU power in a background thread using nvidia-smi."""

    def __init__(self):
        self.samples = []
        self._running = False
        self._thread = None

    def start(self):
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _sample_loop(self):
        while self._running:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    power_w = float(result.stdout.strip())
                    self.samples.append((time.perf_counter(), power_w))
            except Exception:
                pass
            time.sleep(0.05)  # 50ms sampling interval

    def get_stats(self):
        if not self.samples:
            return {"avg_power_w": 0, "peak_power_w": 0, "samples": 0, "energy_mj": 0}
        powers = [s[1] for s in self.samples]
        times = [s[0] for s in self.samples]
        duration = times[-1] - times[0] if len(times) > 1 else 0.05
        avg_power = sum(powers) / len(powers)
        energy_mj = avg_power * duration * 1000  # W * s * 1000 = mJ
        return {
            "avg_power_w": round(avg_power, 2),
            "peak_power_w": round(max(powers), 2),
            "min_power_w": round(min(powers), 2),
            "samples": len(powers),
            "duration_s": round(duration, 4),
            "energy_mj": round(energy_mj, 2),
        }


# ============================================================
# MLP MODEL
# ============================================================

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ============================================================
# TILING (flexible tile size)
# ============================================================

@dataclass
class Tile:
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    k_start: int = 0
    k_end: int = 0

    @property
    def m(self): return self.row_end - self.row_start
    @property
    def n(self): return self.col_end - self.col_start
    @property
    def k(self): return self.k_end - self.k_start
    @property
    def is_full(self): return False  # set dynamically


def compute_tiles(M, N, K, tile_size):
    tiles = []
    for i in range(0, M, tile_size):
        for j in range(0, N, tile_size):
            for kk in range(0, K, tile_size):
                tiles.append(Tile(
                    row_start=i, row_end=min(i + tile_size, M),
                    col_start=j, col_end=min(j + tile_size, N),
                    k_start=kk, k_end=min(kk + tile_size, K),
                ))
    return tiles


def describe_tiling(M, N, K, tile_size):
    tiles = compute_tiles(M, N, K, tile_size)
    n_row = math.ceil(M / tile_size)
    n_col = math.ceil(N / tile_size)
    n_k = math.ceil(K / tile_size)
    boundary = [t for t in tiles if t.m < tile_size or t.n < tile_size or t.k < tile_size]
    full = len(tiles) - len(boundary)
    print(f"  ({M}x{K}) @ ({K}x{N}) -> ({M}x{N})")
    print(f"    Tile size: {tile_size}x{tile_size}")
    print(f"    Grid: {n_row} x {n_col} x {n_k} = {len(tiles)} tiles")
    print(f"    Full: {full}, Boundary: {len(boundary)}")
    return tiles


# ============================================================
# DAG TASK NODE
# ============================================================

@dataclass
class DAGTask:
    task_id: str
    task_type: str
    layer: Optional[int]
    description: str
    dependencies: List[str] = field(default_factory=list)
    memory: Optional[MemoryPlacement] = None
    runtime_us: float = 0.0
    parallelism_group: str = ""
    tile_coords: Optional[Dict] = None


# ============================================================
# DAG BUILDER
# ============================================================

class MoSAICDAG:
    def __init__(self, batch_size=64, input_dim=1024, hidden_dim=512,
                 output_dim=10, tile_size=128, gpu: GPUSpec = None):
        self.B = batch_size
        self.D_in = input_dim
        self.D_hid = hidden_dim
        self.D_out = output_dim
        self.tile_size = tile_size
        self.gpu = gpu
        self.tasks: OrderedDict[str, DAGTask] = OrderedDict()
        self.counter = 0

        self._build()

    def _add(self, ttype, layer, desc, deps=None, mem=None, par_group="", tile_coords=None):
        self.counter += 1
        tid = f"T{self.counter:04d}"
        self.tasks[tid] = DAGTask(
            task_id=tid, task_type=ttype, layer=layer,
            description=desc, dependencies=deps or [],
            memory=mem, parallelism_group=par_group,
            tile_coords=tile_coords,
        )
        return tid

    def _build_tiled_matmul(self, M, N, K, prefix, layer, deps, par_prefix):
        tiles = compute_tiles(M, N, K, self.tile_size)
        output_blocks = defaultdict(list)

        for tile in tiles:
            i_b = tile.row_start // self.tile_size
            j_b = tile.col_start // self.tile_size
            k_b = tile.k_start // self.tile_size

            mem = estimate_matmul_tile_memory(tile.m, tile.n, tile.k, self.gpu, self.tile_size) if self.gpu else None

            tid = self._add(
                "matmul_tile", layer,
                f"{prefix}_r{i_b}_c{j_b}_k{k_b} [{tile.m}x{tile.k}]@[{tile.k}x{tile.n}]",
                deps=deps, mem=mem, par_group=f"{par_prefix}_tiles",
                tile_coords={"i": i_b, "j": j_b, "k": k_b, "m": tile.m, "n": tile.n, "kk": tile.k},
            )
            output_blocks[(i_b, j_b)].append(tid)

        accum_ids = []
        for (i_b, j_b), tile_ids in sorted(output_blocks.items()):
            if len(tile_ids) == 1:
                accum_ids.append(tile_ids[0])
            else:
                m_out = min(self.tile_size, M - i_b * self.tile_size)
                n_out = min(self.tile_size, N - j_b * self.tile_size)
                mem = estimate_elementwise_memory(m_out * n_out, self.gpu) if self.gpu else None
                aid = self._add(
                    "accumulate", layer,
                    f"{prefix}_accum_r{i_b}_c{j_b} (sum {len(tile_ids)} partials)",
                    deps=tile_ids, mem=mem, par_group=f"{par_prefix}_accum",
                )
                accum_ids.append(aid)

        return accum_ids, tiles

    def _build(self):
        B, D_in, D_hid, D_out = self.B, self.D_in, self.D_hid, self.D_out

        # Input
        mem_in = estimate_elementwise_memory(B * D_in, self.gpu) if self.gpu else None
        self.input_id = self._add("input", None, f"Load X ({B}x{D_in})", mem=mem_in)

        # Forward L1: X(B,D_in) @ W1(D_in,D_hid)
        self.l1_fwd_ids, _ = self._build_tiled_matmul(
            B, D_hid, D_in, "L1_fwd", 1, [self.input_id], "L1_fwd")

        # ReLU
        self.relu_ids = []
        for i, acc_id in enumerate(self.l1_fwd_ids):
            m_out = min(self.tile_size, B)
            n_out = min(self.tile_size, D_hid - i * self.tile_size)
            mem = estimate_elementwise_memory(m_out * n_out, self.gpu) if self.gpu else None
            rid = self._add("activation", 1, f"ReLU_block_{i}",
                            deps=[acc_id], mem=mem, par_group="L1_relu")
            self.relu_ids.append(rid)

        # Forward L2: A1(B,D_hid) @ W2(D_hid,D_out)
        self.l2_fwd_ids, _ = self._build_tiled_matmul(
            B, D_out, D_hid, "L2_fwd", 2, self.relu_ids, "L2_fwd")

        # Loss
        mem_loss = estimate_elementwise_memory(B * D_out + B, self.gpu) if self.gpu else None
        self.loss_id = self._add("loss", None, f"CrossEntropy ({B}x{D_out})",
                                 deps=self.l2_fwd_ids, mem=mem_loss)

        # Grad loss
        mem_gl = estimate_elementwise_memory(B * D_out, self.gpu) if self.gpu else None
        self.grad_h2_id = self._add("grad_loss", None, f"dL/dH2 ({B}x{D_out})",
                                    deps=[self.loss_id], mem=mem_gl)

        # dW2 = A1^T(D_hid,B) @ dH2(B,D_out)
        self.grad_w2_ids, _ = self._build_tiled_matmul(
            D_hid, D_out, B, "dW2", 2,
            [self.grad_h2_id] + self.relu_ids, "bwd_dW2")

        # dA1 = dH2(B,D_out) @ W2^T(D_out,D_hid)
        self.grad_a1_ids, _ = self._build_tiled_matmul(
            B, D_hid, D_out, "dA1", 2, [self.grad_h2_id], "bwd_dA1")

        # dReLU
        self.grad_h1_ids = []
        for i, (ga1_id, relu_id) in enumerate(zip(self.grad_a1_ids, self.relu_ids)):
            m_out = min(self.tile_size, B)
            n_out = min(self.tile_size, D_hid - i * self.tile_size)
            mem = estimate_elementwise_memory(m_out * n_out, self.gpu) if self.gpu else None
            gid = self._add("grad_activation", 1, f"dReLU_block_{i}",
                            deps=[ga1_id, relu_id], mem=mem, par_group="bwd_drelu")
            self.grad_h1_ids.append(gid)

        # dW1 = X^T(D_in,B) @ dH1(B,D_hid)
        self.grad_w1_ids, _ = self._build_tiled_matmul(
            D_in, D_hid, B, "dW1", 1,
            [self.input_id] + self.grad_h1_ids, "bwd_dW1")

        # Weight updates
        self.wu_ids = []
        for i, gw2_id in enumerate(self.grad_w2_ids):
            m_out = min(self.tile_size, D_hid - i * self.tile_size)
            n_out = min(self.tile_size, D_out)
            mem = estimate_elementwise_memory(m_out * n_out, self.gpu) if self.gpu else None
            self.wu_ids.append(self._add(
                "weight_update", 2, f"W2_update_{i}",
                deps=[gw2_id], mem=mem, par_group="wu_W2"))

        for i, gw1_id in enumerate(self.grad_w1_ids):
            m_out = min(self.tile_size, D_in)
            n_out = min(self.tile_size, D_hid)
            mem = estimate_elementwise_memory(m_out * n_out, self.gpu) if self.gpu else None
            self.wu_ids.append(self._add(
                "weight_update", 1, f"W1_update_{i}",
                deps=[gw1_id], mem=mem, par_group="wu_W1"))


# ============================================================
# PROFILING
# ============================================================

def profile_task(task: DAGTask, device):
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    reps = 100

    if task.task_type == "matmul_tile" and task.tile_coords:
        tc = task.tile_coords
        A = torch.randn(tc["m"], tc["kk"], device=device)
        B = torch.randn(tc["kk"], tc["n"], device=device)
        _ = A @ B; torch.cuda.synchronize()
        start_ev.record()
        for _ in range(reps): _ = A @ B
        end_ev.record()
        torch.cuda.synchronize()
        task.runtime_us = start_ev.elapsed_time(end_ev) * 1000 / reps

    elif task.task_type == "activation":
        sz = max(1, task.memory.global_mem_read_bytes // 4 if task.memory else 64*128)
        A = torch.randn(sz, device=device)
        torch.cuda.synchronize()
        start_ev.record()
        for _ in range(reps): _ = F.relu(A)
        end_ev.record(); torch.cuda.synchronize()
        task.runtime_us = start_ev.elapsed_time(end_ev) * 1000 / reps

    elif task.task_type in ("loss", "grad_loss"):
        B_out = 64  # batch
        logits = torch.randn(B_out, 10, device=device)
        y = torch.randint(0, 10, (B_out,), device=device)
        torch.cuda.synchronize()
        start_ev.record()
        for _ in range(reps): _ = F.cross_entropy(logits, y)
        end_ev.record(); torch.cuda.synchronize()
        task.runtime_us = start_ev.elapsed_time(end_ev) * 1000 / reps

    elif task.task_type in ("accumulate", "grad_activation", "weight_update"):
        sz = max(1, task.memory.global_mem_read_bytes // 4 if task.memory else 128*128)
        A = torch.randn(sz, device=device)
        B = torch.randn(sz, device=device)
        torch.cuda.synchronize()
        start_ev.record()
        for _ in range(reps): _ = A + B
        end_ev.record(); torch.cuda.synchronize()
        task.runtime_us = start_ev.elapsed_time(end_ev) * 1000 / reps

    elif task.task_type == "input":
        task.runtime_us = 0.1

    return task.runtime_us


# ============================================================
# ANALYSIS
# ============================================================

def analyze_dag(dag: MoSAICDAG):
    tasks = dag.tasks

    # Parallelism groups
    groups = defaultdict(list)
    for t in tasks.values():
        if t.parallelism_group:
            groups[t.parallelism_group].append(t.task_id)

    # Critical path
    dist = {}
    pred = {}
    for tid, task in tasks.items():
        if not task.dependencies:
            dist[tid] = task.runtime_us
            pred[tid] = None
        else:
            best = max((dist.get(d, 0), d) for d in task.dependencies)
            dist[tid] = best[0] + task.runtime_us
            pred[tid] = best[1]

    cp_end = max(dist, key=lambda t: dist[t])
    critical_path = []
    cur = cp_end
    while cur:
        critical_path.append(cur)
        cur = pred[cur]
    critical_path.reverse()

    # Depth levels
    depth = {}
    for tid, task in tasks.items():
        if not task.dependencies:
            depth[tid] = 0
        else:
            depth[tid] = max(depth[d] for d in task.dependencies) + 1

    level_widths = defaultdict(int)
    for d in depth.values():
        level_widths[d] += 1

    # Type summary
    type_counts = defaultdict(int)
    type_times = defaultdict(float)
    for t in tasks.values():
        type_counts[t.task_type] += 1
        type_times[t.task_type] += t.runtime_us

    total_work = sum(t.runtime_us for t in tasks.values())
    cp_time = dist[cp_end]

    # Memory hierarchy aggregates
    total_global_read = sum(t.memory.global_mem_read_bytes for t in tasks.values() if t.memory)
    total_global_write = sum(t.memory.global_mem_write_bytes for t in tasks.values() if t.memory)
    total_shared = sum(t.memory.shared_mem_bytes for t in tasks.values() if t.memory)
    total_regs = sum(t.memory.registers_bytes for t in tasks.values() if t.memory)
    max_shared_per_task = max((t.memory.shared_mem_per_block for t in tasks.values() if t.memory), default=0)
    avg_occupancy = 0
    occ_tasks = [t for t in tasks.values() if t.memory and t.memory.occupancy_pct > 0]
    if occ_tasks:
        avg_occupancy = sum(t.memory.occupancy_pct for t in occ_tasks) / len(occ_tasks)

    return {
        "total_tasks": len(tasks),
        "total_work_us": total_work,
        "critical_path_us": cp_time,
        "critical_path_tasks": critical_path,
        "critical_path_len": len(critical_path),
        "max_parallelism": max(level_widths.values()),
        "avg_parallelism": total_work / cp_time if cp_time > 0 else 1,
        "max_depth": max(depth.values()),
        "type_counts": dict(type_counts),
        "type_times_us": dict(type_times),
        "parallelism_groups": {k: len(v) for k, v in groups.items()},
        "level_widths": dict(level_widths),
        "depth_map": depth,
        # Memory hierarchy
        "total_global_read_KB": total_global_read / 1024,
        "total_global_write_KB": total_global_write / 1024,
        "total_shared_mem_KB": total_shared / 1024,
        "total_register_KB": total_regs / 1024,
        "max_shared_per_task_KB": max_shared_per_task / 1024,
        "avg_occupancy_pct": avg_occupancy,
    }


# ============================================================
# PRINTING
# ============================================================

def print_dag_summary(dag: MoSAICDAG, analysis: dict, gpu: GPUSpec):
    print(f"\n{'='*70}")
    print(f"DAG SUMMARY (tile_size={dag.tile_size})")
    print(f"{'='*70}")

    print(f"\n--- Structure ---")
    print(f"  Total tasks:              {analysis['total_tasks']}")
    print(f"  DAG depth:                {analysis['max_depth'] + 1} levels")
    print(f"  Critical path:            {analysis['critical_path_len']} tasks, {analysis['critical_path_us']:.1f} us")
    print(f"  Total work:               {analysis['total_work_us']:.1f} us")
    print(f"  Max parallelism:          {analysis['max_parallelism']}")
    print(f"  Avg parallelism:          {analysis['avg_parallelism']:.1f}x")

    print(f"\n--- GPU Memory Hierarchy Usage ---")
    print(f"  Total global mem reads:   {analysis['total_global_read_KB']:.1f} KB")
    print(f"  Total global mem writes:  {analysis['total_global_write_KB']:.1f} KB")
    print(f"  Total data movement:      {(analysis['total_global_read_KB'] + analysis['total_global_write_KB']):.1f} KB")
    print(f"  Total shared mem used:    {analysis['total_shared_mem_KB']:.1f} KB")
    print(f"  Total register usage:     {analysis['total_register_KB']:.1f} KB")
    print(f"  Max shared mem / task:    {analysis['max_shared_per_task_KB']:.1f} KB (limit: {gpu.shared_mem_per_block/1024:.0f} KB/block)")
    print(f"  Avg occupancy:            {analysis['avg_occupancy_pct']:.1f}%")

    # Check if any task exceeds shared memory limit
    violations = []
    for t in dag.tasks.values():
        if t.memory and t.memory.shared_mem_per_block > gpu.shared_mem_per_block:
            violations.append((t.task_id, t.memory.shared_mem_per_block))
    if violations:
        print(f"\n  WARNING: {len(violations)} tasks exceed shared memory limit!")
        for tid, smem in violations[:5]:
            print(f"    {tid}: needs {smem/1024:.1f} KB > {gpu.shared_mem_per_block/1024:.0f} KB limit")
        print(f"    -> These tiles must be further subdivided or use global memory fallback")

    print(f"\n--- Task Type Breakdown ---")
    print(f"  {'Type':<20} {'Count':>6} {'Time (us)':>12}")
    print(f"  {'-'*40}")
    for tt in sorted(analysis['type_counts'].keys()):
        print(f"  {tt:<20} {analysis['type_counts'][tt]:>6} {analysis['type_times_us'].get(tt,0):>12.1f}")

    print(f"\n--- Parallelism Groups ---")
    for grp, cnt in sorted(analysis['parallelism_groups'].items()):
        print(f"  {grp}: {cnt} parallel tasks")

    print(f"\n--- Level Widths ---")
    widths = analysis['level_widths']
    max_w = max(widths.values())
    for d in sorted(widths.keys()):
        bar = "#" * int(widths[d] / max_w * 40)
        print(f"  Level {d:>3}: {widths[d]:>4} tasks  {bar}")

    print(f"\n--- Critical Path ---")
    for tid in analysis['critical_path_tasks'][:15]:
        t = dag.tasks[tid]
        mem_info = ""
        if t.memory:
            mem_info = (f" [smem={t.memory.shared_mem_bytes/1024:.0f}KB "
                        f"regs={t.memory.regs_per_thread}/thr "
                        f"occ={t.memory.occupancy_pct:.0f}%]")
        print(f"  {tid}: {t.task_type:<18} {t.runtime_us:>8.1f} us  {t.description[:35]}{mem_info}")
    if len(analysis['critical_path_tasks']) > 15:
        print(f"  ... +{len(analysis['critical_path_tasks']) - 15} more")


def print_memory_hierarchy_detail(dag: MoSAICDAG, gpu: GPUSpec):
    """Print detailed memory hierarchy analysis for sample tasks."""
    print(f"\n{'='*70}")
    print(f"MEMORY HIERARCHY DETAIL (sample tasks)")
    print(f"{'='*70}")

    # Show one matmul tile, one activation, one weight update
    samples = {}
    for t in dag.tasks.values():
        if t.task_type not in samples and t.memory:
            samples[t.task_type] = t
        if len(samples) >= 4:
            break

    for ttype, task in samples.items():
        m = task.memory
        print(f"\n  Task: {task.task_id} ({task.task_type}) - {task.description}")
        print(f"  {'─'*50}")
        print(f"  Registers:     {m.registers_bytes:>8} B  ({m.regs_per_thread} regs/thread x {m.threads_per_block} threads)")
        print(f"  Shared memory: {m.shared_mem_bytes:>8} B  ({m.shared_mem_per_block/1024:.1f} KB/block, "
              f"limit={gpu.shared_mem_per_block/1024:.0f} KB)")
        print(f"  L2 cache:      {m.l2_cache_bytes:>8} B  (L2 size={gpu.l2_cache_size/1024/1024:.0f} MB)")
        print(f"  Global read:   {m.global_mem_read_bytes:>8} B")
        print(f"  Global write:  {m.global_mem_write_bytes:>8} B")
        print(f"  Blocks:        {m.blocks_needed:>8}    (threads/block={m.threads_per_block})")
        print(f"  Occupancy:     {m.occupancy_pct:>7.1f}%")
        fits_shared = "YES" if m.shared_mem_per_block <= gpu.shared_mem_per_block else "NO - needs splitting!"
        fits_l2 = "YES" if (m.global_mem_read_bytes + m.global_mem_write_bytes) <= gpu.l2_cache_size else "partial"
        print(f"  Fits in shared mem?  {fits_shared}")
        print(f"  Fits in L2 cache?    {fits_l2}")


# ============================================================
# DOT VISUALIZATION
# ============================================================

def generate_dag_dot(dag: MoSAICDAG, analysis: dict, filename="mosaic_dag_v2"):
    depth_map = analysis["depth_map"]
    cp_set = set(analysis["critical_path_tasks"])

    colors = {
        "input": "#4CAF50", "matmul_tile": "#2196F3", "accumulate": "#FF9800",
        "activation": "#9C27B0", "loss": "#F44336", "grad_loss": "#F44336",
        "grad_activation": "#E91E63", "weight_update": "#00BCD4",
    }

    lines = [
        'digraph MoSAIC_DAG {',
        '  rankdir=TB;',
        '  node [shape=box, style="filled,rounded", fontsize=8, fontname="Helvetica"];',
        '  edge [color="#666666", arrowsize=0.5];',
        f'  label="MoSAIC DAG (tile={dag.tile_size}) | '
        f'{analysis["total_tasks"]} tasks | '
        f'CP={analysis["critical_path_us"]:.0f}us | '
        f'MaxPar={analysis["max_parallelism"]} | '
        f'Occ={analysis["avg_occupancy_pct"]:.0f}%";',
        '  labelloc=t; fontsize=12;', '',
    ]

    levels = defaultdict(list)
    for tid, d in depth_map.items():
        levels[d].append(tid)

    for d in sorted(levels.keys()):
        lines.append(f'  subgraph cluster_level_{d} {{ style=invis; rank=same;')
        for tid in levels[d]:
            task = dag.tasks[tid]
            color = colors.get(task.task_type, "#9E9E9E")
            border = "red" if tid in cp_set else "#333333"
            pw = "3" if tid in cp_set else "1"
            mem_str = ""
            if task.memory:
                mem_str = f"\\nsmem={task.memory.shared_mem_bytes//1024}KB occ={task.memory.occupancy_pct:.0f}%"
            label = f"{tid}\\n{task.task_type}\\n{task.runtime_us:.1f}us{mem_str}"
            lines.append(f'    {tid} [label="{label}", fillcolor="{color}", '
                         f'color="{border}", penwidth={pw}, fontcolor="white"];')
        lines.append('  }')

    for tid, task in dag.tasks.items():
        for dep in task.dependencies:
            style = "bold" if (dep in cp_set and tid in cp_set) else "solid"
            col = "red" if (dep in cp_set and tid in cp_set) else "#666666"
            lines.append(f'  {dep} -> {tid} [style={style}, color="{col}"];')

    lines.append('}')

    dot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{filename}.dot")
    with open(dot_path, 'w') as f:
        f.write('\n'.join(lines))

    try:
        svg_path = dot_path.replace('.dot', '.svg')
        subprocess.run(['dot', '-Tsvg', dot_path, '-o', svg_path], check=True, timeout=30)
        print(f"  DAG rendered: {svg_path}")
    except Exception as e:
        print(f"  DOT file: {dot_path} (render failed: {e})")

    return dot_path


# ============================================================
# SCALING EXPERIMENT
# ============================================================

def run_scaling_experiment(device, gpu: GPUSpec):
    """Run the full experiment across 1X, 2X, 3X scales and multiple tile sizes."""

    scales = [
        ("1X", 64, 1024, 512, 10),
        ("2X", 128, 2048, 1024, 10),
        ("3X", 192, 3072, 1536, 10),
    ]

    tile_sizes = [32, 64, 128, 256]

    all_results = []

    for scale_name, B, D_in, D_hid, D_out in scales:
        print(f"\n{'='*70}")
        print(f"SCALE {scale_name}: batch={B}, dims={D_in}->{D_hid}->{D_out}")
        print(f"{'='*70}")

        # --- Baseline PyTorch training ---
        model = SimpleMLP(D_in, D_hid, D_out).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        X = torch.randn(B, D_in, device=device)
        y = torch.randint(0, D_out, (B,), device=device)

        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        # Timed run with power monitoring
        pm = PowerMonitor()
        torch.cuda.synchronize()
        pm.start()
        t0 = time.perf_counter()

        for _ in range(10):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        baseline_ms = (time.perf_counter() - t0) * 1000 / 10  # per iteration
        pm.stop()
        power_stats = pm.get_stats()

        # Memory usage
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        print(f"\n  Baseline: {baseline_ms:.2f} ms/iter, loss={loss.item():.4f}")
        print(f"  Peak GPU memory: {peak_mem:.1f} MB")
        print(f"  Power: avg={power_stats['avg_power_w']}W, peak={power_stats['peak_power_w']}W, "
              f"energy={power_stats['energy_mj']}mJ")

        # --- Tiling analysis ---
        print(f"\n  --- Tiling Analysis (Layer 1: ({B}x{D_in})@({D_in}x{D_hid})) ---")

        for ts in tile_sizes:
            print(f"\n  Tile size {ts}x{ts}:")
            tiles_l1 = compute_tiles(B, D_hid, D_in, ts)
            tiles_l2 = compute_tiles(B, D_out, D_hid, ts)
            n_boundary_l1 = sum(1 for t in tiles_l1 if t.m < ts or t.n < ts or t.k < ts)

            print(f"    L1 tiles: {len(tiles_l1)} ({len(tiles_l1)-n_boundary_l1} full, {n_boundary_l1} boundary)")
            print(f"    L2 tiles: {len(tiles_l2)}")

            # Check shared memory feasibility
            max_smem = min(ts, B) * min(ts, D_in) * 4 + min(ts, D_in) * min(ts, D_hid) * 4
            fits = "YES" if max_smem <= gpu.shared_mem_per_block else "NO"
            print(f"    Shared mem/tile: {max_smem/1024:.1f} KB (fits in {gpu.shared_mem_per_block/1024:.0f}KB block? {fits})")

        # --- Build DAG for default tile size (128) and profile ---
        print(f"\n  --- DAG (tile=128) ---")
        dag = MoSAICDAG(B, D_in, D_hid, D_out, tile_size=128, gpu=gpu)
        for task in dag.tasks.values():
            profile_task(task, device)
        analysis = analyze_dag(dag)
        print_dag_summary(dag, analysis, gpu)

        # Power during tiled execution simulation
        pm2 = PowerMonitor()
        pm2.start()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Simulate tiled execution (run each tile op)
        for task in dag.tasks.values():
            if task.task_type == "matmul_tile" and task.tile_coords:
                tc = task.tile_coords
                A = torch.randn(tc["m"], tc["kk"], device=device)
                B_mat = torch.randn(tc["kk"], tc["n"], device=device)
                _ = A @ B_mat
            elif task.task_type in ("activation", "grad_activation"):
                sz = task.memory.global_mem_read_bytes // 4 if task.memory else 128
                _ = F.relu(torch.randn(sz, device=device))
            elif task.task_type in ("accumulate", "weight_update"):
                sz = task.memory.global_mem_read_bytes // 4 if task.memory else 128
                a = torch.randn(sz, device=device)
                b = torch.randn(sz, device=device)
                _ = a + b

        torch.cuda.synchronize()
        tiled_ms = (time.perf_counter() - t0) * 1000
        pm2.stop()
        tiled_power = pm2.get_stats()

        print(f"\n  --- Tiled Execution ---")
        print(f"  Serial tiled time: {tiled_ms:.2f} ms")
        print(f"  Power: avg={tiled_power['avg_power_w']}W, peak={tiled_power['peak_power_w']}W, "
              f"energy={tiled_power['energy_mj']}mJ")

        # Generate DAG visualization
        generate_dag_dot(dag, analysis, f"dag_{scale_name}")

        scale_result = {
            "scale": scale_name,
            "batch_size": B,
            "dims": f"{D_in}->{D_hid}->{D_out}",
            "baseline_ms_per_iter": round(baseline_ms, 3),
            "baseline_power": power_stats,
            "peak_memory_MB": round(peak_mem, 1),
            "dag_tasks": analysis["total_tasks"],
            "dag_depth": analysis["max_depth"] + 1,
            "dag_work_us": round(analysis["total_work_us"], 1),
            "dag_critical_path_us": round(analysis["critical_path_us"], 1),
            "dag_max_parallelism": analysis["max_parallelism"],
            "dag_avg_parallelism": round(analysis["avg_parallelism"], 2),
            "dag_avg_occupancy_pct": round(analysis["avg_occupancy_pct"], 1),
            "global_mem_read_KB": round(analysis["total_global_read_KB"], 1),
            "global_mem_write_KB": round(analysis["total_global_write_KB"], 1),
            "shared_mem_total_KB": round(analysis["total_shared_mem_KB"], 1),
            "register_total_KB": round(analysis["total_register_KB"], 1),
            "tiled_serial_ms": round(tiled_ms, 3),
            "tiled_power": tiled_power,
            "type_counts": analysis["type_counts"],
        }
        all_results.append(scale_result)

    return all_results


# ============================================================
# TILE SIZE COMPARISON
# ============================================================

def tile_size_comparison(device, gpu: GPUSpec):
    """Compare different tile sizes on the base problem."""
    print(f"\n{'='*70}")
    print(f"TILE SIZE COMPARISON (base: 64x1024 @ 1024x512)")
    print(f"{'='*70}")

    tile_sizes = [32, 64, 128, 256]
    results = []

    for ts in tile_sizes:
        dag = MoSAICDAG(64, 1024, 512, 10, tile_size=ts, gpu=gpu)
        for task in dag.tasks.values():
            profile_task(task, device)
        analysis = analyze_dag(dag)

        # Shared memory violations
        violations = sum(1 for t in dag.tasks.values()
                         if t.memory and t.memory.shared_mem_per_block > gpu.shared_mem_per_block)

        r = {
            "tile_size": ts,
            "total_tasks": analysis["total_tasks"],
            "depth": analysis["max_depth"] + 1,
            "critical_path_us": round(analysis["critical_path_us"], 1),
            "total_work_us": round(analysis["total_work_us"], 1),
            "max_parallelism": analysis["max_parallelism"],
            "avg_parallelism": round(analysis["avg_parallelism"], 2),
            "avg_occupancy": round(analysis["avg_occupancy_pct"], 1),
            "shared_mem_violations": violations,
            "global_data_movement_KB": round(analysis["total_global_read_KB"] + analysis["total_global_write_KB"], 1),
        }
        results.append(r)

        print(f"\n  Tile {ts}x{ts}:")
        print(f"    Tasks: {r['total_tasks']}, Depth: {r['depth']}")
        print(f"    CP: {r['critical_path_us']} us, Work: {r['total_work_us']} us")
        print(f"    Parallelism: max={r['max_parallelism']}, avg={r['avg_parallelism']}x")
        print(f"    Occupancy: {r['avg_occupancy']}%")
        print(f"    Data movement: {r['global_data_movement_KB']} KB")
        print(f"    Shared mem violations: {r['shared_mem_violations']}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device('cuda')
    gpu = detect_gpu_spec(device)
    gpu.print_summary()

    # Memory hierarchy detail for base case
    print(f"\n{'='*70}")
    print("STEP 3: Tiling Breakdown (base case, tile=128)")
    print(f"{'='*70}")
    for name, M, N, K in [("Fwd L1", 64, 512, 1024), ("Fwd L2", 64, 10, 512),
                           ("Bwd dW2", 512, 10, 64), ("Bwd dA1", 64, 512, 10),
                           ("Bwd dW1", 1024, 512, 64)]:
        print(f"\n  {name}:")
        describe_tiling(M, N, K, 128)

    # Build base DAG with memory hierarchy
    print(f"\n{'='*70}")
    print("STEP 4: Architecture-Aware DAG (base case)")
    print(f"{'='*70}")
    dag = MoSAICDAG(64, 1024, 512, 10, tile_size=128, gpu=gpu)
    for task in dag.tasks.values():
        profile_task(task, device)
    analysis = analyze_dag(dag)
    print_dag_summary(dag, analysis, gpu)
    print_memory_hierarchy_detail(dag, gpu)
    generate_dag_dot(dag, analysis, "dag_base")

    # Tile size comparison
    tile_results = tile_size_comparison(device, gpu)

    # Scaling experiment
    scale_results = run_scaling_experiment(device, gpu)

    # ============================================================
    # FINAL COMPARISON TABLE
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*70}")

    print(f"\n  {'Scale':<6} {'Tasks':>6} {'Depth':>6} {'CP(us)':>8} {'Work(us)':>9} "
          f"{'MaxPar':>7} {'AvgPar':>7} {'Occ%':>5} {'GlobMem(KB)':>12} {'Power(W)':>9}")
    print(f"  {'-'*85}")
    for r in scale_results:
        glob_mem = r['global_mem_read_KB'] + r['global_mem_write_KB']
        print(f"  {r['scale']:<6} {r['dag_tasks']:>6} {r['dag_depth']:>6} "
              f"{r['dag_critical_path_us']:>8.1f} {r['dag_work_us']:>9.1f} "
              f"{r['dag_max_parallelism']:>7} {r['dag_avg_parallelism']:>7.2f} "
              f"{r['dag_avg_occupancy_pct']:>5.1f} {glob_mem:>12.1f} "
              f"{r['baseline_power']['avg_power_w']:>9.2f}")

    print(f"\n  {'Tile':>6} {'Tasks':>6} {'Depth':>6} {'CP(us)':>8} {'Work(us)':>9} "
          f"{'MaxPar':>7} {'AvgPar':>7} {'Occ%':>5} {'DataMov(KB)':>12} {'SmemViol':>9}")
    print(f"  {'-'*85}")
    for r in tile_results:
        print(f"  {r['tile_size']:>6} {r['total_tasks']:>6} {r['depth']:>6} "
              f"{r['critical_path_us']:>8.1f} {r['total_work_us']:>9.1f} "
              f"{r['max_parallelism']:>7} {r['avg_parallelism']:>7.2f} "
              f"{r['avg_occupancy']:>5.1f} {r['global_data_movement_KB']:>12.1f} "
              f"{r['shared_mem_violations']:>9}")

    # Save everything
    all_results = {
        "gpu_spec": {
            "name": gpu.name, "sm_count": gpu.sm_count,
            "shared_mem_per_block": gpu.shared_mem_per_block,
            "shared_mem_per_sm": gpu.shared_mem_per_sm,
            "regs_per_sm": gpu.regs_per_sm,
            "l2_cache_bytes": gpu.l2_cache_size,
            "global_mem_GB": round(gpu.global_mem_bytes / 1024**3, 1),
            "bandwidth_gbps": round(gpu.global_mem_bandwidth_gbps, 1),
        },
        "tile_size_comparison": tile_results,
        "scaling_results": scale_results,
    }

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dag_results_v2.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
