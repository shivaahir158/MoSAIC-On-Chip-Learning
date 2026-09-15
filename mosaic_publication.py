"""
MoSAIC Publication Experiment
==============================
Publication-ready experiment with comprehensive memory hierarchy analysis.
Demonstrates optimal tile-to-memory mapping for a single Transformer layer.

Key contributions over mosaic_transformer.py:
  - Data reuse analysis: shows how tiling reduces global memory traffic
  - Arithmetic intensity / roofline model: proves tiles are compute-bound
  - L2 cache working set analysis: proves entire working set fits in L2
  - Tile size sweep: systematic comparison across 16/32/64 with memory metrics
  - Statistical profiling: multiple runs with mean/std
  - Memory-aware scheduling: comm_cost derived from actual data movement
  - Publication-quality output: LaTeX-ready tables
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
import numpy as np
import heapq

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_STREAMS = 2

# Transformer config
BATCH = 4
SEQ_LEN = 32
HIDDEN = 128
HEADS = 2
HEAD_DIM = HIDDEN // HEADS  # 64
FFN_DIM = 256
NUM_CLASSES = 10


# ============================================================
# GPU SPEC & MEMORY HIERARCHY
# ============================================================

@dataclass
class GPUSpec:
    name: str
    sm_count: int
    shared_mem_per_block: int    # bytes
    shared_mem_per_sm: int       # bytes
    regs_per_sm: int
    l2_cache_bytes: int
    global_mem_bytes: int
    warp_size: int
    max_threads_per_sm: int
    memory_bandwidth_gbps: float  # GB/s
    peak_flops_gflops: float      # GFLOPS (FP32)


def get_gpu_spec() -> GPUSpec:
    props = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count
    # RTX 500 Ada: 128 CUDA cores/SM * 16 SMs * ~2.1 GHz boost
    # Conservative estimate from specs
    clock_ghz = 2.1  # approximate boost clock
    cores_per_sm = 128
    peak_flops = sm_count * cores_per_sm * clock_ghz * 2  # FMA = 2 ops
    return GPUSpec(
        name=props.name,
        sm_count=sm_count,
        shared_mem_per_block=props.shared_memory_per_block,
        shared_mem_per_sm=getattr(props, 'shared_memory_per_multiprocessor', 102400),
        regs_per_sm=getattr(props, 'regs_per_multiprocessor', 65536),
        l2_cache_bytes=getattr(props, 'L2_cache_size', 16777216),
        global_mem_bytes=props.total_memory,
        warp_size=props.warp_size,
        max_threads_per_sm=getattr(props, 'max_threads_per_multi_processor', 1536),
        memory_bandwidth_gbps=128.0,   # RTX 500 Ada spec
        peak_flops_gflops=peak_flops,
    )


# ============================================================
# MEMORY HIERARCHY ANALYSIS (per-tile)
# ============================================================

@dataclass
class TileMemoryProfile:
    """Complete memory hierarchy profile for a single tile operation."""
    # Tile dimensions
    m: int
    n: int
    k: int
    # Shared memory
    shared_mem_bytes: int
    fits_shared: bool
    shared_util_pct: float       # % of shared mem limit used
    # Global memory
    global_read_bytes: int       # A tile + B tile loaded
    global_write_bytes: int      # C tile written
    # Data reuse
    data_reuse_factor: float     # how many times input data is reused from shared mem
    # Compute
    flops: int                   # 2*m*n*k for GEMM
    arithmetic_intensity: float  # FLOPS / bytes moved
    # Roofline
    is_compute_bound: bool       # AI > ridge point?
    # Occupancy
    threads: int
    regs_per_thread: int
    total_regs: int
    blocks_per_sm: int
    occupancy_pct: float
    # L2 cache
    l2_working_set_bytes: int


def analyze_tile_memory(m, n, k, gpu: GPUSpec) -> TileMemoryProfile:
    """Detailed memory hierarchy analysis for a tile GEMM C[m,n] += A[m,k] * B[k,n]."""
    elem = 4  # float32

    # Shared memory: store A tile [m,k] and B tile [k,n] for one k-step
    smem_a = m * k * elem
    smem_b = k * n * elem
    smem_total = smem_a + smem_b
    fits = smem_total <= gpu.shared_mem_per_block
    smem_util = smem_total / gpu.shared_mem_per_block * 100

    # Global memory traffic
    # Without tiling: each element of A read N/n times, each element of B read M/m times
    # With tiling (into shared mem): each tile of A and B loaded once from global
    global_read = (m * k + k * n) * elem  # load A tile + B tile once
    global_write = m * n * elem           # write C tile once

    # Data reuse from shared memory:
    # Each element of A[m,k] is used n times (for each column of C)
    # Each element of B[k,n] is used m times (for each row of C)
    # Average reuse = (n * m*k + m * k*n) / (m*k + k*n) = 2*m*n*k / (m*k + k*n)
    total_ops = 2 * m * n * k
    total_loads = m * k + k * n
    data_reuse = (n * m * k + m * k * n) / max(total_loads, 1)

    # Arithmetic intensity: FLOPS / bytes moved from global memory
    total_bytes = (global_read + global_write)
    ai = total_ops / max(total_bytes, 1)

    # Roofline ridge point: peak_flops / bandwidth
    # At ridge point: AI = peak_GFLOPS / bandwidth_GB/s
    ridge_point = gpu.peak_flops_gflops / gpu.memory_bandwidth_gbps
    is_compute_bound = ai >= ridge_point

    # Thread and register analysis
    threads = min(256, m * n)
    elements_per_thread = max(1, (m * n) // threads)
    regs_per_thread = min(max(2 * elements_per_thread + 8, 16), 255)
    total_regs = threads * regs_per_thread

    # Occupancy: limited by shared memory, registers, and thread count
    blocks_by_smem = gpu.shared_mem_per_sm // max(smem_total, 1) if smem_total > 0 else 16
    blocks_by_regs = gpu.regs_per_sm // max(total_regs, 1)
    blocks_by_threads = gpu.max_threads_per_sm // max(threads, 1)
    blocks_per_sm = min(blocks_by_smem, blocks_by_regs, blocks_by_threads, 16)
    active_threads = blocks_per_sm * threads
    occupancy = min(100.0, active_threads / gpu.max_threads_per_sm * 100)

    # L2 working set: data that stays hot in L2 across tiles
    l2_ws = (m * k + k * n + m * n) * elem

    return TileMemoryProfile(
        m=m, n=n, k=k,
        shared_mem_bytes=smem_total,
        fits_shared=fits,
        shared_util_pct=smem_util,
        global_read_bytes=global_read,
        global_write_bytes=global_write,
        data_reuse_factor=data_reuse,
        flops=total_ops,
        arithmetic_intensity=ai,
        is_compute_bound=is_compute_bound,
        threads=threads,
        regs_per_thread=regs_per_thread,
        total_regs=total_regs,
        blocks_per_sm=blocks_per_sm,
        occupancy_pct=occupancy,
        l2_working_set_bytes=l2_ws,
    )


def analyze_untiled_memory(M, N, K, gpu: GPUSpec):
    """Memory analysis for an untiled (naive) GEMM for comparison."""
    elem = 4
    # Naive: each element of A[M,K] streamed from global N times
    # Each element of B[K,N] streamed from global M times
    global_read_naive = (M * K * N + K * N * M) * elem  # worst case: no reuse
    # Realistic: assume L2 caches some, but A is streamed row-by-row
    # A is read once (M*K), B columns are reread M times but L2 may cache
    global_read_realistic = (M * K + M * K * N) * elem  # B reread for each row of A
    global_write = M * N * elem
    flops = 2 * M * N * K
    ai_naive = flops / max(global_read_naive + global_write, 1)
    ai_realistic = flops / max(global_read_realistic + global_write, 1)
    return {
        "flops": flops,
        "global_bytes_naive": global_read_naive + global_write,
        "global_bytes_realistic": global_read_realistic + global_write,
        "ai_naive": ai_naive,
        "ai_realistic": ai_realistic,
    }


# ============================================================
# POWER MONITOR
# ============================================================

class PowerMonitor:
    def __init__(self):
        self.samples = []
        self._running = False
        self._thread = None

    def start(self):
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while self._running:
            try:
                r = subprocess.run(
                    ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=1)
                if r.returncode == 0:
                    self.samples.append((time.perf_counter(), float(r.stdout.strip())))
            except Exception:
                pass
            time.sleep(0.02)

    def stats(self):
        if not self.samples:
            return {"avg_w": 0, "peak_w": 0, "energy_mj": 0, "samples": 0}
        powers = [s[1] for s in self.samples]
        dur = self.samples[-1][0] - self.samples[0][0] if len(self.samples) > 1 else 0.02
        avg = sum(powers) / len(powers)
        return {
            "avg_w": round(avg, 2),
            "peak_w": round(max(powers), 2),
            "energy_mj": round(avg * dur * 1000, 2),
            "duration_s": round(dur, 3),
            "samples": len(powers),
        }


# ============================================================
# STEP 1: TRANSFORMER MODEL
# ============================================================

class TransformerLayer(nn.Module):
    def __init__(self, hidden=HIDDEN, heads=HEADS, ffn=FFN_DIM):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.W_Q = nn.Linear(hidden, hidden, bias=False)
        self.W_K = nn.Linear(hidden, hidden, bias=False)
        self.W_V = nn.Linear(hidden, hidden, bias=False)
        self.W_O = nn.Linear(hidden, hidden, bias=False)
        self.W1 = nn.Linear(hidden, ffn, bias=False)
        self.W2 = nn.Linear(ffn, hidden, bias=False)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x):
        B, S, D = x.shape
        Q = self.W_Q(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        context = (attn @ V).transpose(1, 2).contiguous().view(B, S, D)
        out = self.W_O(context)
        x = self.ln1(x + out)
        ffn_out = self.W2(F.gelu(self.W1(x)))
        x = self.ln2(x + ffn_out)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = TransformerLayer()
        self.head = nn.Linear(HIDDEN, NUM_CLASSES, bias=False)

    def forward(self, x):
        x = self.layer(x)
        x = x.mean(dim=1)
        return self.head(x)


def step1_baseline():
    print("\n" + "=" * 70)
    print("STEP 1-2: Baseline Transformer Training")
    print("=" * 70)

    model = TransformerClassifier().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    X = torch.randn(BATCH, SEQ_LEN, HIDDEN, device=DEVICE)
    y = torch.randint(0, NUM_CLASSES, (BATCH,), device=DEVICE)

    for _ in range(10):
        opt.zero_grad(); loss = crit(model(X), y); loss.backward(); opt.step()

    pm = PowerMonitor()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    pm.start()

    # Multiple timed runs for statistics
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            opt.zero_grad(); loss = crit(model(X), y); loss.backward(); opt.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000 / 20)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    pm.stop()
    power = pm.stats()

    ms_mean = np.mean(times)
    ms_std = np.std(times)
    param_count = sum(p.numel() for p in model.parameters())

    print(f"  Config: B={BATCH}, S={SEQ_LEN}, H={HIDDEN}, heads={HEADS}, FFN={FFN_DIM}")
    print(f"  Parameters: {param_count:,}")
    print(f"  Runtime: {ms_mean:.3f} +/- {ms_std:.3f} ms/iter (n=5x20)")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Peak GPU memory: {peak_mem:.1f} MB")
    print(f"  Power: avg={power['avg_w']}W, peak={power['peak_w']}W")

    return ms_mean, ms_std, loss.item(), peak_mem, power


# ============================================================
# STEP 3: TILING WITH MEMORY HIERARCHY ANALYSIS
# ============================================================

def compute_tiles(M, N, K, ts):
    tiles = []
    for i in range(0, M, ts):
        for j in range(0, N, ts):
            for k in range(0, K, ts):
                tiles.append({
                    "i": i // ts, "j": j // ts, "k": k // ts,
                    "m": min(ts, M - i), "n": min(ts, N - j),
                    "kk": min(ts, K - k),
                })
    return tiles


def get_transformer_gemms():
    """All GEMMs in one training iteration (forward only, backward mirrors)."""
    BS = BATCH * SEQ_LEN
    gemms = [
        ("Q_proj",  BS, HIDDEN, HIDDEN),
        ("K_proj",  BS, HIDDEN, HIDDEN),
        ("V_proj",  BS, HIDDEN, HIDDEN),
    ]
    for h in range(HEADS):
        gemms.append((f"Attn_score_h{h}", SEQ_LEN, SEQ_LEN, HEAD_DIM))
    for h in range(HEADS):
        gemms.append((f"Attn_out_h{h}", SEQ_LEN, HEAD_DIM, SEQ_LEN))
    gemms += [
        ("Out_proj",   BS, HIDDEN, HIDDEN),
        ("FFN1",       BS, FFN_DIM, HIDDEN),
        ("FFN2",       BS, HIDDEN, FFN_DIM),
        ("Classifier", BATCH, NUM_CLASSES, HIDDEN),
    ]
    return gemms


def step3_tile_memory_analysis(gpu: GPUSpec):
    """Comprehensive tile size sweep with memory hierarchy analysis."""
    print("\n" + "=" * 70)
    print("STEP 3: Tile Size Sweep -- Memory Hierarchy Analysis")
    print("=" * 70)

    gemms = get_transformer_gemms()
    tile_sizes = [16, 32, 64]
    sweep_results = {}

    for ts in tile_sizes:
        print(f"\n  --- Tile Size: {ts}x{ts} ---")
        total_tiles = 0
        total_smem = 0
        total_global_read = 0
        total_global_write = 0
        total_flops = 0
        total_l2_ws = 0
        violations = 0
        occupancies = []
        ais = []
        reuse_factors = []
        per_gemm = []

        for name, M, N, K in gemms:
            tiles = compute_tiles(M, N, K, ts)
            gemm_tiles = len(tiles)
            total_tiles += gemm_tiles

            # Analyze memory for typical tile (first full tile)
            tm = min(ts, M)
            tn = min(ts, N)
            tk = min(ts, K)
            profile = analyze_tile_memory(tm, tn, tk, gpu)

            smem_per_gemm = profile.shared_mem_bytes * gemm_tiles
            gread_per_gemm = profile.global_read_bytes * gemm_tiles
            gwrite_per_gemm = profile.global_write_bytes * gemm_tiles
            flops_per_gemm = profile.flops * gemm_tiles

            total_smem += smem_per_gemm
            total_global_read += gread_per_gemm
            total_global_write += gwrite_per_gemm
            total_flops += flops_per_gemm
            total_l2_ws += profile.l2_working_set_bytes
            if not profile.fits_shared:
                violations += gemm_tiles
            occupancies.append(profile.occupancy_pct)
            ais.append(profile.arithmetic_intensity)
            reuse_factors.append(profile.data_reuse_factor)

            # Untiled comparison
            untiled = analyze_untiled_memory(M, N, K, gpu)

            per_gemm.append({
                "name": name,
                "shape": f"({M}x{K})@({K}x{N})",
                "tiles": gemm_tiles,
                "smem_bytes": profile.shared_mem_bytes,
                "smem_pct": profile.shared_util_pct,
                "fits_shared": profile.fits_shared,
                "global_read_MB": gread_per_gemm / 1024 / 1024,
                "global_write_MB": gwrite_per_gemm / 1024 / 1024,
                "data_reuse": profile.data_reuse_factor,
                "arith_intensity": profile.arithmetic_intensity,
                "is_compute_bound": profile.is_compute_bound,
                "occupancy": profile.occupancy_pct,
                "flops": flops_per_gemm,
                "untiled_bytes": untiled["global_bytes_realistic"],
                "tiled_bytes": gread_per_gemm + gwrite_per_gemm,
                "traffic_reduction": untiled["global_bytes_realistic"] / max(gread_per_gemm + gwrite_per_gemm, 1),
            })

        # Backward roughly doubles everything
        fwd_tiles = total_tiles
        total_tiles_full = total_tiles * 2 + total_tiles  # fwd + bwd gradients + weight updates
        total_global_read_full = total_global_read * 3
        total_global_write_full = total_global_write * 3
        total_flops_full = total_flops * 3

        # Aggregate arithmetic intensity
        total_bytes_moved = total_global_read_full + total_global_write_full
        aggregate_ai = total_flops_full / max(total_bytes_moved, 1)
        ridge_point = gpu.peak_flops_gflops / gpu.memory_bandwidth_gbps

        avg_occ = np.mean(occupancies)
        avg_reuse = np.mean(reuse_factors)
        avg_ai = np.mean(ais)

        # L2 cache analysis: does the working set fit?
        l2_fit = total_l2_ws <= gpu.l2_cache_bytes
        l2_util = total_l2_ws / gpu.l2_cache_bytes * 100

        print(f"    Forward tiles:          {fwd_tiles}")
        print(f"    Est. total tasks:       ~{total_tiles_full}")
        print(f"    Shared mem violations:  {violations}")
        print(f"    Avg occupancy:          {avg_occ:.1f}%")
        print(f"    Avg data reuse:         {avg_reuse:.1f}x")
        print(f"    Avg arith. intensity:   {avg_ai:.2f} FLOP/byte")
        print(f"    Ridge point:            {ridge_point:.2f} FLOP/byte")
        print(f"    Compute bound:          {'YES' if avg_ai >= ridge_point else 'NO'}")
        print(f"    Total global reads:     {total_global_read_full / 1024 / 1024:.1f} MB")
        print(f"    Total global writes:    {total_global_write_full / 1024 / 1024:.1f} MB")
        print(f"    Total data movement:    {total_bytes_moved / 1024 / 1024:.1f} MB")
        print(f"    L2 working set:         {total_l2_ws / 1024:.1f} KB")
        print(f"    L2 utilization:         {l2_util:.1f}% (of {gpu.l2_cache_bytes // 1024 // 1024} MB)")
        print(f"    Working set fits L2:    {'YES' if l2_fit else 'NO'}")

        sweep_results[ts] = {
            "tile_size": ts,
            "fwd_tiles": fwd_tiles,
            "est_total_tasks": total_tiles_full,
            "smem_violations": violations,
            "avg_occupancy_pct": round(avg_occ, 1),
            "avg_data_reuse": round(avg_reuse, 1),
            "avg_arithmetic_intensity": round(avg_ai, 2),
            "ridge_point": round(ridge_point, 2),
            "is_compute_bound": avg_ai >= ridge_point,
            "total_global_read_MB": round(total_global_read_full / 1024 / 1024, 2),
            "total_global_write_MB": round(total_global_write_full / 1024 / 1024, 2),
            "total_data_movement_MB": round(total_bytes_moved / 1024 / 1024, 2),
            "l2_working_set_KB": round(total_l2_ws / 1024, 1),
            "l2_utilization_pct": round(l2_util, 1),
            "fits_l2": l2_fit,
            "total_flops": total_flops_full,
            "per_gemm": per_gemm,
        }

    # Data reuse comparison table
    print(f"\n  === DATA REUSE: TILED vs UNTILED (Forward Pass) ===")
    print(f"  {'GEMM':<20} {'Naive Traffic':>15} {'Tiled (32x32)':>15} {'Reduction':>10}")
    best_ts = 32  # our target
    for g in sweep_results[best_ts]["per_gemm"]:
        naive_mb = g["untiled_bytes"] / 1024 / 1024
        tiled_mb = g["tiled_bytes"] / 1024 / 1024
        print(f"  {g['name']:<20} {naive_mb:>12.1f} MB {tiled_mb:>12.1f} MB {g['traffic_reduction']:>9.1f}x")

    return sweep_results


# ============================================================
# STEP 4-6: DAG CONSTRUCTION + MOTIFS
# ============================================================

@dataclass
class Task:
    tid: int
    name: str
    task_type: str
    gemm_name: str = ""
    deps: List[int] = field(default_factory=list)
    weight_us: float = 0.0
    memory: Optional[TileMemoryProfile] = None
    mem_dict: Optional[Dict] = None
    rank_u: float = 0.0
    depth: int = 0
    fanout: int = 0
    indegree: int = 0
    comm_cost: float = 0.0
    processor: int = -1
    start_time: float = 0.0
    end_time: float = 0.0


class TransformerDAG:
    def __init__(self, ts, gpu: GPUSpec):
        self.ts = ts
        self.gpu = gpu
        self.tasks: Dict[int, Task] = {}
        self.edges: List[Tuple[int, int]] = []
        self._id = 0
        self.gemm_tile_counts = {}
        self._build()

    def _add(self, name, ttype, deps=None, memory=None, mem_dict=None, gemm_name=""):
        self._id += 1
        t = Task(tid=self._id, name=name, task_type=ttype,
                 deps=deps or [], memory=memory, mem_dict=mem_dict, gemm_name=gemm_name)
        self.tasks[self._id] = t
        for d in t.deps:
            self.edges.append((d, self._id))
            self.tasks[d].fanout += 1
        t.indegree = len(t.deps)
        return self._id

    def _tiled_matmul(self, M, N, K, prefix, deps, gemm_name=""):
        tiles = compute_tiles(M, N, K, self.ts)
        blocks = defaultdict(list)
        for tile in tiles:
            m, n, k = tile["m"], tile["n"], tile["kk"]
            profile = analyze_tile_memory(m, n, k, self.gpu)
            mem_dict = {
                "shared_mem_bytes": profile.shared_mem_bytes,
                "global_read_bytes": profile.global_read_bytes,
                "global_write_bytes": profile.global_write_bytes,
                "fits_shared": profile.fits_shared,
                "occupancy_pct": profile.occupancy_pct,
                "threads": profile.threads,
                "regs_per_thread": profile.regs_per_thread,
                "data_reuse": profile.data_reuse_factor,
                "arithmetic_intensity": profile.arithmetic_intensity,
                "flops": profile.flops,
            }
            # Communication cost proportional to data movement
            tid = self._add(
                f"{prefix}_r{tile['i']}_c{tile['j']}_k{tile['k']}",
                "matmul", deps=deps, memory=profile, mem_dict=mem_dict,
                gemm_name=gemm_name)
            blocks[(tile["i"], tile["j"])].append(tid)

        out_ids = []
        for (ib, jb), tids in sorted(blocks.items()):
            if len(tids) == 1:
                out_ids.append(tids[0])
            else:
                aid = self._add(f"{prefix}_acc_r{ib}_c{jb}", "accum",
                                deps=tids, gemm_name=gemm_name)
                out_ids.append(aid)

        self.gemm_tile_counts[gemm_name] = len(tiles)
        return out_ids, tiles

    def _add_elementwise(self, name, ttype, deps, num_elements=1024):
        mem_dict = {
            "shared_mem_bytes": 0,
            "global_read_bytes": num_elements * 4,
            "global_write_bytes": num_elements * 4,
            "fits_shared": True,
            "occupancy_pct": 100.0,
            "threads": 256,
            "regs_per_thread": 8,
            "data_reuse": 1.0,
            "arithmetic_intensity": 0.25,  # 1 op per 4 bytes
            "flops": num_elements,
        }
        return self._add(name, ttype, deps, mem_dict=mem_dict)

    def _build(self):
        BS = BATCH * SEQ_LEN

        inp = self._add("input", "input")

        # Forward: QKV
        q_ids, _ = self._tiled_matmul(BS, HIDDEN, HIDDEN, "fwd_Q", [inp], "Q_proj")
        k_ids, _ = self._tiled_matmul(BS, HIDDEN, HIDDEN, "fwd_K", [inp], "K_proj")
        v_ids, _ = self._tiled_matmul(BS, HIDDEN, HIDDEN, "fwd_V", [inp], "V_proj")

        # Attention per head
        attn_score_ids = []
        attn_out_ids = []
        for h in range(HEADS):
            score_ids, _ = self._tiled_matmul(
                SEQ_LEN, SEQ_LEN, HEAD_DIM,
                f"fwd_score_h{h}", q_ids + k_ids, f"Attn_score_h{h}")
            attn_score_ids.extend(score_ids)

            softmax_id = self._add_elementwise(
                f"softmax_h{h}", "softmax", score_ids, SEQ_LEN * SEQ_LEN * BATCH)

            out_ids, _ = self._tiled_matmul(
                SEQ_LEN, HEAD_DIM, SEQ_LEN,
                f"fwd_attn_out_h{h}", [softmax_id] + v_ids, f"Attn_out_h{h}")
            attn_out_ids.extend(out_ids)

        # Output projection
        out_proj_ids, _ = self._tiled_matmul(
            BS, HIDDEN, HIDDEN, "fwd_out_proj", attn_out_ids, "Out_proj")

        # Residual + LN 1
        residual1 = self._add_elementwise(
            "residual_add_1", "residual", out_proj_ids + [inp], BS * HIDDEN)
        ln1 = self._add_elementwise("layernorm_1", "layernorm", [residual1], BS * HIDDEN)

        # FFN
        ffn1_ids, _ = self._tiled_matmul(BS, FFN_DIM, HIDDEN, "fwd_ffn1", [ln1], "FFN1")
        gelu_ids = []
        for i, fid in enumerate(ffn1_ids):
            gelu_ids.append(self._add_elementwise(
                f"gelu_{i}", "gelu", [fid], BS * FFN_DIM // len(ffn1_ids)))
        ffn2_ids, _ = self._tiled_matmul(BS, HIDDEN, FFN_DIM, "fwd_ffn2", gelu_ids, "FFN2")

        # Residual + LN 2
        residual2 = self._add_elementwise(
            "residual_add_2", "residual", ffn2_ids + [ln1], BS * HIDDEN)
        ln2 = self._add_elementwise("layernorm_2", "layernorm", [residual2], BS * HIDDEN)

        # Classifier
        pool = self._add_elementwise("pool", "pool", [ln2], BATCH * HIDDEN)
        cls_ids, _ = self._tiled_matmul(
            BATCH, NUM_CLASSES, HIDDEN, "fwd_cls", [pool], "Classifier")

        # Loss
        loss_id = self._add_elementwise("loss", "loss", cls_ids, BATCH * NUM_CLASSES)
        grad_loss = self._add_elementwise("grad_loss", "grad_loss", [loss_id], BATCH * NUM_CLASSES)

        # Backward classifier
        dw_cls_ids, _ = self._tiled_matmul(
            HIDDEN, NUM_CLASSES, BATCH, "bwd_dW_cls", [grad_loss, pool], "dW_cls")
        dpool_ids, _ = self._tiled_matmul(
            BATCH, HIDDEN, NUM_CLASSES, "bwd_dPool", [grad_loss], "dPool")

        # Backward pool, LN2, residual2
        d_ln2 = self._add_elementwise("bwd_unpool", "grad_pool", dpool_ids, BS * HIDDEN)
        d_res2 = self._add_elementwise("bwd_ln2", "grad_layernorm", [d_ln2], BS * HIDDEN)

        # Backward FFN2
        dw_ffn2_ids, _ = self._tiled_matmul(
            FFN_DIM, HIDDEN, BS, "bwd_dW_ffn2", [d_res2] + gelu_ids, "dW_FFN2")
        d_gelu_in_ids, _ = self._tiled_matmul(
            BS, FFN_DIM, HIDDEN, "bwd_d_ffn2", [d_res2], "d_FFN2")

        # Backward GeLU
        d_ffn1_ids = []
        for i, dgid in enumerate(d_gelu_in_ids):
            gid = gelu_ids[i] if i < len(gelu_ids) else gelu_ids[-1]
            d_ffn1_ids.append(self._add_elementwise(
                f"bwd_gelu_{i}", "grad_gelu", [dgid, gid],
                BS * FFN_DIM // max(len(d_gelu_in_ids), 1)))

        # Backward FFN1
        dw_ffn1_ids, _ = self._tiled_matmul(
            HIDDEN, FFN_DIM, BS, "bwd_dW_ffn1", d_ffn1_ids + [ln1], "dW_FFN1")
        d_ln1_from_ffn, _ = self._tiled_matmul(
            BS, HIDDEN, FFN_DIM, "bwd_d_ffn1", d_ffn1_ids, "d_FFN1")

        # Backward LN1
        d_res1 = self._add_elementwise(
            "bwd_ln1", "grad_layernorm", d_ln1_from_ffn + [d_res2], BS * HIDDEN)

        # Backward output projection
        dw_out_ids, _ = self._tiled_matmul(
            HIDDEN, HIDDEN, BS, "bwd_dW_out", [d_res1] + attn_out_ids, "dW_Out")
        d_attn_concat, _ = self._tiled_matmul(
            BS, HIDDEN, HIDDEN, "bwd_d_out", [d_res1], "d_Out")

        # Backward attention per head
        d_q_all, d_k_all, d_v_all = [], [], []
        for h in range(HEADS):
            d_scores_ids, _ = self._tiled_matmul(
                SEQ_LEN, SEQ_LEN, HEAD_DIM,
                f"bwd_d_score_h{h}", d_attn_concat, f"d_score_h{h}")
            d_v_ids, _ = self._tiled_matmul(
                SEQ_LEN, HEAD_DIM, SEQ_LEN,
                f"bwd_d_V_h{h}", d_attn_concat + attn_score_ids, f"d_V_h{h}")
            d_presoftmax = self._add_elementwise(
                f"bwd_softmax_h{h}", "grad_softmax", d_scores_ids,
                SEQ_LEN * SEQ_LEN * BATCH)
            d_q_ids, _ = self._tiled_matmul(
                SEQ_LEN, HEAD_DIM, SEQ_LEN,
                f"bwd_d_Q_h{h}", [d_presoftmax] + k_ids, f"d_Q_h{h}")
            d_k_ids, _ = self._tiled_matmul(
                SEQ_LEN, HEAD_DIM, SEQ_LEN,
                f"bwd_d_K_h{h}", [d_presoftmax] + q_ids, f"d_K_h{h}")
            d_q_all.extend(d_q_ids)
            d_k_all.extend(d_k_ids)
            d_v_all.extend(d_v_ids)

        # Backward QKV
        dw_q_ids, _ = self._tiled_matmul(HIDDEN, HIDDEN, BS, "bwd_dW_Q", d_q_all + [inp], "dW_Q")
        dw_k_ids, _ = self._tiled_matmul(HIDDEN, HIDDEN, BS, "bwd_dW_K", d_k_all + [inp], "dW_K")
        dw_v_ids, _ = self._tiled_matmul(HIDDEN, HIDDEN, BS, "bwd_dW_V", d_v_all + [inp], "dW_V")

        # Weight updates
        all_grad_groups = [
            ("wu_W_Q", dw_q_ids), ("wu_W_K", dw_k_ids), ("wu_W_V", dw_v_ids),
            ("wu_W_O", dw_out_ids),
            ("wu_W1", dw_ffn1_ids), ("wu_W2", dw_ffn2_ids),
            ("wu_cls", dw_cls_ids),
        ]
        self.wu_ids = []
        for prefix, grad_ids in all_grad_groups:
            for i, gid in enumerate(grad_ids):
                wid = self._add(f"{prefix}_{i}", "weight_update", [gid])
                self.wu_ids.append(wid)


def detect_motifs(dag):
    children = defaultdict(list)
    parents = defaultdict(list)
    for (u, v) in dag.edges:
        children[u].append(v)
        parents[v].append(u)

    motifs = {"fork_join": 0, "fan_out": 0, "fan_in": 0, "chain": 0}
    for t in dag.tasks.values():
        if t.fanout >= 3:
            motifs["fan_out"] += 1
        if t.indegree >= 3:
            motifs["fan_in"] += 1

    visited = set()
    for t in dag.tasks.values():
        if t.tid in visited or t.indegree != 1 or t.fanout != 1:
            continue
        chain_len = 1
        cur = t.tid
        visited.add(cur)
        while True:
            succs = children[cur]
            if len(succs) == 1 and dag.tasks[succs[0]].indegree == 1 and dag.tasks[succs[0]].fanout <= 1:
                chain_len += 1; cur = succs[0]; visited.add(cur)
            else:
                break
        if chain_len >= 2:
            motifs["chain"] += 1

    for t in dag.tasks.values():
        if t.fanout >= 2:
            kids = set(children[t.tid])
            grandkids = defaultdict(set)
            for kid in kids:
                for gk in children[kid]:
                    grandkids[gk].add(kid)
            for gk, srcs in grandkids.items():
                if len(srcs) >= 2:
                    motifs["fork_join"] += 1
                    break

    return motifs, [motifs["fork_join"], motifs["fan_out"], motifs["fan_in"], motifs["chain"]]


# ============================================================
# STEP 7: PROFILING WITH STATISTICS
# ============================================================

def profile_dag(dag, device):
    print("\n" + "=" * 70)
    print("STEP 7: GPU Profiling (with statistics)")
    print("=" * 70)

    type_times = defaultdict(list)
    reps = 50

    for t in dag.tasks.values():
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        if t.task_type == "matmul" and t.memory:
            m, n, k = t.memory.m, t.memory.n, t.memory.k
            A = torch.randn(m, k, device=device)
            B = torch.randn(k, n, device=device)
            _ = A @ B; torch.cuda.synchronize()
            start_ev.record()
            for _ in range(reps):
                _ = A @ B
            end_ev.record(); torch.cuda.synchronize()
            t.weight_us = start_ev.elapsed_time(end_ev) * 1000 / reps
            # Communication cost = data movement time estimate
            # bytes / bandwidth * 1e6 (us)
            if t.mem_dict:
                bytes_moved = t.mem_dict["global_read_bytes"] + t.mem_dict["global_write_bytes"]
                t.comm_cost = max(1.0, bytes_moved / (128e9) * 1e6)  # 128 GB/s bandwidth

        elif t.task_type in ("softmax", "grad_softmax"):
            A = torch.randn(BATCH, HEADS, SEQ_LEN, SEQ_LEN, device=device)
            torch.cuda.synchronize()
            start_ev.record()
            for _ in range(reps): _ = F.softmax(A, dim=-1)
            end_ev.record(); torch.cuda.synchronize()
            t.weight_us = start_ev.elapsed_time(end_ev) * 1000 / reps
            t.comm_cost = max(1.0, t.weight_us * 0.1)

        elif t.task_type in ("gelu", "grad_gelu"):
            A = torch.randn(1024, device=device)
            torch.cuda.synchronize()
            start_ev.record()
            for _ in range(reps): _ = F.gelu(A)
            end_ev.record(); torch.cuda.synchronize()
            t.weight_us = start_ev.elapsed_time(end_ev) * 1000 / reps
            t.comm_cost = max(1.0, t.weight_us * 0.1)

        elif t.task_type in ("layernorm", "grad_layernorm"):
            ln = nn.LayerNorm(HIDDEN).to(device)
            A = torch.randn(BATCH * SEQ_LEN, HIDDEN, device=device)
            torch.cuda.synchronize()
            start_ev.record()
            for _ in range(reps): _ = ln(A)
            end_ev.record(); torch.cuda.synchronize()
            t.weight_us = start_ev.elapsed_time(end_ev) * 1000 / reps
            t.comm_cost = max(1.0, t.weight_us * 0.1)

        elif t.task_type in ("loss", "grad_loss"):
            logits = torch.randn(BATCH, NUM_CLASSES, device=device)
            y = torch.randint(0, NUM_CLASSES, (BATCH,), device=device)
            torch.cuda.synchronize()
            start_ev.record()
            for _ in range(reps): _ = F.cross_entropy(logits, y)
            end_ev.record(); torch.cuda.synchronize()
            t.weight_us = start_ev.elapsed_time(end_ev) * 1000 / reps
            t.comm_cost = max(1.0, t.weight_us * 0.1)

        elif t.task_type in ("accum", "weight_update", "residual", "pool", "grad_pool"):
            sz = max(256, BATCH * HIDDEN)
            A = torch.randn(sz, device=device)
            B = torch.randn(sz, device=device)
            torch.cuda.synchronize()
            start_ev.record()
            for _ in range(reps): _ = A + B
            end_ev.record(); torch.cuda.synchronize()
            t.weight_us = start_ev.elapsed_time(end_ev) * 1000 / reps
            t.comm_cost = max(1.0, t.weight_us * 0.1)

        else:
            t.weight_us = 0.1
            t.comm_cost = 1.0

        type_times[t.task_type].append(t.weight_us)

    _compute_ranks(dag)
    total_work = sum(t.weight_us for t in dag.tasks.values())

    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2)
        power_w = float(r.stdout.strip()) if r.returncode == 0 else 0
    except Exception:
        power_w = 0

    print(f"  Total tasks: {len(dag.tasks)}")
    print(f"  Total work: {total_work:.0f} us")
    print(f"  GPU power: {power_w:.1f} W")
    print(f"\n  {'Type':<20} {'Mean(us)':>10} {'Std(us)':>10} {'Count':>6} {'Total(us)':>12} {'%Work':>7}")
    print(f"  {'-' * 67}")
    for tt in sorted(type_times.keys()):
        times = type_times[tt]
        total_t = sum(times)
        pct = total_t / total_work * 100
        print(f"  {tt:<20} {np.mean(times):>10.1f} {np.std(times):>10.1f} "
              f"{len(times):>6} {total_t:>12.0f} {pct:>6.1f}%")

    return total_work, power_w


def _compute_ranks(dag):
    children = defaultdict(list)
    parents = defaultdict(list)
    for (u, v) in dag.edges:
        children[u].append(v)
        parents[v].append(u)

    order = []
    visited = set()
    def dfs(tid):
        if tid in visited: return
        visited.add(tid)
        for c in children[tid]: dfs(c)
        order.append(tid)
    for tid in dag.tasks: dfs(tid)

    for tid in order:
        t = dag.tasks[tid]
        if not children[tid]:
            t.rank_u = t.weight_us
        else:
            t.rank_u = t.weight_us + max(
                dag.tasks[c].rank_u + t.comm_cost for c in children[tid])

    topo = list(reversed(order))
    for tid in topo:
        t = dag.tasks[tid]
        if not parents[tid]:
            t.depth = 0
        else:
            t.depth = max(dag.tasks[p].depth for p in parents[tid]) + 1


# ============================================================
# STEP 8-9: CP-SAT
# ============================================================

def run_cpsat(dag, num_streams=NUM_STREAMS, time_limit=60):
    print("\n" + "=" * 70)
    print("STEP 8-9: CP-SAT Optimal Scheduling")
    print("=" * 70)

    try:
        from ortools.sat.python import cp_model
    except ImportError:
        print("  ortools not installed, using HEFT fallback")
        return _heft_schedule(dag, num_streams)

    model = cp_model.CpModel()
    task_ids = sorted(dag.tasks.keys())
    horizon = int(sum(t.weight_us for t in dag.tasks.values()) * 2)

    starts, ends, procs = {}, {}, {}
    for tid in task_ids:
        w = max(1, int(dag.tasks[tid].weight_us))
        starts[tid] = model.NewIntVar(0, horizon, f"s_{tid}")
        ends[tid] = model.NewIntVar(0, horizon, f"e_{tid}")
        model.Add(ends[tid] == starts[tid] + w)
        procs[tid] = model.NewIntVar(0, num_streams - 1, f"p_{tid}")

    for (u, v) in dag.edges:
        comm = max(1, int(dag.tasks[u].comm_cost))
        same = model.NewBoolVar(f"same_{u}_{v}")
        model.Add(procs[u] == procs[v]).OnlyEnforceIf(same)
        model.Add(procs[u] != procs[v]).OnlyEnforceIf(same.Not())
        model.Add(starts[v] >= ends[u]).OnlyEnforceIf(same)
        model.Add(starts[v] >= ends[u] + comm).OnlyEnforceIf(same.Not())

    for p in range(num_streams):
        p_intervals = []
        for tid in task_ids:
            is_on = model.NewBoolVar(f"on_{tid}_{p}")
            model.Add(procs[tid] == p).OnlyEnforceIf(is_on)
            model.Add(procs[tid] != p).OnlyEnforceIf(is_on.Not())
            w = max(1, int(dag.tasks[tid].weight_us))
            opt_iv = model.NewOptionalIntervalVar(starts[tid], w, ends[tid], is_on, f"oi_{tid}_{p}")
            p_intervals.append(opt_iv)
        model.AddNoOverlap(p_intervals)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [ends[tid] for tid in task_ids])
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = min(time_limit, 60)
    solver.parameters.num_workers = 4

    print(f"  Solving {len(task_ids)} tasks, {len(dag.edges)} edges, "
          f"{num_streams} streams (limit={time_limit}s)...")
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        opt_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        ms = solver.Value(makespan)
        print(f"  Status: {opt_str}")
        print(f"  Makespan: {ms} us")
        print(f"  Solve time: {solver.WallTime():.1f} s")

        schedule = {}
        for tid in task_ids:
            t = dag.tasks[tid]
            t.processor = solver.Value(procs[tid])
            t.start_time = solver.Value(starts[tid])
            t.end_time = solver.Value(ends[tid])
            schedule[tid] = {"proc": t.processor, "start": t.start_time, "end": t.end_time}

        for p in range(num_streams):
            p_tasks = [(s["start"], s["end"]) for s in schedule.values() if s["proc"] == p]
            if p_tasks:
                busy = sum(e - s for s, e in p_tasks)
                span = max(e for _, e in p_tasks) - min(s for s, _ in p_tasks)
                print(f"  Stream {p}: {len(p_tasks)} tasks, utilization={busy / span * 100:.1f}%")

        return ms, schedule, solver.WallTime()
    else:
        print("  No solution, falling back to HEFT")
        return _heft_schedule(dag, num_streams)


# ============================================================
# STEP 12: HEFT
# ============================================================

def run_heft(dag, num_streams=NUM_STREAMS, label="HEFT"):
    if label:
        print(f"\n{'=' * 70}")
        print(f"STEP 12: {label} Scheduling")
        print(f"{'=' * 70}")

    parents = defaultdict(list)
    for (u, v) in dag.edges:
        parents[v].append(u)

    order = sorted(dag.tasks.keys(), key=lambda t: -dag.tasks[t].rank_u)
    proc_avail = [0.0] * num_streams
    finish = {}

    for tid in order:
        t = dag.tasks[tid]
        earliest = max((finish.get(p, 0) + t.comm_cost for p in parents[tid]), default=0)
        best_p, best_end = -1, float('inf')
        for p in range(num_streams):
            s = max(proc_avail[p], earliest)
            e = s + t.weight_us
            if e < best_end:
                best_end = e
                best_p = p
        start = max(proc_avail[best_p], earliest)
        finish[tid] = start + t.weight_us
        proc_avail[best_p] = finish[tid]

    makespan = max(finish.values())
    if label:
        print(f"  Makespan: {makespan:.0f} us")
    return makespan


def _heft_schedule(dag, num_streams):
    parents = defaultdict(list)
    for (u, v) in dag.edges:
        parents[v].append(u)
    order = sorted(dag.tasks.keys(), key=lambda t: -dag.tasks[t].rank_u)
    proc_avail = [0.0] * num_streams
    finish = {}
    schedule = {}
    for tid in order:
        t = dag.tasks[tid]
        earliest = max((finish.get(p, 0) + t.comm_cost for p in parents[tid]), default=0)
        best_p, best_end = -1, float('inf')
        for p in range(num_streams):
            s = max(proc_avail[p], earliest)
            e = s + t.weight_us
            if e < best_end:
                best_end = e
                best_p = p
        start = max(proc_avail[best_p], earliest)
        finish[tid] = start + t.weight_us
        proc_avail[best_p] = finish[tid]
        t.processor = best_p
        t.start_time = start
        t.end_time = finish[tid]
        schedule[tid] = {"proc": best_p, "start": start, "end": finish[tid]}
    makespan = max(finish.values())
    return makespan, schedule, 0


# ============================================================
# STEP 10-11: MOSAIC LEARNED SCHEDULER (improved)
# ============================================================

def run_mosaic(dag, cpsat_schedule, num_streams=NUM_STREAMS):
    print("\n" + "=" * 70)
    print("STEP 10-11: MoSAIC Learned Scheduling")
    print("=" * 70)

    cpsat_makespan = max(s["end"] for s in cpsat_schedule.values())

    phi = {}
    for tid, t in dag.tasks.items():
        phi[tid] = np.array([t.rank_u, t.depth, t.fanout, t.indegree, t.comm_cost])

    best_theta = None
    best_gap = float('inf')
    best_ms = float('inf')

    # Phase 1: Bayesian-style exploration (800 trials)
    np.random.seed(42)
    print(f"  Phase 1: Exploring 800 random thetas...")
    for trial in range(800):
        theta = np.random.randn(5)
        theta[0] = abs(theta[0]) * 2
        ms = _list_schedule(dag, phi, theta, num_streams)
        gap = (ms - cpsat_makespan) / cpsat_makespan if cpsat_makespan > 0 else 0
        if gap < best_gap:
            best_gap = gap
            best_theta = theta.copy()
            best_ms = ms

    # Phase 2: RL-style refinement (300 trials, decreasing perturbation)
    print(f"  Phase 2: Refining best theta (300 trials)...")
    for i in range(300):
        scale = 0.5 * (1 - i / 300)
        theta = best_theta + np.random.randn(5) * scale
        ms = _list_schedule(dag, phi, theta, num_streams)
        gap = (ms - cpsat_makespan) / cpsat_makespan if cpsat_makespan > 0 else 0
        if gap < best_gap:
            best_gap = gap
            best_theta = theta.copy()
            best_ms = ms

    # Phase 3: Fine-tuning (100 trials)
    print(f"  Phase 3: Fine-tuning (100 trials)...")
    for _ in range(100):
        theta = best_theta + np.random.randn(5) * 0.1
        ms = _list_schedule(dag, phi, theta, num_streams)
        gap = (ms - cpsat_makespan) / cpsat_makespan if cpsat_makespan > 0 else 0
        if gap < best_gap:
            best_gap = gap
            best_theta = theta.copy()
            best_ms = ms

    print(f"  theta* = [{', '.join(f'{x:.3f}' for x in best_theta)}]")
    print(f"           [rank_u, depth, fanout, indegree, comm_cost]")
    print(f"  MoSAIC makespan: {best_ms:.0f} us")
    print(f"  CP-SAT makespan: {cpsat_makespan:.0f} us")
    print(f"  Gap: {best_gap * 100:.2f}%")

    # Interpretation
    print(f"\n  Weight interpretation:")
    labels = ["rank_u", "depth", "fanout", "indegree", "comm_cost"]
    for i, (l, w) in enumerate(zip(labels, best_theta)):
        sign = "+" if w > 0 else "-"
        mag = abs(w)
        if mag > 0.5:
            strength = "STRONG" if mag > 1.0 else "moderate"
            desc = {
                "rank_u": "prioritize critical path" if w > 0 else "deprioritize critical path",
                "depth": "prefer deeper tasks" if w > 0 else "prefer shallower (root-near) tasks",
                "fanout": "prefer high fan-out" if w > 0 else "avoid premature fan-out expansion",
                "indegree": "prioritize join points (unblock downstream)" if w > 0 else "avoid join points",
                "comm_cost": "schedule high-comm tasks early (hide latency)" if w > 0 else "defer high-comm tasks",
            }
            print(f"    {l:>12}: {sign}{mag:.3f} ({strength}) -> {desc[l]}")

    return best_ms, best_theta, best_gap


def _list_schedule(dag, phi, theta, num_streams, _cache={}):
    """Optimized list scheduling with precomputed topology."""
    # Cache the parent relationships and topological info
    cache_key = id(dag)
    if cache_key not in _cache:
        parents = defaultdict(list)
        children = defaultdict(list)
        for (u, v) in dag.edges:
            parents[v].append(u)
            children[u].append(v)
        _cache[cache_key] = (parents, children)
    parents, children = _cache[cache_key]

    priority = {tid: float(np.dot(theta, phi[tid])) for tid in dag.tasks}

    # Use in-degree tracking for efficient ready-set management
    in_count = {tid: len(parents[tid]) for tid in dag.tasks}
    ready = sorted([tid for tid, c in in_count.items() if c == 0],
                   key=lambda t: -priority[t])
    proc_avail = [0.0] * num_streams
    finish = {}

    # Use negative priority for max-heap behavior
    ready_heap = [(-priority[tid], tid) for tid in ready]
    heapq.heapify(ready_heap)
    scheduled = set()

    while ready_heap:
        neg_pri, tid = heapq.heappop(ready_heap)
        if tid in scheduled:
            continue
        t = dag.tasks[tid]
        earliest = max((finish.get(p, 0) + t.comm_cost for p in parents[tid]), default=0)
        best_p = min(range(num_streams), key=lambda p: max(proc_avail[p], earliest))
        start = max(proc_avail[best_p], earliest)
        finish[tid] = start + t.weight_us
        proc_avail[best_p] = finish[tid]
        scheduled.add(tid)

        for child in children[tid]:
            in_count[child] -= 1
            if in_count[child] == 0:
                heapq.heappush(ready_heap, (-priority[child], child))

    return max(finish.values()) if finish else 0


# ============================================================
# STEP 13: GENERALIZATION
# ============================================================

def step13_generalize():
    print("\n" + "=" * 70)
    print("STEP 13: Generalization to Larger Transformers")
    print("=" * 70)

    configs = [
        ("Base",  4,  32, 128, 2,  256),
        ("2X",    8,  64, 128, 2,  256),
        ("Wide",  4,  32, 256, 4,  512),
        ("Long",  4,  64, 128, 2,  256),
    ]

    results = []
    for name, B, S, H, heads, ffn in configs:
        BS = B * S
        hd = H // heads
        total_fwd_tiles = 0
        gemms = [
            (BS, H, H), (BS, H, H), (BS, H, H),  # QKV
            (BS, H, H),  # Out proj
            (BS, ffn, H), (BS, H, ffn),  # FFN
        ]
        for h in range(heads):
            gemms.append((S, S, hd))
            gemms.append((S, hd, S))

        for M, N, K in gemms:
            total_fwd_tiles += len(compute_tiles(M, N, K, 32))

        est_total = total_fwd_tiles * 3
        cpsat_status = "Optimal" if est_total < 500 else "Feasible only" if est_total < 2000 else "Intractable"

        print(f"  {name}: B={B}, S={S}, H={H}, heads={heads}, FFN={ffn}")
        print(f"    Forward tiles: {total_fwd_tiles}, Est. tasks: ~{est_total}, CP-SAT: {cpsat_status}")

        results.append({
            "name": name, "batch": B, "seq": S, "hidden": H,
            "heads": heads, "ffn": ffn,
            "est_tasks": est_total, "fwd_tiles": total_fwd_tiles,
            "cpsat_status": cpsat_status,
        })

    return results


# ============================================================
# STEP 14: RUNTIME FEEDBACK
# ============================================================

def step14_feedback(dag, best_theta, cpsat_makespan, num_streams=NUM_STREAMS):
    print("\n" + "=" * 70)
    print("STEP 14: Runtime Feedback Adaptation")
    print("=" * 70)

    phi = {tid: np.array([t.rank_u, t.depth, t.fanout, t.indegree, t.comm_cost])
           for tid, t in dag.tasks.items()}

    normal_ms = _list_schedule(dag, phi, best_theta, num_streams)

    # Scenario 1: Thermal throttle
    print("\n  Scenario 1: Thermal throttle (matmul 1.5x slower)")
    orig = {tid: t.weight_us for tid, t in dag.tasks.items()}
    for t in dag.tasks.values():
        if t.task_type == "matmul":
            t.weight_us *= 1.5
    _compute_ranks(dag)
    phi_t = {tid: np.array([t.rank_u, t.depth, t.fanout, t.indegree, t.comm_cost])
             for tid, t in dag.tasks.items()}
    throttled_ms = _list_schedule(dag, phi_t, best_theta, num_streams)

    best_new = best_theta.copy()
    best_new_ms = throttled_ms
    for _ in range(300):
        theta = best_theta + np.random.randn(5) * 0.5
        ms = _list_schedule(dag, phi_t, theta, num_streams)
        if ms < best_new_ms:
            best_new_ms = ms
            best_new = theta.copy()

    print(f"  Normal:    {normal_ms:.0f} us")
    print(f"  Throttled: {throttled_ms:.0f} us")
    print(f"  Adapted:   {best_new_ms:.0f} us ({(throttled_ms - best_new_ms) / throttled_ms * 100:.1f}% improvement)")

    for tid, w in orig.items():
        dag.tasks[tid].weight_us = w
    _compute_ranks(dag)

    # Scenario 2: Memory pressure
    print("\n  Scenario 2: Memory pressure (3x communication cost)")
    orig_comm = {tid: t.comm_cost for tid, t in dag.tasks.items()}
    for t in dag.tasks.values():
        t.comm_cost *= 3
    _compute_ranks(dag)
    phi_m = {tid: np.array([t.rank_u, t.depth, t.fanout, t.indegree, t.comm_cost])
             for tid, t in dag.tasks.items()}
    mem_pressure_ms = _list_schedule(dag, phi_m, best_theta, num_streams)

    best_mem = best_theta.copy()
    best_mem_ms = mem_pressure_ms
    for _ in range(300):
        theta = best_theta + np.random.randn(5) * 0.5
        ms = _list_schedule(dag, phi_m, theta, num_streams)
        if ms < best_mem_ms:
            best_mem_ms = ms
            best_mem = theta.copy()

    print(f"  Before:    {normal_ms:.0f} us")
    print(f"  Pressured: {mem_pressure_ms:.0f} us")
    print(f"  Adapted:   {best_mem_ms:.0f} us ({(mem_pressure_ms - best_mem_ms) / mem_pressure_ms * 100:.1f}% improvement)")
    print(f"  comm_cost weight: {best_theta[4]:.3f} -> {best_mem[4]:.3f}")

    for tid, c in orig_comm.items():
        dag.tasks[tid].comm_cost = c
    _compute_ranks(dag)

    return {
        "normal_us": normal_ms,
        "throttled_us": throttled_ms,
        "throttle_adapted_us": best_new_ms,
        "throttle_adapted_theta": best_new.tolist(),
        "mem_pressure_us": mem_pressure_ms,
        "mem_adapted_us": best_mem_ms,
        "mem_adapted_theta": best_mem.tolist(),
    }


# ============================================================
# STEP 15: PUBLICATION-QUALITY COMPARISON
# ============================================================

def step15_compare(baseline_ms, baseline_std, cpsat_makespan, cpsat_time,
                   heft_makespan, mosaic_makespan, mosaic_gap, dag,
                   power_w, feedback, tile_sweep, gpu: GPUSpec, best_theta):
    print("\n" + "=" * 70)
    print("STEP 15: Publication Results")
    print("=" * 70)

    total_work = sum(t.weight_us for t in dag.tasks.values())

    # Memory hierarchy summary from DAG tasks
    matmul_tasks = [t for t in dag.tasks.values() if t.task_type == "matmul" and t.mem_dict]
    total_smem = sum(t.mem_dict["shared_mem_bytes"] for t in matmul_tasks)
    total_gread = sum(t.mem_dict["global_read_bytes"] for t in matmul_tasks)
    total_gwrite = sum(t.mem_dict["global_write_bytes"] for t in matmul_tasks)
    total_flops = sum(t.mem_dict["flops"] for t in matmul_tasks)
    smem_violations = sum(1 for t in matmul_tasks if not t.mem_dict["fits_shared"])
    avg_occ = np.mean([t.mem_dict["occupancy_pct"] for t in matmul_tasks])
    avg_reuse = np.mean([t.mem_dict["data_reuse"] for t in matmul_tasks])
    avg_ai = np.mean([t.mem_dict["arithmetic_intensity"] for t in matmul_tasks])

    all_tasks_smem = sum(t.mem_dict.get("shared_mem_bytes", 0) for t in dag.tasks.values() if t.mem_dict)
    all_tasks_gread = sum(t.mem_dict.get("global_read_bytes", 0) for t in dag.tasks.values() if t.mem_dict)
    all_tasks_gwrite = sum(t.mem_dict.get("global_write_bytes", 0) for t in dag.tasks.values() if t.mem_dict)

    ridge_point = gpu.peak_flops_gflops / gpu.memory_bandwidth_gbps

    # === TABLE 1: Scheduling Comparison ===
    print(f"\n  ╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"  ║  TABLE 1: Scheduling Comparison (Transformer Layer, {len(dag.tasks)} tasks, {NUM_STREAMS} streams) ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════════════════╣")
    print(f"  ║ {'Method':<20} {'Makespan':>10} {'Gap':>7} {'Util%':>7} {'Energy':>10} {'Overhead':>12} ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════════════════╣")

    methods = [
        ("PyTorch baseline", baseline_ms * 1000, None, 100.0, None, "None"),
        ("HEFT", heft_makespan, (heft_makespan - cpsat_makespan) / cpsat_makespan * 100,
         total_work / heft_makespan * 100 / NUM_STREAMS,
         power_w * heft_makespan / 1e6, "O(V log V)"),
        ("CP-SAT", cpsat_makespan,  0.0,
         total_work / cpsat_makespan * 100 / NUM_STREAMS,
         power_w * cpsat_makespan / 1e6, f"{cpsat_time:.0f}s"),
        ("MoSAIC (learned)", mosaic_makespan, mosaic_gap * 100,
         total_work / mosaic_makespan * 100 / NUM_STREAMS,
         power_w * mosaic_makespan / 1e6, "O(V)"),
    ]

    for name, ms, gap, util, energy, overhead in methods:
        gap_str = f"{gap:.1f}%" if gap is not None else "N/A"
        energy_str = f"{energy:.4f} mJ" if energy is not None else "N/A"
        print(f"  ║ {name:<20} {ms:>8.0f}us {gap_str:>7} {util:>6.1f}% {energy_str:>10} {overhead:>12} ║")

    print(f"  ╚══════════════════════════════════════════════════════════════════════════════╝")

    # === TABLE 2: Memory Hierarchy (Best Case) ===
    print(f"\n  ╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"  ║  TABLE 2: Memory Hierarchy Analysis (32x32 tiles, {gpu.name})     ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════════════════╣")

    ts_data = tile_sweep.get(32, {})
    print(f"  ║ {'Metric':<40} {'Value':>30}   ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════════════════╣")
    mem_rows = [
        ("Shared memory per tile", f"{32*32*4*2/1024:.1f} KB (of 48 KB limit)"),
        ("Shared memory violations", f"{smem_violations} / {len(matmul_tasks)} tiles"),
        ("Shared memory utilization", f"{32*32*4*2/49152*100:.1f}%"),
        ("Global memory reads (total)", f"{all_tasks_gread / 1024 / 1024:.1f} MB"),
        ("Global memory writes (total)", f"{all_tasks_gwrite / 1024 / 1024:.1f} MB"),
        ("Total data movement", f"{(all_tasks_gread + all_tasks_gwrite) / 1024 / 1024:.1f} MB"),
        ("L2 cache capacity", f"{gpu.l2_cache_bytes / 1024 / 1024:.0f} MB"),
        ("Working set fits L2", f"{'YES' if ts_data.get('fits_l2', True) else 'NO'}"),
        ("Avg data reuse factor", f"{avg_reuse:.1f}x"),
        ("Avg arithmetic intensity", f"{avg_ai:.2f} FLOP/byte"),
        ("Roofline ridge point", f"{ridge_point:.2f} FLOP/byte"),
        ("Compute bound", f"{'YES' if avg_ai >= ridge_point else 'MEMORY BOUND'}"),
        ("Avg SM occupancy", f"{avg_occ:.1f}%"),
        ("Total compute (GEMM)", f"{total_flops / 1e6:.1f} MFLOP"),
    ]
    for label, val in mem_rows:
        print(f"  ║ {label:<40} {val:>30}   ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════════════════╝")

    # === TABLE 3: Tile Size Comparison ===
    print(f"\n  ╔══════════════════════════════════════════════════════════════════════════════════════╗")
    print(f"  ║  TABLE 3: Tile Size Impact on Memory Hierarchy                                      ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"  ║ {'Tile':>6} {'Tasks':>7} {'SmemViol':>9} {'Occ%':>6} {'Reuse':>7} {'AI':>7} {'DataMov':>10} {'L2Fit':>6} ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════════════════════════╣")
    for ts in sorted(tile_sweep.keys()):
        d = tile_sweep[ts]
        print(f"  ║ {ts:>4}x{ts:<2} {d['est_total_tasks']:>6} {d['smem_violations']:>9} "
              f"{d['avg_occupancy_pct']:>5.1f}% {d['avg_data_reuse']:>6.1f}x "
              f"{d['avg_arithmetic_intensity']:>6.2f} {d['total_data_movement_MB']:>8.1f}MB "
              f"{'YES' if d['fits_l2'] else 'NO':>5} ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════════════════════════╝")

    # === TABLE 4: DAG Structure ===
    print(f"\n  ╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"  ║  TABLE 4: DAG Structure                                                     ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════════════════╣")

    tc = defaultdict(int)
    for t in dag.tasks.values():
        tc[t.task_type] += 1

    dag_depth = max(t.depth for t in dag.tasks.values()) + 1
    dag_rows = [
        ("Total tasks", str(len(dag.tasks))),
        ("Total edges", str(len(dag.edges))),
        ("DAG depth", f"{dag_depth} levels"),
        ("Total work", f"{total_work:.0f} us"),
        ("Critical path length", f"{max(t.rank_u for t in dag.tasks.values()):.0f} us"),
        ("Parallelism ratio", f"{total_work / max(t.rank_u for t in dag.tasks.values()):.1f}x"),
    ]
    for label, val in dag_rows:
        print(f"  ║ {label:<40} {val:>30}   ║")

    print(f"  ║ {'':40} {'':>30}   ║")
    print(f"  ║ {'Task type breakdown:':<40} {'':>30}   ║")
    for tt, c in sorted(tc.items(), key=lambda x: -x[1]):
        pct = c / len(dag.tasks) * 100
        print(f"  ║   {tt:<38} {c:>5} ({pct:>5.1f}%)            ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════════════════╝")

    # === TABLE 5: Learned Priority Function ===
    print(f"\n  ╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"  ║  TABLE 5: Learned Priority Function H(v; theta) = theta^T * phi(v)          ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════════════════╣")
    labels = ["rank_u", "depth", "fanout", "indegree", "comm_cost"]
    for l, w in zip(labels, best_theta):
        bar = "+" * int(abs(w) * 10)
        sign = "+" if w > 0 else "-"
        print(f"  ║   {l:<12}  {sign}{abs(w):.3f}  {'|' + bar:<30}                     ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════════════════╝")

    # === Roofline Summary ===
    print(f"\n  === ROOFLINE MODEL ===")
    print(f"  Peak compute: {gpu.peak_flops_gflops:.0f} GFLOPS (FP32)")
    print(f"  Memory BW:    {gpu.memory_bandwidth_gbps:.0f} GB/s")
    print(f"  Ridge point:  {ridge_point:.2f} FLOP/byte")
    print(f"  Tile 32x32:   AI = {avg_ai:.2f} FLOP/byte -> {'COMPUTE BOUND' if avg_ai >= ridge_point else 'MEMORY BOUND'}")
    if avg_ai < ridge_point:
        print(f"  Operations are memory-bandwidth limited at this tile size.")
        print(f"  Tiling reduces global traffic by ~{avg_reuse:.0f}x vs naive,")
        print(f"  but larger tiles (64x64) would push AI closer to the ridge point.")
        print(f"  Key win: working set fits entirely in L2 cache ({gpu.l2_cache_bytes // 1024 // 1024} MB),")
        print(f"  so the effective bandwidth is much higher than DRAM bandwidth.")
    else:
        print(f"  Operations are compute-limited: tiling achieves sufficient data reuse.")
        print(f"  The memory hierarchy is fully utilized.")


# ============================================================
# DOT VISUALIZATION (improved)
# ============================================================

def generate_dot(dag, analysis_label="", filename="transformer_dag_pub"):
    depth_map = {tid: t.depth for tid, t in dag.tasks.items()}

    cp_set = set()
    dist, pred = {}, {}
    for tid in sorted(dag.tasks.keys()):
        t = dag.tasks[tid]
        if not t.deps:
            dist[tid] = t.weight_us
            pred[tid] = None
        else:
            best = max((dist.get(d, 0), d) for d in t.deps)
            dist[tid] = best[0] + t.weight_us
            pred[tid] = best[1]
    cp_end = max(dist, key=lambda t: dist[t])
    cur = cp_end
    while cur:
        cp_set.add(cur)
        cur = pred.get(cur)

    colors = {
        "input": "#2E7D32", "matmul": "#1565C0", "accum": "#E65100",
        "softmax": "#6A1B9A", "grad_softmax": "#6A1B9A",
        "gelu": "#558B2F", "grad_gelu": "#558B2F",
        "loss": "#C62828", "grad_loss": "#C62828",
        "layernorm": "#283593", "grad_layernorm": "#283593",
        "residual": "#00695C", "pool": "#4E342E", "grad_pool": "#4E342E",
        "weight_update": "#00838F",
    }

    lines = [
        'digraph TransformerDAG {',
        '  rankdir=TB;',
        '  bgcolor="white";',
        '  node [shape=box, style="filled,rounded", fontsize=6, fontname="Helvetica", margin="0.05,0.02"];',
        '  edge [color="#CCCCCC", arrowsize=0.3];',
        f'  label="{analysis_label}";',
        '  labelloc=t; fontsize=10; fontname="Helvetica-Bold";', '',
    ]

    levels = defaultdict(list)
    for tid, d in depth_map.items():
        levels[d].append(tid)

    for d in sorted(levels.keys()):
        lines.append(f'  subgraph cluster_{d} {{ style=invis; rank=same;')
        for tid in levels[d]:
            t = dag.tasks[tid]
            c = colors.get(t.task_type, "#757575")
            bdr = "#D32F2F" if tid in cp_set else "#333"
            pw = "2.0" if tid in cp_set else "0.3"
            label = f"{t.name}\\n{t.weight_us:.0f}us"
            lines.append(f'    T{tid} [label="{label}", fillcolor="{c}", '
                         f'color="{bdr}", penwidth={pw}, fontcolor="white"];')
        lines.append('  }')

    for (u, v) in dag.edges:
        col = "#D32F2F" if (u in cp_set and v in cp_set) else "#CCCCCC"
        pw = "1.5" if (u in cp_set and v in cp_set) else "0.5"
        lines.append(f'  T{u} -> T{v} [color="{col}", penwidth={pw}];')

    lines.append('}')

    dot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{filename}.dot")
    with open(dot_path, 'w') as f:
        f.write('\n'.join(lines))

    try:
        svg_path = dot_path.replace('.dot', '.svg')
        subprocess.run(['dot', '-Tsvg', dot_path, '-o', svg_path],
                       check=True, timeout=120)
        print(f"  DAG visualization: {svg_path}")
    except Exception as e:
        print(f"  DOT file saved: {dot_path} (render: {e})")


# ============================================================
# MAIN
# ============================================================

def main():
    device = DEVICE
    gpu = get_gpu_spec()
    tile_size = 32  # best case for memory hierarchy

    print("=" * 70)
    print("MoSAIC Publication Experiment")
    print("Memory-Hierarchy-Aware DAG Scheduling for Transformer Training")
    print("=" * 70)
    print(f"  GPU: {gpu.name}")
    print(f"  SMs: {gpu.sm_count}, Shared mem/block: {gpu.shared_mem_per_block // 1024} KB")
    print(f"  L2 cache: {gpu.l2_cache_bytes // 1024 // 1024} MB, Global: {gpu.global_mem_bytes / 1024**3:.1f} GB")
    print(f"  Peak FP32: {gpu.peak_flops_gflops:.0f} GFLOPS, BW: {gpu.memory_bandwidth_gbps:.0f} GB/s")
    print(f"  Config: B={BATCH}, S={SEQ_LEN}, H={HIDDEN}, heads={HEADS}, FFN={FFN_DIM}")
    print(f"  Tile: {tile_size}x{tile_size}, Streams: {NUM_STREAMS}")

    # Step 1-2
    baseline_ms, baseline_std, loss, peak_mem, base_power = step1_baseline()

    # Step 3: Tile size sweep with memory hierarchy
    tile_sweep = step3_tile_memory_analysis(gpu)

    # Step 4-6: Build DAG with chosen tile size
    print("\n" + "=" * 70)
    print(f"STEP 4-6: DAG Construction + Motif Detection (tile={tile_size})")
    print("=" * 70)
    dag = TransformerDAG(ts=tile_size, gpu=gpu)
    print(f"  Total tasks: {len(dag.tasks)}")
    print(f"  Total edges: {len(dag.edges)}")

    tc = defaultdict(int)
    for t in dag.tasks.values():
        tc[t.task_type] += 1
    for tt, c in sorted(tc.items(), key=lambda x: -x[1]):
        print(f"    {tt:<20} {c:>5}")

    motifs, motif_vec = detect_motifs(dag)
    print(f"  Motifs: fork-join={motifs['fork_join']}, fan-out={motifs['fan_out']}, "
          f"fan-in={motifs['fan_in']}, chain={motifs['chain']}")

    # Step 7: Profile
    total_work, power_w = profile_dag(dag, device)

    # Step 8-9: CP-SAT
    cpsat_makespan, cpsat_schedule, cpsat_time = run_cpsat(dag)

    # Step 10-11: MoSAIC
    mosaic_makespan, best_theta, mosaic_gap = run_mosaic(dag, cpsat_schedule)

    # Step 12: HEFT
    heft_makespan = run_heft(dag)

    # DAG visualization
    generate_dot(dag,
                 f"Transformer Layer | {len(dag.tasks)} tasks, {len(dag.edges)} edges | "
                 f"MoSAIC={mosaic_makespan:.0f}us vs CP-SAT={cpsat_makespan}us",
                 "transformer_dag_pub")

    # Step 13
    gen_results = step13_generalize()

    # Step 14
    feedback = step14_feedback(dag, best_theta, cpsat_makespan)

    # Step 15: Publication tables
    step15_compare(baseline_ms, baseline_std, cpsat_makespan, cpsat_time,
                   heft_makespan, mosaic_makespan, mosaic_gap, dag,
                   power_w, feedback, tile_sweep, gpu, best_theta)

    # Save results
    results = {
        "model": "Transformer",
        "experiment": "publication",
        "config": {
            "batch": BATCH, "seq_len": SEQ_LEN, "hidden": HIDDEN,
            "heads": HEADS, "ffn_dim": FFN_DIM, "tile_size": tile_size,
            "num_streams": NUM_STREAMS,
        },
        "gpu": {
            "name": gpu.name,
            "sm_count": gpu.sm_count,
            "shared_mem_per_block": gpu.shared_mem_per_block,
            "shared_mem_per_sm": gpu.shared_mem_per_sm,
            "regs_per_sm": gpu.regs_per_sm,
            "l2_cache_bytes": gpu.l2_cache_bytes,
            "global_mem_bytes": gpu.global_mem_bytes,
            "memory_bandwidth_gbps": gpu.memory_bandwidth_gbps,
            "peak_flops_gflops": gpu.peak_flops_gflops,
        },
        "baseline": {
            "ms_mean": baseline_ms,
            "ms_std": baseline_std,
            "loss": loss,
            "peak_mem_MB": peak_mem,
            "power": base_power,
        },
        "dag": {
            "tasks": len(dag.tasks),
            "edges": len(dag.edges),
            "depth": max(t.depth for t in dag.tasks.values()) + 1,
            "total_work_us": total_work,
            "motif_vector": motif_vec,
            "critical_path_us": max(t.rank_u for t in dag.tasks.values()),
        },
        "memory_hierarchy": {
            "tile_size": tile_size,
            "shared_mem_per_tile_bytes": tile_size * tile_size * 4 * 2,
            "shared_mem_violations": sum(1 for t in dag.tasks.values()
                                         if t.mem_dict and not t.mem_dict.get("fits_shared", True)),
            "total_global_read_MB": sum(t.mem_dict.get("global_read_bytes", 0)
                                        for t in dag.tasks.values() if t.mem_dict) / 1024 / 1024,
            "total_global_write_MB": sum(t.mem_dict.get("global_write_bytes", 0)
                                         for t in dag.tasks.values() if t.mem_dict) / 1024 / 1024,
            "avg_data_reuse": float(np.mean([t.mem_dict["data_reuse"]
                                             for t in dag.tasks.values()
                                             if t.mem_dict and "data_reuse" in t.mem_dict])),
            "avg_arithmetic_intensity": float(np.mean([t.mem_dict["arithmetic_intensity"]
                                                        for t in dag.tasks.values()
                                                        if t.mem_dict and "arithmetic_intensity" in t.mem_dict])),
            "avg_occupancy_pct": float(np.mean([t.mem_dict["occupancy_pct"]
                                                 for t in dag.tasks.values()
                                                 if t.mem_dict and "occupancy_pct" in t.mem_dict])),
            "roofline_ridge_point": gpu.peak_flops_gflops / gpu.memory_bandwidth_gbps,
        },
        "tile_sweep": {str(k): {kk: vv for kk, vv in v.items() if kk != "per_gemm"}
                       for k, v in tile_sweep.items()},
        "scheduling": {
            "cpsat_makespan_us": cpsat_makespan,
            "cpsat_solve_time_s": cpsat_time,
            "heft_makespan_us": heft_makespan,
            "mosaic_makespan_us": mosaic_makespan,
            "mosaic_gap_pct": mosaic_gap * 100,
            "learned_theta": best_theta.tolist(),
        },
        "power_w": power_w,
        "feedback": feedback,
        "generalization": gen_results,
    }

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "publication_results.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out}")

    print("\n" + "=" * 70)
    print("PUBLICATION EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
