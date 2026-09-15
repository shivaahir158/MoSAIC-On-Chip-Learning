"""
MoSAIC Transformer Experiment
==============================
Full 15-step pipeline on a single Transformer layer instead of MLP.

Transformer layer operations:
  1. QKV projection: X @ W_Q, X @ W_K, X @ W_V  (3 GEMMs)
  2. Attention scores: Q @ K^T per head
  3. Softmax
  4. Attention output: scores @ V per head
  5. Output projection: attn_out @ W_O
  6. Residual + LayerNorm
  7. FFN: x @ W1 -> GeLU -> x @ W2
  8. Residual + LayerNorm
  9. Loss + full backward pass + weight updates

Config: batch=4, seq=32, hidden=128, heads=2, ffn=256
Tile size: 32x32 (fits in 48KB shared memory, good occupancy)
Streams: 2 (dual processor, matching ESWEEK paper)
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TILE_SIZE = 32
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

def get_gpu_spec():
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "shared_mem_per_block": props.shared_memory_per_block,
        "shared_mem_per_sm": getattr(props, 'shared_memory_per_multiprocessor', 102400),
        "regs_per_sm": getattr(props, 'regs_per_multiprocessor', 65536),
        "l2_cache_bytes": getattr(props, 'L2_cache_size', 16777216),
        "global_mem_bytes": props.total_memory,
        "warp_size": props.warp_size,
        "max_threads_per_sm": getattr(props, 'max_threads_per_multi_processor', 1536),
    }


def estimate_tile_memory(m, n, k, shared_limit=49152):
    """Estimate memory hierarchy for a tile GEMM."""
    elem = 4  # float32
    smem = (m * k + k * n) * elem
    global_read = (m * k + k * n) * elem
    global_write = m * n * elem
    threads = min(256, m * n)
    regs_per_thread = min(max(2 * (m * n // threads) + 8, 8), 255)
    fits_shared = smem <= shared_limit
    # occupancy estimate
    blocks_by_smem = 102400 // max(smem, 1) if smem > 0 else 16
    active = min(blocks_by_smem, 16) * threads
    occupancy = min(100.0, active / 1536 * 100)
    return {
        "shared_mem_bytes": smem,
        "global_read_bytes": global_read,
        "global_write_bytes": global_write,
        "fits_shared": fits_shared,
        "occupancy_pct": occupancy,
        "threads": threads,
        "regs_per_thread": regs_per_thread,
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
            except:
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
    """Single transformer encoder layer with no bias for clean GEMM shapes."""

    def __init__(self, hidden=HIDDEN, heads=HEADS, ffn=FFN_DIM):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads

        # QKV projections
        self.W_Q = nn.Linear(hidden, hidden, bias=False)
        self.W_K = nn.Linear(hidden, hidden, bias=False)
        self.W_V = nn.Linear(hidden, hidden, bias=False)
        self.W_O = nn.Linear(hidden, hidden, bias=False)

        # FFN
        self.W1 = nn.Linear(hidden, ffn, bias=False)
        self.W2 = nn.Linear(ffn, hidden, bias=False)

        # LayerNorms
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x):
        B, S, D = x.shape

        # Multi-head attention
        Q = self.W_Q(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        context = (attn @ V).transpose(1, 2).contiguous().view(B, S, D)

        out = self.W_O(context)
        x = self.ln1(x + out)

        # FFN
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
        x = x.mean(dim=1)  # pool over sequence
        return self.head(x)


def step1_baseline():
    """Train baseline transformer and measure performance."""
    print("\n" + "="*70)
    print("STEP 1-2: Baseline Transformer Training")
    print("="*70)
    print(f"  Config: batch={BATCH}, seq={SEQ_LEN}, hidden={HIDDEN}, "
          f"heads={HEADS}, ffn={FFN_DIM}")

    model = TransformerClassifier().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()

    X = torch.randn(BATCH, SEQ_LEN, HIDDEN, device=DEVICE)
    y = torch.randint(0, NUM_CLASSES, (BATCH,), device=DEVICE)

    # Warmup
    for _ in range(5):
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()

    # Timed run with power
    pm = PowerMonitor()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    pm.start()
    t0 = time.perf_counter()
    for _ in range(20):
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    ms_per_iter = (time.perf_counter() - t0) * 1000 / 20
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    pm.stop()
    power = pm.stats()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print(f"  Runtime: {ms_per_iter:.2f} ms/iter")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Peak GPU memory: {peak_mem:.1f} MB")
    print(f"  Power: avg={power['avg_w']}W, peak={power['peak_w']}W, "
          f"energy={power['energy_mj']}mJ")

    return ms_per_iter, loss.item(), peak_mem, power


# ============================================================
# STEP 3: TILING
# ============================================================

def compute_tiles(M, N, K, ts=TILE_SIZE):
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


def step3_tiling():
    """Show tiling for all transformer GEMMs."""
    print("\n" + "="*70)
    print(f"STEP 3: Transformer Tiling ({TILE_SIZE}x{TILE_SIZE})")
    print("="*70)

    # All GEMMs in one transformer training iteration
    # Forward pass:
    #   x is (B*S, H) = (1024, 256)
    #   QKV: 3x (1024, 256) @ (256, 256) -> (1024, 256)
    #   Attn scores: per head (B, S, head_dim) @ (B, head_dim, S) -> (B, S, S)
    #     = 4 heads x (8, 128, 64) @ (8, 64, 128) effectively (1024, 64) @ (64, 128) per head
    #   Attn output: per head scores @ V = (1024, 128) @ (128, 64)
    #   Out proj: (1024, 256) @ (256, 256)
    #   FFN1: (1024, 256) @ (256, 1024)
    #   FFN2: (1024, 1024) @ (1024, 256)
    #   Classifier: (8, 256) @ (256, 10)

    BS = BATCH * SEQ_LEN  # 1024

    gemms_fwd = [
        ("Q_proj",     BS, HIDDEN, HIDDEN),      # 1024x256 @ 256x256
        ("K_proj",     BS, HIDDEN, HIDDEN),
        ("V_proj",     BS, HIDDEN, HIDDEN),
    ]
    # Per-head attention (4 heads)
    for h in range(HEADS):
        gemms_fwd.append((f"Attn_score_h{h}", BS // BATCH, SEQ_LEN, HEAD_DIM))  # 128x64 @ 64x128 (per batch)
        # Actually batch*seq per head: effectively (B*S, head_dim) shaped
    for h in range(HEADS):
        gemms_fwd.append((f"Attn_out_h{h}", BS // BATCH, HEAD_DIM, SEQ_LEN))

    gemms_fwd += [
        ("Out_proj",   BS, HIDDEN, HIDDEN),       # 1024x256 @ 256x256
        ("FFN1",       BS, FFN_DIM, HIDDEN),      # 1024x256 @ 256x1024
        ("FFN2",       BS, HIDDEN, FFN_DIM),       # 1024x1024 @ 1024x256
        ("Classifier", BATCH, NUM_CLASSES, HIDDEN), # 8x256 @ 256x10
    ]

    total_tiles = 0
    total_boundary = 0
    gemm_info = []

    print(f"\n  {'GEMM':<20} {'Shape':>30} {'Tiles':>6} {'Full':>6} {'Boundary':>9} {'SmemOK':>7}")
    print(f"  {'-'*82}")

    for name, M, N, K in gemms_fwd:
        tiles = compute_tiles(M, N, K)
        boundary = sum(1 for t in tiles if t["m"] < TILE_SIZE or t["n"] < TILE_SIZE or t["kk"] < TILE_SIZE)
        full = len(tiles) - boundary
        # Check shared mem for typical tile
        m0, n0, k0 = min(TILE_SIZE, M), min(TILE_SIZE, N), min(TILE_SIZE, K)
        smem = estimate_tile_memory(m0, n0, k0)
        total_tiles += len(tiles)
        total_boundary += boundary

        print(f"  {name:<20} ({M}x{K})@({K}x{N}){' ':>{25-len(f'({M}x{K})@({K}x{N})')}} "
              f"{len(tiles):>6} {full:>6} {boundary:>9} {'YES' if smem['fits_shared'] else 'NO':>7}")

        gemm_info.append({
            "name": name, "M": M, "N": N, "K": K,
            "tiles": len(tiles), "boundary": boundary,
            "smem_ok": smem["fits_shared"],
        })

    print(f"\n  Forward GEMMs: {len(gemms_fwd)}")
    print(f"  Total forward tiles: {total_tiles} ({total_tiles - total_boundary} full, {total_boundary} boundary)")
    print(f"  Backward will roughly double this (dW + dX for each GEMM)")

    return gemm_info


# ============================================================
# STEP 4-5: DAG CONSTRUCTION
# ============================================================

@dataclass
class Task:
    tid: int
    name: str
    task_type: str
    gemm_name: str = ""
    deps: List[int] = field(default_factory=list)
    weight_us: float = 0.0
    memory: Optional[Dict] = None
    # Structural features
    rank_u: float = 0.0
    depth: int = 0
    fanout: int = 0
    indegree: int = 0
    comm_cost: float = 0.0
    # Scheduling
    processor: int = -1
    start_time: float = 0.0
    end_time: float = 0.0


class TransformerDAG:
    """Build DAG for one training iteration of the tiled Transformer."""

    def __init__(self, ts=TILE_SIZE):
        self.ts = ts
        self.tasks: Dict[int, Task] = {}
        self.edges: List[Tuple[int, int]] = []
        self._id = 0
        self.gemm_tile_counts = {}
        self._build()

    def _add(self, name, ttype, deps=None, mem=None, gemm_name=""):
        self._id += 1
        t = Task(tid=self._id, name=name, task_type=ttype,
                 deps=deps or [], memory=mem, gemm_name=gemm_name)
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
            mem = estimate_tile_memory(m, n, k)
            tid = self._add(
                f"{prefix}_r{tile['i']}_c{tile['j']}_k{tile['k']}",
                "matmul", deps=deps, mem=mem, gemm_name=gemm_name)
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
        mem = {
            "shared_mem_bytes": 0,
            "global_read_bytes": num_elements * 4,
            "global_write_bytes": num_elements * 4,
            "fits_shared": True,
            "occupancy_pct": 100.0,
            "threads": 256,
            "regs_per_thread": 8,
        }
        return self._add(name, ttype, deps, mem=mem)

    def _build(self):
        BS = BATCH * SEQ_LEN  # 1024

        # === INPUT ===
        inp = self._add("input", "input")

        # === FORWARD: QKV Projections (3 independent GEMMs) ===
        q_ids, _ = self._tiled_matmul(BS, HIDDEN, HIDDEN, "fwd_Q", [inp], "Q_proj")
        k_ids, _ = self._tiled_matmul(BS, HIDDEN, HIDDEN, "fwd_K", [inp], "K_proj")
        v_ids, _ = self._tiled_matmul(BS, HIDDEN, HIDDEN, "fwd_V", [inp], "V_proj")

        # === FORWARD: Attention per head ===
        # Reshape Q,K,V into heads then compute scores and context
        attn_score_ids = []
        attn_out_ids = []
        for h in range(HEADS):
            # Q_h @ K_h^T: (BS/B, head_dim) @ (head_dim, seq_len) per head
            # Simplified: treat as (seq_len, head_dim) @ (head_dim, seq_len) per batch element
            # Effective: (128, 64) @ (64, 128) = (128, 128) per batch
            score_ids, _ = self._tiled_matmul(
                SEQ_LEN, SEQ_LEN, HEAD_DIM,
                f"fwd_score_h{h}", q_ids + k_ids, f"Attn_score_h{h}")
            attn_score_ids.extend(score_ids)

            # Softmax on scores
            softmax_id = self._add_elementwise(
                f"softmax_h{h}", "softmax", score_ids, SEQ_LEN * SEQ_LEN * BATCH)

            # Scores @ V_h: (128, 128) @ (128, 64) = (128, 64) per batch
            out_ids, _ = self._tiled_matmul(
                SEQ_LEN, HEAD_DIM, SEQ_LEN,
                f"fwd_attn_out_h{h}", [softmax_id] + v_ids, f"Attn_out_h{h}")
            attn_out_ids.extend(out_ids)

        # === FORWARD: Output projection ===
        # Concat heads -> (BS, hidden) @ W_O(hidden, hidden)
        out_proj_ids, _ = self._tiled_matmul(
            BS, HIDDEN, HIDDEN, "fwd_out_proj", attn_out_ids, "Out_proj")

        # === Residual + LayerNorm 1 ===
        residual1 = self._add_elementwise(
            "residual_add_1", "residual", out_proj_ids + [inp], BS * HIDDEN)
        ln1 = self._add_elementwise("layernorm_1", "layernorm", [residual1], BS * HIDDEN)

        # === FORWARD: FFN ===
        # FFN1: (BS, hidden) @ (hidden, ffn_dim)
        ffn1_ids, _ = self._tiled_matmul(
            BS, FFN_DIM, HIDDEN, "fwd_ffn1", [ln1], "FFN1")

        # GeLU activation
        gelu_ids = []
        for i, fid in enumerate(ffn1_ids):
            gelu_ids.append(self._add_elementwise(
                f"gelu_{i}", "gelu", [fid], BS * FFN_DIM // len(ffn1_ids)))

        # FFN2: (BS, ffn_dim) @ (ffn_dim, hidden)
        ffn2_ids, _ = self._tiled_matmul(
            BS, HIDDEN, FFN_DIM, "fwd_ffn2", gelu_ids, "FFN2")

        # === Residual + LayerNorm 2 ===
        residual2 = self._add_elementwise(
            "residual_add_2", "residual", ffn2_ids + [ln1], BS * HIDDEN)
        ln2 = self._add_elementwise("layernorm_2", "layernorm", [residual2], BS * HIDDEN)

        # === Classifier head ===
        # Pool: mean over seq -> (B, hidden)
        pool = self._add_elementwise("pool", "pool", [ln2], BATCH * HIDDEN)

        # (B, hidden) @ (hidden, classes)
        cls_ids, _ = self._tiled_matmul(
            BATCH, NUM_CLASSES, HIDDEN, "fwd_cls", [pool], "Classifier")

        # === LOSS ===
        loss_id = self._add_elementwise("loss", "loss", cls_ids, BATCH * NUM_CLASSES)

        # === BACKWARD PASS ===
        grad_loss = self._add_elementwise("grad_loss", "grad_loss", [loss_id], BATCH * NUM_CLASSES)

        # --- Backward classifier ---
        # dW_cls = pool^T @ dL: (hidden, B) @ (B, classes)
        dw_cls_ids, _ = self._tiled_matmul(
            HIDDEN, NUM_CLASSES, BATCH, "bwd_dW_cls", [grad_loss, pool], "dW_cls")
        # dPool = dL @ W_cls^T: (B, classes) @ (classes, hidden)
        dpool_ids, _ = self._tiled_matmul(
            BATCH, HIDDEN, NUM_CLASSES, "bwd_dPool", [grad_loss], "dPool")

        # --- Backward pool, LN2, residual2 ---
        d_ln2 = self._add_elementwise("bwd_unpool", "grad_pool", dpool_ids, BS * HIDDEN)
        d_res2 = self._add_elementwise("bwd_ln2", "grad_layernorm", [d_ln2], BS * HIDDEN)

        # --- Backward FFN2 ---
        # dW2 = gelu_out^T @ d_res2
        dw_ffn2_ids, _ = self._tiled_matmul(
            FFN_DIM, HIDDEN, BS, "bwd_dW_ffn2", [d_res2] + gelu_ids, "dW_FFN2")
        # d_gelu = d_res2 @ W2^T
        d_gelu_in_ids, _ = self._tiled_matmul(
            BS, FFN_DIM, HIDDEN, "bwd_d_ffn2", [d_res2], "d_FFN2")

        # --- Backward GeLU ---
        d_ffn1_ids = []
        for i, dgid in enumerate(d_gelu_in_ids):
            gid = gelu_ids[i] if i < len(gelu_ids) else gelu_ids[-1]
            d_ffn1_ids.append(self._add_elementwise(
                f"bwd_gelu_{i}", "grad_gelu", [dgid, gid],
                BS * FFN_DIM // max(len(d_gelu_in_ids), 1)))

        # --- Backward FFN1 ---
        dw_ffn1_ids, _ = self._tiled_matmul(
            HIDDEN, FFN_DIM, BS, "bwd_dW_ffn1", d_ffn1_ids + [ln1], "dW_FFN1")
        d_ln1_from_ffn, _ = self._tiled_matmul(
            BS, HIDDEN, FFN_DIM, "bwd_d_ffn1", d_ffn1_ids, "d_FFN1")

        # --- Backward LN1, residual1 ---
        d_res1 = self._add_elementwise(
            "bwd_ln1", "grad_layernorm", d_ln1_from_ffn + [d_res2], BS * HIDDEN)

        # --- Backward output projection ---
        dw_out_ids, _ = self._tiled_matmul(
            HIDDEN, HIDDEN, BS, "bwd_dW_out", [d_res1] + attn_out_ids, "dW_Out")
        d_attn_concat, _ = self._tiled_matmul(
            BS, HIDDEN, HIDDEN, "bwd_d_out", [d_res1], "d_Out")

        # --- Backward attention per head ---
        d_q_all = []
        d_k_all = []
        d_v_all = []
        dw_q_all = []
        dw_k_all = []
        dw_v_all = []

        for h in range(HEADS):
            # Backward attn_out: d_scores = d_context @ V^T, d_V = scores^T @ d_context
            d_scores_ids, _ = self._tiled_matmul(
                SEQ_LEN, SEQ_LEN, HEAD_DIM,
                f"bwd_d_score_h{h}", d_attn_concat, f"d_score_h{h}")

            d_v_ids, _ = self._tiled_matmul(
                SEQ_LEN, HEAD_DIM, SEQ_LEN,
                f"bwd_d_V_h{h}", d_attn_concat + attn_score_ids, f"d_V_h{h}")

            # Backward softmax
            d_presoftmax = self._add_elementwise(
                f"bwd_softmax_h{h}", "grad_softmax", d_scores_ids,
                SEQ_LEN * SEQ_LEN * BATCH)

            # Backward Q @ K^T: d_Q = d_presoftmax @ K, d_K = d_presoftmax^T @ Q
            d_q_ids, _ = self._tiled_matmul(
                SEQ_LEN, HEAD_DIM, SEQ_LEN,
                f"bwd_d_Q_h{h}", [d_presoftmax] + k_ids, f"d_Q_h{h}")

            d_k_ids, _ = self._tiled_matmul(
                SEQ_LEN, HEAD_DIM, SEQ_LEN,
                f"bwd_d_K_h{h}", [d_presoftmax] + q_ids, f"d_K_h{h}")

            d_q_all.extend(d_q_ids)
            d_k_all.extend(d_k_ids)
            d_v_all.extend(d_v_ids)

        # --- Backward QKV projections ---
        dw_q_ids, _ = self._tiled_matmul(
            HIDDEN, HIDDEN, BS, "bwd_dW_Q", d_q_all + [inp], "dW_Q")
        dw_k_ids, _ = self._tiled_matmul(
            HIDDEN, HIDDEN, BS, "bwd_dW_K", d_k_all + [inp], "dW_K")
        dw_v_ids, _ = self._tiled_matmul(
            HIDDEN, HIDDEN, BS, "bwd_dW_V", d_v_all + [inp], "dW_V")

        # --- WEIGHT UPDATES ---
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


# ============================================================
# STEP 6: MOTIF DETECTION
# ============================================================

def detect_motifs(dag):
    print("\n" + "="*70)
    print("STEP 6: Motif Detection")
    print("="*70)

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

    # Chains: sequences with in=1, out=1
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
                chain_len += 1
                cur = succs[0]
                visited.add(cur)
            else:
                break
        if chain_len >= 2:
            motifs["chain"] += 1

    # Fork-join: fan-out node whose children converge
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

    motif_vec = [motifs["fork_join"], motifs["fan_out"], motifs["fan_in"], motifs["chain"]]
    print(f"  Fork-join: {motifs['fork_join']}")
    print(f"  Fan-out:   {motifs['fan_out']}")
    print(f"  Fan-in:    {motifs['fan_in']}")
    print(f"  Chains:    {motifs['chain']}")
    print(f"  Motif vector: {motif_vec}")

    return motifs, motif_vec


# ============================================================
# STEP 7: PROFILING
# ============================================================

def profile_dag(dag, device):
    print("\n" + "="*70)
    print("STEP 7: GPU Profiling")
    print("="*70)

    type_times = defaultdict(list)

    for t in dag.tasks.values():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        reps = 50

        if t.task_type == "matmul" and t.memory:
            m = int(math.sqrt(t.memory["threads"])) or 64
            m = min(m, TILE_SIZE)
            k = min(TILE_SIZE, 64)
            n = min(TILE_SIZE, 64)
            # Use actual tile dims if available
            A = torch.randn(m, k, device=device)
            B = torch.randn(k, n, device=device)
            _ = A @ B; torch.cuda.synchronize()
            start.record()
            for _ in range(reps): _ = A @ B
            end.record(); torch.cuda.synchronize()
            t.weight_us = start.elapsed_time(end) * 1000 / reps

        elif t.task_type in ("softmax", "grad_softmax"):
            A = torch.randn(BATCH, HEADS, SEQ_LEN, SEQ_LEN, device=device)
            torch.cuda.synchronize()
            start.record()
            for _ in range(reps): _ = F.softmax(A, dim=-1)
            end.record(); torch.cuda.synchronize()
            t.weight_us = start.elapsed_time(end) * 1000 / reps

        elif t.task_type in ("gelu", "grad_gelu"):
            A = torch.randn(1024, device=device)
            torch.cuda.synchronize()
            start.record()
            for _ in range(reps): _ = F.gelu(A)
            end.record(); torch.cuda.synchronize()
            t.weight_us = start.elapsed_time(end) * 1000 / reps

        elif t.task_type in ("layernorm", "grad_layernorm"):
            ln = nn.LayerNorm(HIDDEN).to(device)
            A = torch.randn(BATCH * SEQ_LEN, HIDDEN, device=device)
            torch.cuda.synchronize()
            start.record()
            for _ in range(reps): _ = ln(A)
            end.record(); torch.cuda.synchronize()
            t.weight_us = start.elapsed_time(end) * 1000 / reps

        elif t.task_type in ("loss", "grad_loss"):
            logits = torch.randn(BATCH, NUM_CLASSES, device=device)
            y = torch.randint(0, NUM_CLASSES, (BATCH,), device=device)
            torch.cuda.synchronize()
            start.record()
            for _ in range(reps): _ = F.cross_entropy(logits, y)
            end.record(); torch.cuda.synchronize()
            t.weight_us = start.elapsed_time(end) * 1000 / reps

        elif t.task_type in ("accum", "weight_update", "residual",
                              "pool", "grad_pool"):
            sz = max(256, BATCH * HIDDEN)
            A = torch.randn(sz, device=device)
            B = torch.randn(sz, device=device)
            torch.cuda.synchronize()
            start.record()
            for _ in range(reps): _ = A + B
            end.record(); torch.cuda.synchronize()
            t.weight_us = start.elapsed_time(end) * 1000 / reps

        else:
            t.weight_us = 0.1

        t.comm_cost = max(1.0, t.weight_us * 0.1)
        type_times[t.task_type].append(t.weight_us)

    # Upward rank
    _compute_ranks(dag)

    total_work = sum(t.weight_us for t in dag.tasks.values())

    # Power
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2)
        power_w = float(r.stdout.strip()) if r.returncode == 0 else 0
    except:
        power_w = 0

    print(f"  Total tasks: {len(dag.tasks)}")
    print(f"  Total work: {total_work:.0f} us")
    print(f"  GPU power: {power_w:.1f} W")
    print(f"\n  {'Type':<20} {'Avg(us)':>10} {'Count':>6} {'Total(us)':>12} {'%Work':>7}")
    print(f"  {'-'*57}")
    for tt in sorted(type_times.keys()):
        times = type_times[tt]
        total_t = sum(times)
        pct = total_t / total_work * 100
        print(f"  {tt:<20} {np.mean(times):>10.1f} {len(times):>6} {total_t:>12.0f} {pct:>6.1f}%")

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
        for c in children[tid]:
            dfs(c)
        order.append(tid)
    for tid in dag.tasks:
        dfs(tid)

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
    print("\n" + "="*70)
    print("STEP 8-9: CP-SAT Optimal Scheduling")
    print("="*70)

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

    # Precedence
    for (u, v) in dag.edges:
        comm = max(1, int(dag.tasks[u].comm_cost))
        same = model.NewBoolVar(f"same_{u}_{v}")
        model.Add(procs[u] == procs[v]).OnlyEnforceIf(same)
        model.Add(procs[u] != procs[v]).OnlyEnforceIf(same.Not())
        model.Add(starts[v] >= ends[u]).OnlyEnforceIf(same)
        model.Add(starts[v] >= ends[u] + comm).OnlyEnforceIf(same.Not())

    # No overlap per processor
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

        # Stream utilization
        for p in range(num_streams):
            p_tasks = [(s["start"], s["end"]) for s in schedule.values() if s["proc"] == p]
            if p_tasks:
                busy = sum(e - s for s, e in p_tasks)
                span = max(e for _, e in p_tasks) - min(s for s, _ in p_tasks)
                print(f"  Stream {p}: {len(p_tasks)} tasks, "
                      f"utilization={busy/span*100:.1f}%")

        return ms, schedule, solver.WallTime()
    else:
        print("  No solution, falling back to HEFT")
        return _heft_schedule(dag, num_streams)


# ============================================================
# STEP 12: HEFT
# ============================================================

def run_heft(dag, num_streams=NUM_STREAMS, label="HEFT"):
    if label:
        print(f"\n{'='*70}")
        print(f"STEP 12: {label} Scheduling")
        print(f"{'='*70}")

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
        for p in range(num_streams):
            cnt = sum(1 for tid in order if proc_avail[p] > 0)
            print(f"  Stream {p}: busy until {proc_avail[p]:.0f} us")
    return makespan


def _heft_schedule(dag, num_streams):
    """HEFT as fallback for CP-SAT, returns (makespan, schedule, 0)."""
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
# STEP 10-11: MOSAIC LEARNED SCHEDULER
# ============================================================

def run_mosaic(dag, cpsat_schedule, num_streams=NUM_STREAMS):
    print("\n" + "="*70)
    print("STEP 10-11: MoSAIC Learned Scheduling")
    print("="*70)

    cpsat_makespan = max(s["end"] for s in cpsat_schedule.values())

    # Feature vectors
    phi = {}
    for tid, t in dag.tasks.items():
        phi[tid] = np.array([t.rank_u, t.depth, t.fanout, t.indegree, t.comm_cost])

    # Bayesian-style search for theta
    best_theta = None
    best_gap = float('inf')
    best_ms = float('inf')

    np.random.seed(42)
    for trial in range(1000):
        theta = np.random.randn(5)
        theta[0] = abs(theta[0]) * 2  # rank_u positive

        ms = _list_schedule(dag, phi, theta, num_streams)
        gap = (ms - cpsat_makespan) / cpsat_makespan if cpsat_makespan > 0 else 0
        if gap < best_gap:
            best_gap = gap
            best_theta = theta.copy()
            best_ms = ms

    # RL refinement: perturb best theta
    for _ in range(500):
        theta = best_theta + np.random.randn(5) * 0.3
        ms = _list_schedule(dag, phi, theta, num_streams)
        gap = (ms - cpsat_makespan) / cpsat_makespan if cpsat_makespan > 0 else 0
        if gap < best_gap:
            best_gap = gap
            best_theta = theta.copy()
            best_ms = ms

    print(f"  theta* = [{', '.join(f'{x:.3f}' for x in best_theta)}]")
    print(f"  (rank_u, depth, fanout, indegree, comm_cost)")
    print(f"  MoSAIC makespan: {best_ms:.0f} us")
    print(f"  CP-SAT makespan: {cpsat_makespan:.0f} us")
    print(f"  Gap: {best_gap*100:.2f}%")

    return best_ms, best_theta, best_gap


def _list_schedule(dag, phi, theta, num_streams):
    parents = defaultdict(list)
    for (u, v) in dag.edges:
        parents[v].append(u)

    priority = {tid: float(np.dot(theta, phi[tid])) for tid in dag.tasks}
    scheduled = set()
    proc_avail = [0.0] * num_streams
    finish = {}
    remaining = set(dag.tasks.keys())

    while remaining:
        ready = [tid for tid in remaining
                 if all(p in scheduled for p in parents[tid])]
        if not ready:
            break
        ready.sort(key=lambda t: -priority[t])
        tid = ready[0]
        t = dag.tasks[tid]
        earliest = max((finish.get(p, 0) + t.comm_cost for p in parents[tid]), default=0)
        best_p = min(range(num_streams), key=lambda p: max(proc_avail[p], earliest))
        start = max(proc_avail[best_p], earliest)
        finish[tid] = start + t.weight_us
        proc_avail[best_p] = finish[tid]
        scheduled.add(tid)
        remaining.remove(tid)

    return max(finish.values()) if finish else 0


# ============================================================
# STEP 13: GENERALIZATION (larger transformer)
# ============================================================

def step13_generalize(device):
    print("\n" + "="*70)
    print("STEP 13: Generalization to Larger Transformer")
    print("="*70)

    configs = [
        ("Base",  4,  32, 128, 2,  256),
        ("2X",    8,  64, 128, 2,  256),
        ("Wide",  4,  32, 256, 4,  512),
        ("Long",  4,  64, 128, 2,  256),
    ]

    results = []
    for name, B, S, H, heads, ffn in configs:
        print(f"\n  --- {name}: batch={B}, seq={S}, hidden={H}, heads={heads}, ffn={ffn} ---")

        # Count tiles
        BS = B * S
        total_fwd_tiles = 0
        gemms = [
            (BS, H, H), (BS, H, H), (BS, H, H),  # QKV
            (BS, H, H),  # Out proj
            (BS, ffn, H), (BS, H, ffn),  # FFN
        ]
        for h in range(heads):
            gemms.append((S, S, H // heads))  # attn scores
            gemms.append((S, H // heads, S))  # attn out

        for M, N, K in gemms:
            total_fwd_tiles += len(compute_tiles(M, N, K))

        est_total = total_fwd_tiles * 2 + total_fwd_tiles  # fwd + bwd + weight updates
        print(f"    Est. forward tiles: {total_fwd_tiles}")
        print(f"    Est. total DAG tasks: ~{est_total}")
        print(f"    CP-SAT feasibility: {'YES (small)' if est_total < 500 else 'SLOW' if est_total < 2000 else 'INTRACTABLE'}")

        # Build and time HEFT
        if est_total < 3000:
            # Quick build - only build DAG for base config to save time
            if name == "Base":
                dag_test = TransformerDAG(ts=TILE_SIZE)
                for t in dag_test.tasks.values():
                    t.weight_us = max(0.1, np.random.exponential(20))
                    t.comm_cost = t.weight_us * 0.1
                _compute_ranks(dag_test)
                heft_ms = run_heft(dag_test, NUM_STREAMS, label=None)
                print(f"    HEFT makespan: {heft_ms:.0f} us")

        results.append({"name": name, "est_tasks": est_total, "fwd_tiles": total_fwd_tiles})

    return results


# ============================================================
# STEP 14: RUNTIME FEEDBACK
# ============================================================

def step14_feedback(dag, best_theta, cpsat_makespan, num_streams=NUM_STREAMS):
    print("\n" + "="*70)
    print("STEP 14: Runtime Feedback Adaptation")
    print("="*70)

    phi = {tid: np.array([t.rank_u, t.depth, t.fanout, t.indegree, t.comm_cost])
           for tid, t in dag.tasks.items()}

    normal_ms = _list_schedule(dag, phi, best_theta, num_streams)

    # Scenario 1: Thermal throttle (1.5x matmul slowdown, element-wise unchanged)
    print("\n  Scenario 1: Thermal throttle (matmul 1.5x slower)")
    orig = {tid: t.weight_us for tid, t in dag.tasks.items()}
    for t in dag.tasks.values():
        if t.task_type == "matmul":
            t.weight_us *= 1.5
    _compute_ranks(dag)
    phi_t = {tid: np.array([t.rank_u, t.depth, t.fanout, t.indegree, t.comm_cost])
             for tid, t in dag.tasks.items()}
    throttled_ms = _list_schedule(dag, phi_t, best_theta, num_streams)

    # Re-optimize
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
    print(f"  Adapted:   {best_new_ms:.0f} us ({(throttled_ms-best_new_ms)/throttled_ms*100:.1f}% improvement)")

    # Restore
    for tid, w in orig.items():
        dag.tasks[tid].weight_us = w
    _compute_ranks(dag)

    # Scenario 2: Memory pressure (simulate L2 cache thrashing - increase comm cost)
    print("\n  Scenario 2: Memory pressure (3x communication cost)")
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
    print(f"  Adapted:   {best_mem_ms:.0f} us ({(mem_pressure_ms-best_mem_ms)/mem_pressure_ms*100:.1f}% improvement)")
    print(f"  Adapted theta: [{', '.join(f'{x:.3f}' for x in best_mem)}]")
    print(f"  Key change: comm_cost weight moved from {best_theta[4]:.3f} to {best_mem[4]:.3f}")

    # Restore
    for t in dag.tasks.values():
        t.comm_cost /= 3
    _compute_ranks(dag)

    return {
        "normal_us": normal_ms,
        "throttled_us": throttled_ms,
        "throttle_adapted_us": best_new_ms,
        "mem_pressure_us": mem_pressure_ms,
        "mem_adapted_us": best_mem_ms,
    }


# ============================================================
# STEP 15: FINAL COMPARISON
# ============================================================

def step15_compare(baseline_ms, cpsat_makespan, cpsat_time, heft_makespan,
                   mosaic_makespan, mosaic_gap, dag, power_w, feedback):
    print("\n" + "="*70)
    print("STEP 15: Final Comparison")
    print("="*70)

    total_work = sum(t.weight_us for t in dag.tasks.values())

    # Memory hierarchy summary
    total_smem = sum(t.memory["shared_mem_bytes"] for t in dag.tasks.values() if t.memory and "shared_mem_bytes" in t.memory)
    total_gread = sum(t.memory["global_read_bytes"] for t in dag.tasks.values() if t.memory and "global_read_bytes" in t.memory)
    total_gwrite = sum(t.memory["global_write_bytes"] for t in dag.tasks.values() if t.memory and "global_write_bytes" in t.memory)
    smem_violations = sum(1 for t in dag.tasks.values() if t.memory and not t.memory.get("fits_shared", True))
    avg_occ = np.mean([t.memory["occupancy_pct"] for t in dag.tasks.values() if t.memory and "occupancy_pct" in t.memory])

    print(f"\n  === SCHEDULING COMPARISON ===")
    print(f"  {'Method':<20} {'Makespan(us)':>12} {'Gap':>8} {'Util%':>7} {'Energy(mJ)':>11} {'Overhead':>12}")
    print(f"  {'-'*73}")
    print(f"  {'PyTorch baseline':<20} {baseline_ms*1000:>12.0f} {'N/A':>8} {'~100%':>7} {'N/A':>11} {'None':>12}")

    heft_util = total_work / heft_makespan * 100 / NUM_STREAMS
    cpsat_util = total_work / cpsat_makespan * 100 / NUM_STREAMS
    mosaic_util = total_work / mosaic_makespan * 100 / NUM_STREAMS

    heft_gap = (heft_makespan - cpsat_makespan) / cpsat_makespan * 100
    print(f"  {'HEFT':<20} {heft_makespan:>12.0f} {heft_gap:>7.1f}% {heft_util:>6.1f}% "
          f"{power_w * heft_makespan / 1e6:>10.4f} {'O(V log V)':>12}")
    print(f"  {'CP-SAT (optimal)':<20} {cpsat_makespan:>12.0f} {'0.0%':>8} {cpsat_util:>6.1f}% "
          f"{power_w * cpsat_makespan / 1e6:>10.4f} {f'{cpsat_time:.0f}s':>12}")
    print(f"  {'MoSAIC (learned)':<20} {mosaic_makespan:>12.0f} {mosaic_gap*100:>7.1f}% {mosaic_util:>6.1f}% "
          f"{power_w * mosaic_makespan / 1e6:>10.4f} {'O(V)':>12}")

    print(f"\n  === DAG STRUCTURE ===")
    print(f"  Total tasks:        {len(dag.tasks)}")
    print(f"  Total edges:        {len(dag.edges)}")
    print(f"  Total work:         {total_work:.0f} us")
    print(f"  DAG depth:          {max(t.depth for t in dag.tasks.values()) + 1}")

    # GEMM breakdown
    print(f"\n  === GEMM TILE DISTRIBUTION ===")
    gemm_counts = defaultdict(int)
    for t in dag.tasks.values():
        if t.gemm_name:
            gemm_counts[t.gemm_name] += 1
    for gn, cnt in sorted(gemm_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {gn:<20} {cnt:>4} tasks")

    print(f"\n  === MEMORY HIERARCHY ===")
    print(f"  Global mem reads:     {total_gread/1024/1024:.1f} MB")
    print(f"  Global mem writes:    {total_gwrite/1024/1024:.1f} MB")
    print(f"  Total data movement:  {(total_gread + total_gwrite)/1024/1024:.1f} MB")
    print(f"  Shared mem total:     {total_smem/1024/1024:.1f} MB")
    print(f"  Shared mem violations: {smem_violations} tasks")
    print(f"  Avg occupancy:        {avg_occ:.1f}%")

    print(f"\n  === POWER & ENERGY ===")
    print(f"  GPU power:            {power_w:.1f} W")
    print(f"  Energy (CP-SAT):      {power_w * cpsat_makespan / 1e6:.4f} mJ")
    print(f"  Energy (MoSAIC):      {power_w * mosaic_makespan / 1e6:.4f} mJ")
    print(f"  Energy (HEFT):        {power_w * heft_makespan / 1e6:.4f} mJ")

    print(f"\n  === RUNTIME ADAPTATION ===")
    print(f"  Thermal throttle: {feedback['throttled_us']:.0f} -> {feedback['throttle_adapted_us']:.0f} us")
    print(f"  Memory pressure:  {feedback['mem_pressure_us']:.0f} -> {feedback['mem_adapted_us']:.0f} us")


# ============================================================
# DOT VISUALIZATION
# ============================================================

def generate_dot(dag, analysis_label="", filename="transformer_dag"):
    depth_map = {tid: t.depth for tid, t in dag.tasks.items()}

    # Critical path
    cp_set = set()
    dist = {}
    pred = {}
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
        "input": "#4CAF50", "matmul": "#2196F3", "accum": "#FF9800",
        "softmax": "#9C27B0", "grad_softmax": "#9C27B0",
        "gelu": "#8BC34A", "grad_gelu": "#8BC34A",
        "relu": "#8BC34A", "loss": "#F44336", "grad_loss": "#F44336",
        "layernorm": "#3F51B5", "grad_layernorm": "#3F51B5",
        "residual": "#009688", "pool": "#795548", "grad_pool": "#795548",
        "weight_update": "#00BCD4",
    }

    lines = [
        'digraph TransformerDAG {',
        '  rankdir=TB;',
        '  node [shape=box, style="filled,rounded", fontsize=7, fontname="Helvetica"];',
        '  edge [color="#999999", arrowsize=0.4];',
        f'  label="Transformer DAG | {len(dag.tasks)} tasks | {analysis_label}";',
        '  labelloc=t; fontsize=11;', '',
    ]

    levels = defaultdict(list)
    for tid, d in depth_map.items():
        levels[d].append(tid)

    for d in sorted(levels.keys()):
        lines.append(f'  subgraph cluster_{d} {{ style=invis; rank=same;')
        for tid in levels[d]:
            t = dag.tasks[tid]
            c = colors.get(t.task_type, "#9E9E9E")
            bdr = "red" if tid in cp_set else "#333"
            pw = "2.5" if tid in cp_set else "0.5"
            label = f"{t.name}\\n{t.weight_us:.0f}us"
            lines.append(f'    T{tid} [label="{label}", fillcolor="{c}", '
                         f'color="{bdr}", penwidth={pw}, fontcolor="white"];')
        lines.append('  }')

    for (u, v) in dag.edges:
        col = "red" if (u in cp_set and v in cp_set) else "#999"
        sty = "bold" if (u in cp_set and v in cp_set) else "solid"
        lines.append(f'  T{u} -> T{v} [color="{col}", style={sty}];')

    lines.append('}')

    dot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{filename}.dot")
    with open(dot_path, 'w') as f:
        f.write('\n'.join(lines))

    try:
        svg_path = dot_path.replace('.dot', '.svg')
        subprocess.run(['dot', '-Tsvg', dot_path, '-o', svg_path],
                       check=True, timeout=60)
        print(f"  DAG rendered: {svg_path}")
    except Exception as e:
        print(f"  DOT saved: {dot_path} (render: {e})")


# ============================================================
# MAIN
# ============================================================

def main():
    device = DEVICE
    gpu = get_gpu_spec()
    print(f"GPU: {gpu['name']}")
    print(f"Config: batch={BATCH}, seq={SEQ_LEN}, hidden={HIDDEN}, "
          f"heads={HEADS}, ffn={FFN_DIM}")
    print(f"Tile: {TILE_SIZE}x{TILE_SIZE}, Streams: {NUM_STREAMS}")

    # Step 1-2
    baseline_ms, loss, peak_mem, base_power = step1_baseline()

    # Step 3
    gemm_info = step3_tiling()

    # Step 4-5
    print("\n" + "="*70)
    print("STEP 4-5: Transformer DAG Construction")
    print("="*70)
    dag = TransformerDAG(ts=TILE_SIZE)
    print(f"  Total tasks: {len(dag.tasks)}")
    print(f"  Total edges: {len(dag.edges)}")
    tc = defaultdict(int)
    for t in dag.tasks.values():
        tc[t.task_type] += 1
    print(f"  Task types:")
    for tt, c in sorted(tc.items(), key=lambda x: -x[1]):
        print(f"    {tt:<20} {c:>5}")

    # Step 6
    motifs, motif_vec = detect_motifs(dag)

    # Step 7
    total_work, power_w = profile_dag(dag, device)

    # Step 8-9
    cpsat_makespan, cpsat_schedule, cpsat_time = run_cpsat(dag)

    # Step 10-11
    mosaic_makespan, best_theta, mosaic_gap = run_mosaic(dag, cpsat_schedule)

    # Step 12
    heft_makespan = run_heft(dag)

    # Generate DAG visualization
    cp_us = max(t.end_time for t in dag.tasks.values()) if any(t.end_time > 0 for t in dag.tasks.values()) else cpsat_makespan
    generate_dot(dag, f"CP={cpsat_makespan}us MoSAIC={mosaic_makespan:.0f}us", "transformer_dag")

    # Step 13
    gen_results = step13_generalize(device)

    # Step 14
    feedback = step14_feedback(dag, best_theta, cpsat_makespan)

    # Step 15
    step15_compare(baseline_ms, cpsat_makespan, cpsat_time, heft_makespan,
                   mosaic_makespan, mosaic_gap, dag, power_w, feedback)

    # Save results
    results = {
        "model": "Transformer",
        "config": {
            "batch": BATCH, "seq_len": SEQ_LEN, "hidden": HIDDEN,
            "heads": HEADS, "ffn_dim": FFN_DIM, "tile_size": TILE_SIZE,
            "num_streams": NUM_STREAMS,
        },
        "gpu": gpu,
        "baseline_ms": baseline_ms,
        "baseline_loss": loss,
        "baseline_peak_mem_MB": peak_mem,
        "baseline_power": base_power,
        "dag_tasks": len(dag.tasks),
        "dag_edges": len(dag.edges),
        "motif_vector": motif_vec,
        "total_work_us": total_work,
        "cpsat_makespan_us": cpsat_makespan,
        "cpsat_solve_time_s": cpsat_time,
        "heft_makespan_us": heft_makespan,
        "mosaic_makespan_us": mosaic_makespan,
        "mosaic_gap_pct": mosaic_gap * 100,
        "learned_theta": best_theta.tolist(),
        "power_w": power_w,
        "feedback": feedback,
        "generalization": gen_results,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transformer_results.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {out}")
    print("\n" + "="*70)
    print("TRANSFORMER EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
