"""
MoSAIC Deep Analysis
=====================
Publication-quality analyses that reviewers expect:
  1. Ablation study: which features in theta matter?
  2. Multi-stream scaling: 2/4/8 processors
  3. Transfer learning: learn on small, apply to large
  4. Gantt chart visualization: HEFT vs MoSAIC side-by-side
  5. Statistical SAGA benchmark: 5 seeds, mean/std
  6. Motif-specific theta: different DAG structures learn different weights
"""

import time
import json
import os
import math
import heapq
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from saga import (
    TaskGraph, TaskGraphNode, TaskGraphEdge,
    Network, NetworkNode, NetworkEdge, Schedule, ScheduledTask
)
from saga.schedulers import HeftScheduler, CpopScheduler, PEFTScheduler
from saga.schedulers.data.random import (
    gen_in_trees, gen_out_trees, gen_parallel_chains
)


# ============================================================
# HELPERS
# ============================================================

def make_network(num_procs=2, comm_speed=1.0):
    nodes = []
    edges = []
    for i in range(num_procs):
        nodes.append(NetworkNode(name=f"P{i}", speed=1.0))
    for i in range(num_procs):
        for j in range(num_procs):
            sp = float('inf') if i == j else comm_speed
            edges.append(NetworkEdge(source=f"P{i}", target=f"P{j}", speed=sp))
    return Network(nodes=frozenset(nodes), edges=frozenset(edges))


def compute_features(tg: TaskGraph):
    tasks = {t.name: t for t in tg.tasks}
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    dep_size = {}
    for dep in tg.dependencies:
        successors[dep.source].append(dep.target)
        predecessors[dep.target].append(dep.source)
        dep_size[(dep.source, dep.target)] = dep.size

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

    rank_u = {}
    for t_name in reversed(topo):
        t = tasks[t_name]
        succs = successors[t_name]
        if not succs:
            rank_u[t_name] = t.cost
        else:
            rank_u[t_name] = t.cost + max(
                dep_size.get((t_name, s), 0) + rank_u[s] for s in succs
            )

    depth = {}
    for t_name in topo:
        preds = predecessors[t_name]
        depth[t_name] = 0 if not preds else max(depth[p] for p in preds) + 1

    phi = {}
    for t_name in tasks:
        fanout = len(successors[t_name])
        indegree = len(predecessors[t_name])
        comm = max((dep_size.get((t_name, s), 0) for s in successors[t_name]), default=0.0)
        phi[t_name] = np.array([rank_u[t_name], depth[t_name], fanout, indegree, comm])

    return phi, tasks, successors, predecessors, dep_size


def list_schedule(tg, net, phi, theta):
    tasks = {t.name: t for t in tg.tasks}
    proc_names = sorted([n.name for n in net.nodes])
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    dep_size = {}
    for dep in tg.dependencies:
        successors[dep.source].append(dep.target)
        predecessors[dep.target].append(dep.source)
        dep_size[(dep.source, dep.target)] = dep.size

    priority = {name: float(np.dot(theta, phi[name])) for name in tasks}
    in_count = {name: len(predecessors[name]) for name in tasks}
    ready_heap = [(-priority[name], name) for name in tasks if in_count[name] == 0]
    heapq.heapify(ready_heap)

    proc_avail = {p: 0.0 for p in proc_names}
    finish = {}
    task_proc = {}
    scheduled = set()
    mapping = {p: [] for p in proc_names}

    while ready_heap:
        neg_pri, name = heapq.heappop(ready_heap)
        if name in scheduled:
            continue
        t = tasks[name]
        best_p, best_start, best_end = None, 0, float('inf')
        for p in proc_names:
            earliest = proc_avail[p]
            for par in predecessors[name]:
                par_fin = finish[par]
                if task_proc[par] != p:
                    earliest = max(earliest, par_fin + dep_size.get((par, name), 0))
                else:
                    earliest = max(earliest, par_fin)
            end = earliest + t.cost
            if end < best_end:
                best_end = end
                best_start = earliest
                best_p = p

        finish[name] = best_end
        proc_avail[best_p] = best_end
        task_proc[name] = best_p
        scheduled.add(name)
        mapping[best_p].append(ScheduledTask(node=best_p, name=name, start=best_start, end=best_end))

        for child in successors[name]:
            in_count[child] -= 1
            if in_count[child] == 0:
                heapq.heappush(ready_heap, (-priority[child], child))

    makespan = max(finish.values()) if finish else 0
    return makespan, mapping, finish, task_proc


def learn_theta(tg, net, phi, reference_ms, n_explore=500, n_refine=300, n_fine=100, seed=42):
    best_theta = None
    best_gap = float('inf')
    rng = np.random.RandomState(seed)

    for _ in range(n_explore):
        theta = rng.randn(5)
        theta[0] = abs(theta[0]) * 2
        ms, _, _, _ = list_schedule(tg, net, phi, theta)
        gap = (ms - reference_ms) / reference_ms if reference_ms > 0 else 0
        if gap < best_gap:
            best_gap = gap
            best_theta = theta.copy()

    for i in range(n_refine):
        scale = 0.5 * (1 - i / n_refine)
        theta = best_theta + rng.randn(5) * scale
        ms, _, _, _ = list_schedule(tg, net, phi, theta)
        gap = (ms - reference_ms) / reference_ms if reference_ms > 0 else 0
        if gap < best_gap:
            best_gap = gap
            best_theta = theta.copy()

    for _ in range(n_fine):
        theta = best_theta + rng.randn(5) * 0.1
        ms, _, _, _ = list_schedule(tg, net, phi, theta)
        gap = (ms - reference_ms) / reference_ms if reference_ms > 0 else 0
        if gap < best_gap:
            best_gap = gap
            best_theta = theta.copy()

    return best_theta, best_gap


# ============================================================
# DAG GENERATORS
# ============================================================

def gen_erdos_renyi_dag(n, p=0.15, seed=42):
    rng = np.random.RandomState(seed)
    tasks = [TaskGraphNode(name=f"t{i}", cost=max(1.0, rng.exponential(10.0))) for i in range(n)]
    deps = []
    has_pred = set()
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                deps.append(TaskGraphEdge(source=f"t{i}", target=f"t{j}", size=max(0.1, rng.exponential(2.0))))
                has_pred.add(f"t{j}")
    for i in range(1, n):
        if f"t{i}" not in has_pred:
            deps.append(TaskGraphEdge(source="t0", target=f"t{i}", size=max(0.1, rng.exponential(1.0))))
    return TaskGraph(tasks=frozenset(tasks), dependencies=frozenset(deps))


def gen_layered_dag(layers, width, seed=42):
    rng = np.random.RandomState(seed)
    tasks = []
    deps = []
    layer_nodes = []
    for layer in range(layers):
        w = width if 0 < layer < layers - 1 else max(1, width // 4)
        current = []
        for j in range(w):
            tasks.append(TaskGraphNode(name=f"L{layer}_t{j}", cost=max(1.0, rng.exponential(8.0))))
            current.append(f"L{layer}_t{j}")
        if layer > 0:
            prev = layer_nodes[-1]
            for cur_name in current:
                num_p = min(len(prev), rng.randint(1, 4))
                parents = rng.choice(prev, size=num_p, replace=False)
                for par in parents:
                    deps.append(TaskGraphEdge(source=par, target=cur_name, size=max(0.1, rng.exponential(2.0))))
        layer_nodes.append(current)
    return TaskGraph(tasks=frozenset(tasks), dependencies=frozenset(deps))


def gen_fan_in_heavy(n=100, seed=42):
    """DAG dominated by fan-in motifs (many-to-one convergence)."""
    rng = np.random.RandomState(seed)
    tasks = [TaskGraphNode(name=f"t{i}", cost=max(1.0, rng.exponential(8.0))) for i in range(n)]
    deps = []
    # Create groups that converge
    group_size = 8
    num_groups = n // (group_size + 1)
    idx = 0
    sink_nodes = []
    for g in range(num_groups):
        sources = list(range(idx, min(idx + group_size, n - 1)))
        idx += group_size
        if idx >= n:
            break
        sink = idx
        idx += 1
        for s in sources:
            deps.append(TaskGraphEdge(source=f"t{s}", target=f"t{sink}", size=max(0.1, rng.exponential(2.0))))
        sink_nodes.append(sink)
    # Chain sinks together
    for i in range(len(sink_nodes) - 1):
        deps.append(TaskGraphEdge(source=f"t{sink_nodes[i]}", target=f"t{sink_nodes[i+1]}", size=max(0.1, rng.exponential(1.0))))
    return TaskGraph(tasks=frozenset(tasks), dependencies=frozenset(deps))


def gen_fan_out_heavy(n=100, seed=42):
    """DAG dominated by fan-out motifs (one-to-many broadcast)."""
    rng = np.random.RandomState(seed)
    tasks = [TaskGraphNode(name=f"t{i}", cost=max(1.0, rng.exponential(8.0))) for i in range(n)]
    deps = []
    group_size = 8
    num_groups = n // (group_size + 1)
    idx = 0
    source_nodes = []
    for g in range(num_groups):
        source = idx
        idx += 1
        targets = list(range(idx, min(idx + group_size, n)))
        idx += group_size
        for t in targets:
            deps.append(TaskGraphEdge(source=f"t{source}", target=f"t{t}", size=max(0.1, rng.exponential(2.0))))
        source_nodes.append(source)
        if idx >= n:
            break
    # Chain sources
    for i in range(len(source_nodes) - 1):
        deps.append(TaskGraphEdge(source=f"t{source_nodes[i]}", target=f"t{source_nodes[i+1]}", size=max(0.1, rng.exponential(1.0))))
    return TaskGraph(tasks=frozenset(tasks), dependencies=frozenset(deps))


def gen_fork_join(n=100, seed=42):
    """DAG with repeated fork-join patterns."""
    rng = np.random.RandomState(seed)
    tasks = [TaskGraphNode(name=f"t{i}", cost=max(1.0, rng.exponential(8.0))) for i in range(n)]
    deps = []
    width = 6
    idx = 0
    while idx < n - width - 1:
        fork = idx
        idx += 1
        children = list(range(idx, min(idx + width, n - 1)))
        idx += width
        if idx >= n:
            break
        join = idx
        idx += 1
        for c in children:
            deps.append(TaskGraphEdge(source=f"t{fork}", target=f"t{c}", size=max(0.1, rng.exponential(2.0))))
            deps.append(TaskGraphEdge(source=f"t{c}", target=f"t{join}", size=max(0.1, rng.exponential(2.0))))
        if idx < n:
            deps.append(TaskGraphEdge(source=f"t{join}", target=f"t{idx}", size=max(0.1, rng.exponential(1.0))))
    return TaskGraph(tasks=frozenset(tasks), dependencies=frozenset(deps))


# ============================================================
# 1. ABLATION STUDY
# ============================================================

def run_ablation(tg, net):
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Ablation Study -- Feature Importance")
    print("=" * 70)

    phi, *_ = compute_features(tg)
    heft = HeftScheduler()
    heft_ms = heft.schedule(net, tg).makespan

    # Full model
    full_theta, full_gap = learn_theta(tg, net, phi, heft_ms)
    full_ms, _, _, _ = list_schedule(tg, net, phi, full_theta)
    print(f"\n  Full model (all 5 features):")
    print(f"    theta = [{', '.join(f'{x:.3f}' for x in full_theta)}]")
    print(f"    Makespan: {full_ms:.1f} (HEFT: {heft_ms:.1f})")

    labels = ["rank_u", "depth", "fanout", "indegree", "comm_cost"]
    results = {"full": {"theta": full_theta.tolist(), "makespan": full_ms}}

    print(f"\n  {'Removed Feature':<18} {'Makespan':>10} {'Degradation':>13} {'Importance':>11}")
    print(f"  {'-' * 54}")

    for i, label in enumerate(labels):
        # Zero out feature i
        mask = np.ones(5)
        mask[i] = 0
        masked_phi = {name: phi[name] * mask for name in phi}
        ablated_theta, _ = learn_theta(tg, net, masked_phi, heft_ms)
        ablated_ms, _, _, _ = list_schedule(tg, net, masked_phi, ablated_theta)
        degradation = (ablated_ms - full_ms) / full_ms * 100
        importance = "CRITICAL" if degradation > 5 else "HIGH" if degradation > 2 else "MODERATE" if degradation > 0.5 else "LOW"
        print(f"  -{label:<17} {ablated_ms:>10.1f} {degradation:>+12.2f}% {importance:>11}")
        results[f"no_{label}"] = {"makespan": ablated_ms, "degradation_pct": round(degradation, 2), "importance": importance}

    # Random baseline (no learned features)
    random_theta = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # rank_u only
    rank_only_ms, _, _, _ = list_schedule(tg, net, phi, random_theta)
    deg = (rank_only_ms - full_ms) / full_ms * 100
    print(f"  {'rank_u only':<18} {rank_only_ms:>10.1f} {deg:>+12.2f}%  {'(baseline)':>11}")
    results["rank_u_only"] = {"makespan": rank_only_ms, "degradation_pct": round(deg, 2)}

    return results


# ============================================================
# 2. MULTI-STREAM SCALING
# ============================================================

def run_multi_stream(tg):
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Multi-Stream Scaling (2/4/8 processors)")
    print("=" * 70)

    phi, *_ = compute_features(tg)
    results = {}

    print(f"\n  {'Streams':<10} {'HEFT':>10} {'CPOP':>10} {'PEFT':>10} {'MoSAIC':>10} {'MoSAIC Gap':>12} {'Speedup':>10}")
    print(f"  {'-' * 72}")

    baseline_ms = None
    for nprocs in [2, 4, 8]:
        net = make_network(num_procs=nprocs)

        heft_ms = HeftScheduler().schedule(net, tg).makespan
        cpop_ms = CpopScheduler().schedule(net, tg).makespan
        peft_ms = PEFTScheduler().schedule(net, tg).makespan

        best_ref = min(heft_ms, cpop_ms, peft_ms)
        theta, gap = learn_theta(tg, net, phi, best_ref)
        mosaic_ms, _, _, _ = list_schedule(tg, net, phi, theta)

        if baseline_ms is None:
            baseline_ms = mosaic_ms
        speedup = baseline_ms / mosaic_ms

        mosaic_gap = (mosaic_ms - best_ref) / best_ref * 100
        print(f"  {nprocs:<10} {heft_ms:>10.1f} {cpop_ms:>10.1f} {peft_ms:>10.1f} {mosaic_ms:>10.1f} {mosaic_gap:>+11.2f}% {speedup:>9.2f}x")

        results[nprocs] = {
            "heft": heft_ms, "cpop": cpop_ms, "peft": peft_ms,
            "mosaic": mosaic_ms, "mosaic_gap_pct": round(mosaic_gap, 2),
            "theta": theta.tolist(), "speedup_vs_2": round(speedup, 2),
        }

    return results


# ============================================================
# 3. TRANSFER LEARNING
# ============================================================

def run_transfer_learning():
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Transfer Learning (learn small, apply large)")
    print("=" * 70)

    net = make_network(num_procs=2)
    results = {}

    # Learn on small DAGs
    small_dags = {
        "Layered-4x8": gen_layered_dag(4, 8, seed=42),
        "ER-30": gen_erdos_renyi_dag(30, p=0.15, seed=42),
        "InTree-3-3": gen_in_trees(1, 3, 3)[0],
    }

    # Test on larger DAGs
    large_dags = {
        "Layered-10x20": gen_layered_dag(10, 20, seed=99),
        "ER-200": gen_erdos_renyi_dag(200, p=0.1, seed=99),
        "InTree-6-3": gen_in_trees(1, 6, 3)[0],
        "Layered-12x25": gen_layered_dag(12, 25, seed=77),
    }

    # Learn theta on each small DAG
    learned_thetas = {}
    for name, tg in small_dags.items():
        phi, *_ = compute_features(tg)
        heft_ms = HeftScheduler().schedule(net, tg).makespan
        theta, gap = learn_theta(tg, net, phi, heft_ms)
        learned_thetas[name] = theta
        ms, _, _, _ = list_schedule(tg, net, phi, theta)
        print(f"\n  Learned on {name} ({len(tg.tasks)} tasks):")
        print(f"    theta = [{', '.join(f'{x:.3f}' for x in theta)}]")
        print(f"    MoSAIC: {ms:.1f}, HEFT: {heft_ms:.1f}")

    # Apply each learned theta to each large DAG
    print(f"\n  === TRANSFER RESULTS ===")
    print(f"  {'Target DAG':<20} {'Tasks':>6} {'HEFT':>10} {'Native':>10} ", end="")
    for src in small_dags:
        print(f" {src:>15}", end="")
    print()
    print(f"  {'-' * (48 + 16 * len(small_dags))}")

    for target_name, target_tg in large_dags.items():
        target_phi, *_ = compute_features(target_tg)
        heft_ms = HeftScheduler().schedule(net, target_tg).makespan

        # Native: learn directly on this large DAG
        native_theta, _ = learn_theta(target_tg, net, target_phi, heft_ms)
        native_ms, _, _, _ = list_schedule(target_tg, net, target_phi, native_theta)

        row = f"  {target_name:<20} {len(target_tg.tasks):>6} {heft_ms:>10.1f} {native_ms:>10.1f} "

        transfer_results = {}
        for src_name, src_theta in learned_thetas.items():
            transferred_ms, _, _, _ = list_schedule(target_tg, net, target_phi, src_theta)
            gap_vs_native = (transferred_ms - native_ms) / native_ms * 100
            row += f" {transferred_ms:>10.1f}({gap_vs_native:>+.1f}%)"
            transfer_results[src_name] = {
                "makespan": transferred_ms,
                "gap_vs_native_pct": round(gap_vs_native, 2),
                "gap_vs_heft_pct": round((transferred_ms - heft_ms) / heft_ms * 100, 2),
            }

        print(row)
        results[target_name] = {
            "tasks": len(target_tg.tasks),
            "heft": heft_ms,
            "native_mosaic": native_ms,
            "transfers": transfer_results,
        }

    return results


# ============================================================
# 4. GANTT CHART VISUALIZATION
# ============================================================

def run_gantt_chart(tg, net, label=""):
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Gantt Chart -- HEFT vs MoSAIC")
    print("=" * 70)

    phi, *_ = compute_features(tg)

    # HEFT schedule
    heft_result = HeftScheduler().schedule(net, tg)
    heft_ms = heft_result.makespan

    # MoSAIC schedule
    theta, _ = learn_theta(tg, net, phi, heft_ms)
    mosaic_ms, mosaic_mapping, _, _ = list_schedule(tg, net, phi, theta)

    proc_names = sorted([n.name for n in net.nodes])

    # Generate SVG Gantt chart
    svg = generate_gantt_svg(heft_result.mapping, mosaic_mapping, proc_names, heft_ms, mosaic_ms, label)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gantt_chart.svg")
    with open(out_path, 'w') as f:
        f.write(svg)
    print(f"  Gantt chart saved: {out_path}")
    print(f"  HEFT makespan:   {heft_ms:.1f}")
    print(f"  MoSAIC makespan: {mosaic_ms:.1f}")

    return {"heft_makespan": heft_ms, "mosaic_makespan": mosaic_ms, "svg_path": out_path}


def generate_gantt_svg(heft_mapping, mosaic_mapping, proc_names, heft_ms, mosaic_ms, label=""):
    """Generate a side-by-side Gantt chart SVG comparing HEFT and MoSAIC."""
    max_ms = max(heft_ms, mosaic_ms) * 1.05
    chart_w = 700
    chart_h_per_proc = 30
    margin_l = 80
    margin_r = 20
    margin_t = 40
    gap = 60
    num_procs = len(proc_names)
    section_h = num_procs * chart_h_per_proc + 30

    total_w = margin_l + chart_w + margin_r
    total_h = margin_t + section_h * 2 + gap + 40

    colors = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A", "#C62828", "#00695C", "#283593", "#558B2F"]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}">',
        '<style>',
        '  text { font-family: Helvetica, Arial, sans-serif; }',
        '  .title { font-size: 12px; font-weight: bold; }',
        '  .label { font-size: 9px; }',
        '  .task { opacity: 0.85; }',
        '  .axis { font-size: 8px; fill: #666; }',
        '</style>',
        f'<rect width="{total_w}" height="{total_h}" fill="white"/>',
    ]

    if label:
        lines.append(f'<text x="{total_w // 2}" y="15" text-anchor="middle" class="title">{label}</text>')

    for section_idx, (name, mapping, makespan) in enumerate([
        ("HEFT", heft_mapping, heft_ms),
        ("MoSAIC (learned)", mosaic_mapping, mosaic_ms),
    ]):
        y_off = margin_t + section_idx * (section_h + gap)
        lines.append(f'<text x="{margin_l}" y="{y_off}" class="title">{name} (makespan: {makespan:.1f})</text>')
        y_off += 10

        scale = chart_w / max_ms

        # Grid lines
        for tick in range(0, int(max_ms) + 1, max(1, int(max_ms // 10))):
            x = margin_l + tick * scale
            lines.append(f'<line x1="{x}" y1="{y_off}" x2="{x}" y2="{y_off + num_procs * chart_h_per_proc}" stroke="#eee" stroke-width="0.5"/>')
            lines.append(f'<text x="{x}" y="{y_off + num_procs * chart_h_per_proc + 12}" text-anchor="middle" class="axis">{tick}</text>')

        # Makespan line
        ms_x = margin_l + makespan * scale
        lines.append(f'<line x1="{ms_x}" y1="{y_off}" x2="{ms_x}" y2="{y_off + num_procs * chart_h_per_proc}" stroke="red" stroke-width="1.5" stroke-dasharray="4,2"/>')

        for p_idx, p_name in enumerate(proc_names):
            y = y_off + p_idx * chart_h_per_proc
            lines.append(f'<text x="{margin_l - 5}" y="{y + chart_h_per_proc // 2 + 3}" text-anchor="end" class="label">{p_name}</text>')
            lines.append(f'<rect x="{margin_l}" y="{y}" width="{chart_w}" height="{chart_h_per_proc}" fill="none" stroke="#ddd" stroke-width="0.5"/>')

            tasks_on_proc = mapping.get(p_name, [])
            for st in tasks_on_proc:
                x = margin_l + st.start * scale
                w = max(1, (st.end - st.start) * scale)
                c = colors[hash(st.name) % len(colors)]
                lines.append(f'<rect x="{x}" y="{y + 2}" width="{w}" height="{chart_h_per_proc - 4}" fill="{c}" class="task" rx="2"/>')
                if w > 20:
                    lines.append(f'<text x="{x + w / 2}" y="{y + chart_h_per_proc // 2 + 3}" text-anchor="middle" fill="white" font-size="6">{st.name[:8]}</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


# ============================================================
# 5. STATISTICAL SAGA BENCHMARK
# ============================================================

def run_statistical_benchmark():
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Statistical SAGA Benchmark (5 seeds)")
    print("=" * 70)

    net = make_network(num_procs=2)
    n_seeds = 5
    results = {}

    dag_generators = {
        "ER-100": lambda s: gen_erdos_renyi_dag(100, p=0.12, seed=s),
        "ER-200": lambda s: gen_erdos_renyi_dag(200, p=0.08, seed=s),
        "Layered-8x12": lambda s: gen_layered_dag(8, 12, seed=s),
        "Layered-10x15": lambda s: gen_layered_dag(10, 15, seed=s),
        "ForkJoin-100": lambda s: gen_fork_join(100, seed=s),
    }

    algo_makespans = defaultdict(lambda: defaultdict(list))

    for dag_name, gen_fn in dag_generators.items():
        print(f"\n  {dag_name}:")
        for seed in range(n_seeds):
            tg = gen_fn(seed * 100 + 42)
            phi, *_ = compute_features(tg)

            heft_ms = HeftScheduler().schedule(net, tg).makespan
            cpop_ms = CpopScheduler().schedule(net, tg).makespan
            peft_ms = PEFTScheduler().schedule(net, tg).makespan

            best_ref = min(heft_ms, cpop_ms, peft_ms)
            theta, _ = learn_theta(tg, net, phi, best_ref, seed=seed)
            mosaic_ms, _, _, _ = list_schedule(tg, net, phi, theta)

            algo_makespans[dag_name]["HEFT"].append(heft_ms)
            algo_makespans[dag_name]["CPOP"].append(cpop_ms)
            algo_makespans[dag_name]["PEFT"].append(peft_ms)
            algo_makespans[dag_name]["MoSAIC"].append(mosaic_ms)

        print(f"    {'Algorithm':<10} {'Mean':>10} {'Std':>10} {'Best':>10} {'Worst':>10}")
        print(f"    {'-' * 42}")
        for algo in ["MoSAIC", "HEFT", "CPOP", "PEFT"]:
            vals = algo_makespans[dag_name][algo]
            print(f"    {algo:<10} {np.mean(vals):>10.1f} {np.std(vals):>10.1f} {min(vals):>10.1f} {max(vals):>10.1f}")

    # Summary: win rate
    print(f"\n  === WIN RATE ACROSS ALL {len(dag_generators)} DAG TYPES x {n_seeds} SEEDS ===")
    wins = defaultdict(int)
    total = 0
    for dag_name in dag_generators:
        for seed_idx in range(n_seeds):
            total += 1
            best_algo = min(["HEFT", "CPOP", "PEFT", "MoSAIC"],
                            key=lambda a: algo_makespans[dag_name][a][seed_idx])
            wins[best_algo] += 1

    print(f"    {'Algorithm':<10} {'Wins':>6} {'Win Rate':>10}")
    print(f"    {'-' * 28}")
    for algo in sorted(wins, key=lambda a: -wins[a]):
        print(f"    {algo:<10} {wins[algo]:>5}/{total} {wins[algo] / total * 100:>9.1f}%")

    # Build results
    for dag_name in dag_generators:
        results[dag_name] = {}
        for algo in ["MoSAIC", "HEFT", "CPOP", "PEFT"]:
            vals = algo_makespans[dag_name][algo]
            results[dag_name][algo] = {
                "mean": round(np.mean(vals), 1),
                "std": round(np.std(vals), 1),
                "min": round(min(vals), 1),
                "max": round(max(vals), 1),
            }
    results["win_rate"] = {algo: {"wins": w, "total": total, "pct": round(w / total * 100, 1)} for algo, w in wins.items()}

    return results


# ============================================================
# 6. MOTIF-SPECIFIC THETA ANALYSIS
# ============================================================

def run_motif_analysis():
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Motif-Specific Theta Analysis")
    print("=" * 70)

    net = make_network(num_procs=2)
    labels = ["rank_u", "depth", "fanout", "indegree", "comm_cost"]

    motif_dags = {
        "Fan-In Heavy": gen_fan_in_heavy(100, seed=42),
        "Fan-Out Heavy": gen_fan_out_heavy(100, seed=42),
        "Fork-Join": gen_fork_join(100, seed=42),
        "Layered": gen_layered_dag(8, 12, seed=42),
        "ER Random": gen_erdos_renyi_dag(100, p=0.12, seed=42),
        "InTree": gen_in_trees(1, 5, 3)[0],
        "OutTree": gen_out_trees(1, 5, 3)[0],
        "ParChains": gen_parallel_chains(1, 6, 8)[0],
    }

    results = {}
    thetas = {}

    print(f"\n  {'DAG Type':<18} {'Tasks':>6} {'rank_u':>8} {'depth':>8} {'fanout':>8} {'indeg':>8} {'comm':>8} {'Gap':>8}")
    print(f"  {'-' * 82}")

    for name, tg in motif_dags.items():
        phi, *_ = compute_features(tg)
        heft_ms = HeftScheduler().schedule(net, tg).makespan
        theta, gap = learn_theta(tg, net, phi, heft_ms)
        ms, _, _, _ = list_schedule(tg, net, phi, theta)

        thetas[name] = theta
        actual_gap = (ms - heft_ms) / heft_ms * 100
        print(f"  {name:<18} {len(tg.tasks):>6} {theta[0]:>+8.3f} {theta[1]:>+8.3f} {theta[2]:>+8.3f} {theta[3]:>+8.3f} {theta[4]:>+8.3f} {actual_gap:>+7.2f}%")

        results[name] = {
            "tasks": len(tg.tasks),
            "theta": theta.tolist(),
            "mosaic_ms": ms,
            "heft_ms": heft_ms,
            "gap_pct": round(actual_gap, 2),
        }

    # Analyze which features differ most across motif types
    print(f"\n  === FEATURE VARIANCE ACROSS MOTIFS ===")
    all_thetas = np.array([thetas[n] for n in thetas])
    for i, label in enumerate(labels):
        vals = all_thetas[:, i]
        print(f"    {label:<12}: mean={np.mean(vals):>+7.3f}, std={np.std(vals):>6.3f}, range=[{np.min(vals):>+.3f}, {np.max(vals):>+.3f}]")

    # Key finding: which features are stable vs motif-dependent?
    print(f"\n  === KEY FINDINGS ===")
    stds = [np.std(all_thetas[:, i]) for i in range(5)]
    sorted_features = sorted(zip(labels, stds), key=lambda x: -x[1])
    print(f"    Most motif-dependent:  {sorted_features[0][0]} (std={sorted_features[0][1]:.3f})")
    print(f"    Most stable:           {sorted_features[-1][0]} (std={sorted_features[-1][1]:.3f})")

    results["feature_variance"] = {label: round(float(np.std(all_thetas[:, i])), 3) for i, label in enumerate(labels)}

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("MoSAIC Deep Analysis -- Publication-Quality Experiments")
    print("=" * 70)

    all_results = {}

    # Use a mid-size layered DAG for ablation, multi-stream, and gantt
    main_dag = gen_layered_dag(8, 15, seed=42)
    net2 = make_network(num_procs=2)
    print(f"\n  Main DAG: Layered-8x15, {len(main_dag.tasks)} tasks, {len(main_dag.dependencies)} edges")

    # 1. Ablation
    all_results["ablation"] = run_ablation(main_dag, net2)

    # 2. Multi-stream
    all_results["multi_stream"] = run_multi_stream(main_dag)

    # 3. Transfer learning
    all_results["transfer"] = run_transfer_learning()

    # 4. Gantt chart
    gantt_dag = gen_layered_dag(5, 8, seed=42)
    all_results["gantt"] = run_gantt_chart(gantt_dag, net2, "Layered DAG: HEFT vs MoSAIC Schedule Comparison")

    # 5. Statistical benchmark
    all_results["statistical"] = run_statistical_benchmark()

    # 6. Motif analysis
    all_results["motif_analysis"] = run_motif_analysis()

    # Save
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep_analysis_results.json")
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {out}")

    print("\n" + "=" * 70)
    print("DEEP ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
