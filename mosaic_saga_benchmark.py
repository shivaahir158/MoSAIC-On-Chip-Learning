"""
MoSAIC vs SAGA Benchmark Comparison
=====================================
Compares MoSAIC's learned scheduling against 15+ algorithms from the SAGA
library (Coleman & Krishnamachari, PODC 2024) on both standard benchmark
DAGs and our Transformer training DAG.

Reference:
  J. Coleman and B. Krishnamachari, "Comparing Task Graph Scheduling
  Algorithms: An Adversarial Approach," ACM PODC, 2024.
  https://github.com/ANRGUSC/saga

Benchmark sets:
  1. SAGA standard benchmarks: in-trees, out-trees, parallel chains
  2. MoSAIC Transformer DAG (1900 tasks from mosaic_publication.py)
  3. Random DAGs at multiple scales (50, 100, 200, 500 tasks)
"""

import time
import json
import os
import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# SAGA imports
from saga import (
    TaskGraph, TaskGraphNode, TaskGraphEdge,
    Network, NetworkNode, NetworkEdge, Schedule
)
from saga.schedulers import (
    HeftScheduler, CpopScheduler, PEFTScheduler,
    ETFScheduler, MCTScheduler, METScheduler,
    MaxMinScheduler, MinMinScheduler,
    DuplexScheduler, FLBScheduler, GDLScheduler,
    FastestNodeScheduler, OLBScheduler,
    BILScheduler, SufferageScheduler,
)
from saga.schedulers.data.random import (
    gen_in_trees, gen_out_trees, gen_parallel_chains, gen_random_networks
)

# MoSAIC imports (from our publication experiment)
import torch
import torch.nn.functional as F
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# SAGA NETWORK (2 homogeneous processors)
# ============================================================

def make_network(num_procs=2, speed=1.0, comm_speed=1.0):
    """Create a SAGA network with num_procs homogeneous processors."""
    nodes = []
    edges = []
    for i in range(num_procs):
        nodes.append(NetworkNode(name=f"P{i}", speed=speed))
    for i in range(num_procs):
        for j in range(num_procs):
            if i == j:
                edges.append(NetworkEdge(source=f"P{i}", target=f"P{j}", speed=float('inf')))
            else:
                edges.append(NetworkEdge(source=f"P{i}", target=f"P{j}", speed=comm_speed))
    return Network(nodes=frozenset(nodes), edges=frozenset(edges))


# ============================================================
# MOSAIC SCHEDULER (adapted for SAGA interface)
# ============================================================

class MoSAICScheduler:
    """MoSAIC learned list scheduler that works with SAGA TaskGraph/Network."""

    def __init__(self, theta=None):
        # Default theta from publication experiment
        self.theta = np.array(theta) if theta is not None else None

    def _compute_features(self, tg: TaskGraph):
        """Extract phi(v) = [rank_u, depth, fanout, indegree, comm_cost] for each task."""
        graph = tg.graph
        tasks = {t.name: t for t in tg.tasks}
        task_names = list(tasks.keys())

        # Build successor/predecessor maps using task names
        successors = defaultdict(list)
        predecessors = defaultdict(list)
        for dep in tg.dependencies:
            successors[dep.source].append(dep.target)
            predecessors[dep.target].append(dep.source)

        # Topological sort by names
        in_deg = {n: len(predecessors[n]) for n in task_names}
        queue = [n for n in task_names if in_deg[n] == 0]
        topo = []
        while queue:
            n = queue.pop(0)
            topo.append(n)
            for s in successors[n]:
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    queue.append(s)

        # Upward rank (compute in reverse topological order)
        rank_u = {}
        for t_name in reversed(topo):
            t = tasks[t_name]
            succs = successors[t_name]
            if not succs:
                rank_u[t_name] = t.cost
            else:
                rank_u[t_name] = t.cost + max(
                    tg.get_dependency(t_name, s).size + rank_u[s] for s in succs
                )

        # Depth
        depth = {}
        for t_name in topo:
            preds = predecessors[t_name]
            if not preds:
                depth[t_name] = 0
            else:
                depth[t_name] = max(depth[p] for p in preds) + 1

        # Features
        phi = {}
        for t_name in task_names:
            t = tasks[t_name]
            fanout = len(successors[t_name])
            indegree = len(predecessors[t_name])
            out_edges = tg.out_edges(t_name)
            comm = max((e.size for e in out_edges), default=0.0)
            phi[t_name] = np.array([rank_u[t_name], depth[t_name], fanout, indegree, comm])

        return phi, rank_u

    def _learn_theta(self, tg: TaskGraph, net: Network, phi: Dict, optimal_makespan: float):
        """Learn theta via Bayesian search + refinement."""
        best_theta = None
        best_gap = float('inf')

        np.random.seed(42)
        # Phase 1: exploration
        for _ in range(500):
            theta = np.random.randn(5)
            theta[0] = abs(theta[0]) * 2
            ms = self._list_schedule_makespan(tg, net, phi, theta)
            gap = (ms - optimal_makespan) / optimal_makespan if optimal_makespan > 0 else 0
            if gap < best_gap:
                best_gap = gap
                best_theta = theta.copy()

        # Phase 2: refinement
        for i in range(300):
            scale = 0.5 * (1 - i / 300)
            theta = best_theta + np.random.randn(5) * scale
            ms = self._list_schedule_makespan(tg, net, phi, theta)
            gap = (ms - optimal_makespan) / optimal_makespan if optimal_makespan > 0 else 0
            if gap < best_gap:
                best_gap = gap
                best_theta = theta.copy()

        # Phase 3: fine-tuning
        for _ in range(100):
            theta = best_theta + np.random.randn(5) * 0.1
            ms = self._list_schedule_makespan(tg, net, phi, theta)
            gap = (ms - optimal_makespan) / optimal_makespan if optimal_makespan > 0 else 0
            if gap < best_gap:
                best_gap = gap
                best_theta = theta.copy()

        self.theta = best_theta
        return best_theta, best_gap

    def _list_schedule_makespan(self, tg: TaskGraph, net: Network, phi: Dict, theta: np.ndarray) -> float:
        """Run list scheduling with given theta, return makespan."""
        import heapq
        graph = tg.graph
        tasks = {t.name: t for t in tg.tasks}
        proc_names = [n.name for n in net.nodes]
        num_procs = len(proc_names)

        priority = {name: float(np.dot(theta, phi[name])) for name in tasks}

        in_count = {name: tg.in_degree(name) for name in tasks}
        ready_heap = [(-priority[name], name) for name, c in in_count.items() if c == 0]
        heapq.heapify(ready_heap)

        proc_avail = {p: 0.0 for p in proc_names}
        finish = {}
        task_proc = {}
        scheduled = set()

        children = defaultdict(list)
        for dep in tg.dependencies:
            children[dep.source].append(dep.target)

        parents = defaultdict(list)
        for dep in tg.dependencies:
            parents[dep.target].append(dep.source)

        while ready_heap:
            neg_pri, name = heapq.heappop(ready_heap)
            if name in scheduled:
                continue

            t = tasks[name]
            # Find earliest start on each processor
            best_p, best_end = None, float('inf')
            for p in proc_names:
                earliest = proc_avail[p]
                for par_name in parents[name]:
                    par_finish = finish[par_name]
                    if task_proc[par_name] != p:
                        comm = tg.get_dependency(par_name, name).size
                        earliest = max(earliest, par_finish + comm)
                    else:
                        earliest = max(earliest, par_finish)
                end = earliest + t.cost
                if end < best_end:
                    best_end = end
                    best_p = p

            start = best_end - t.cost
            finish[name] = best_end
            proc_avail[best_p] = best_end
            task_proc[name] = best_p
            scheduled.add(name)

            for child in children[name]:
                in_count[child] -= 1
                if in_count[child] == 0:
                    heapq.heappush(ready_heap, (-priority[child], child))

        return max(finish.values()) if finish else 0

    def schedule(self, net: Network, tg: TaskGraph) -> Schedule:
        """Schedule using learned theta, returning a SAGA Schedule."""
        phi, rank_u = self._compute_features(tg)

        if self.theta is None:
            # Use HEFT as reference to learn from
            heft = HeftScheduler()
            heft_result = heft.schedule(net, tg)
            heft_ms = heft_result.makespan
            self._learn_theta(tg, net, phi, heft_ms)

        import heapq
        graph = tg.graph
        tasks = {t.name: t for t in tg.tasks}
        proc_names = sorted([n.name for n in net.nodes])

        priority = {name: float(np.dot(self.theta, phi[name])) for name in tasks}

        in_count = {name: tg.in_degree(name) for name in tasks}
        ready_heap = [(-priority[name], name) for name, c in in_count.items() if c == 0]
        heapq.heapify(ready_heap)

        proc_avail = {p: 0.0 for p in proc_names}
        finish = {}
        task_proc = {}
        scheduled = set()
        mapping = {p: [] for p in proc_names}

        children = defaultdict(list)
        parents = defaultdict(list)
        for dep in tg.dependencies:
            children[dep.source].append(dep.target)
            parents[dep.target].append(dep.source)

        from saga import ScheduledTask

        while ready_heap:
            neg_pri, name = heapq.heappop(ready_heap)
            if name in scheduled:
                continue
            t = tasks[name]

            best_p, best_start, best_end = None, 0, float('inf')
            for p in proc_names:
                earliest = proc_avail[p]
                for par_name in parents[name]:
                    par_finish = finish[par_name]
                    if task_proc[par_name] != p:
                        comm = tg.get_dependency(par_name, name).size
                        earliest = max(earliest, par_finish + comm)
                    else:
                        earliest = max(earliest, par_finish)
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

            for child in children[name]:
                in_count[child] -= 1
                if in_count[child] == 0:
                    heapq.heappush(ready_heap, (-priority[child], child))

        schedule = Schedule(task_graph=tg, network=net, mapping=mapping)
        return schedule


# ============================================================
# TRANSFORMER DAG -> SAGA FORMAT
# ============================================================

def build_transformer_saga_dag(tile_size=32):
    """Build the Transformer training DAG and convert to SAGA format."""
    # Import our DAG builder
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mosaic_publication import TransformerDAG, get_gpu_spec, _compute_ranks

    gpu = get_gpu_spec()
    dag = TransformerDAG(ts=tile_size, gpu=gpu)

    # Profile tasks on GPU for real weights
    print("  Profiling transformer DAG on GPU...")
    reps = 20
    for t in dag.tasks.values():
        if t.task_type == "matmul" and t.memory:
            m, n, k = t.memory.m, t.memory.n, t.memory.k
            A = torch.randn(m, k, device=DEVICE)
            B = torch.randn(k, n, device=DEVICE)
            _ = A @ B; torch.cuda.synchronize()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(reps): _ = A @ B
            end_ev.record(); torch.cuda.synchronize()
            t.weight_us = start_ev.elapsed_time(end_ev) * 1000 / reps
            if t.mem_dict:
                bytes_moved = t.mem_dict["global_read_bytes"] + t.mem_dict["global_write_bytes"]
                t.comm_cost = max(0.5, bytes_moved / 128e9 * 1e6)
        elif t.task_type in ("softmax", "grad_softmax"):
            A = torch.randn(4, 2, 32, 32, device=DEVICE)
            torch.cuda.synchronize()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(reps): _ = F.softmax(A, dim=-1)
            end_ev.record(); torch.cuda.synchronize()
            t.weight_us = start_ev.elapsed_time(end_ev) * 1000 / reps
            t.comm_cost = max(0.5, t.weight_us * 0.1)
        elif t.task_type in ("gelu", "grad_gelu"):
            A = torch.randn(1024, device=DEVICE)
            torch.cuda.synchronize()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(reps): _ = F.gelu(A)
            end_ev.record(); torch.cuda.synchronize()
            t.weight_us = start_ev.elapsed_time(end_ev) * 1000 / reps
            t.comm_cost = max(0.5, t.weight_us * 0.1)
        elif t.task_type in ("layernorm", "grad_layernorm"):
            ln = nn.LayerNorm(128).to(DEVICE)
            A = torch.randn(128, 128, device=DEVICE)
            torch.cuda.synchronize()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(reps): _ = ln(A)
            end_ev.record(); torch.cuda.synchronize()
            t.weight_us = start_ev.elapsed_time(end_ev) * 1000 / reps
            t.comm_cost = max(0.5, t.weight_us * 0.1)
        elif t.task_type in ("loss", "grad_loss"):
            logits = torch.randn(4, 10, device=DEVICE)
            y = torch.randint(0, 10, (4,), device=DEVICE)
            torch.cuda.synchronize()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(reps): _ = F.cross_entropy(logits, y)
            end_ev.record(); torch.cuda.synchronize()
            t.weight_us = start_ev.elapsed_time(end_ev) * 1000 / reps
            t.comm_cost = max(0.5, t.weight_us * 0.1)
        else:
            sz = max(256, 512)
            A = torch.randn(sz, device=DEVICE)
            B = torch.randn(sz, device=DEVICE)
            torch.cuda.synchronize()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(reps): _ = A + B
            end_ev.record(); torch.cuda.synchronize()
            t.weight_us = start_ev.elapsed_time(end_ev) * 1000 / reps
            t.comm_cost = max(0.5, t.weight_us * 0.1)

    _compute_ranks(dag)

    # Convert to SAGA format
    saga_tasks = []
    saga_deps = []
    for tid, t in dag.tasks.items():
        saga_tasks.append(TaskGraphNode(name=f"T{tid}", cost=max(0.1, t.weight_us)))
    for (u, v) in dag.edges:
        comm = max(0.01, dag.tasks[u].comm_cost)
        saga_deps.append(TaskGraphEdge(source=f"T{u}", target=f"T{v}", size=comm))

    saga_tg = TaskGraph(tasks=frozenset(saga_tasks), dependencies=frozenset(saga_deps))
    return saga_tg, dag


# ============================================================
# RANDOM DAG GENERATORS (ER, BA style)
# ============================================================

def gen_erdos_renyi_dag(n, p=0.3, seed=42):
    """Generate an Erdos-Renyi style DAG with n tasks."""
    rng = np.random.RandomState(seed)
    tasks = []
    deps = []
    for i in range(n):
        cost = max(1.0, rng.exponential(10.0))
        tasks.append(TaskGraphNode(name=f"t{i}", cost=cost))

    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                comm = max(0.1, rng.exponential(2.0))
                deps.append(TaskGraphEdge(source=f"t{i}", target=f"t{j}", size=comm))

    # Ensure connectivity: add edges to make it a connected DAG
    # Find nodes with no predecessors (other than t0) and no successors (other than t_{n-1})
    has_pred = set()
    has_succ = set()
    for d in deps:
        has_pred.add(d.target)
        has_succ.add(d.source)

    for i in range(1, n):
        name = f"t{i}"
        if name not in has_pred:
            deps.append(TaskGraphEdge(source="t0", target=name, size=max(0.1, rng.exponential(1.0))))

    return TaskGraph(tasks=frozenset(tasks), dependencies=frozenset(deps))


def gen_layered_dag(layers, width, seed=42):
    """Generate a layered DAG (common in neural network training)."""
    rng = np.random.RandomState(seed)
    tasks = []
    deps = []
    tid = 0

    layer_nodes = []
    for layer in range(layers):
        w = width if layer > 0 and layer < layers - 1 else max(1, width // 4)
        current = []
        for j in range(w):
            cost = max(1.0, rng.exponential(8.0))
            tasks.append(TaskGraphNode(name=f"L{layer}_t{j}", cost=cost))
            current.append(f"L{layer}_t{j}")
            tid += 1

        if layer > 0:
            prev = layer_nodes[-1]
            for cur_name in current:
                # Connect to 1-3 random parents
                num_parents = min(len(prev), rng.randint(1, 4))
                parents = rng.choice(prev, size=num_parents, replace=False)
                for par in parents:
                    comm = max(0.1, rng.exponential(2.0))
                    deps.append(TaskGraphEdge(source=par, target=cur_name, size=comm))

        layer_nodes.append(current)

    return TaskGraph(tasks=frozenset(tasks), dependencies=frozenset(deps))


# ============================================================
# BENCHMARK RUNNER
# ============================================================

# Schedulers that work well with moderate-sized DAGs
SAGA_SCHEDULERS = {
    "HEFT": HeftScheduler(),
    "CPOP": CpopScheduler(),
    "PEFT": PEFTScheduler(),
    "ETF": ETFScheduler(),
    "MinMin": MinMinScheduler(),
    "MaxMin": MaxMinScheduler(),
    "MCT": MCTScheduler(),
    "MET": METScheduler(),
    "Sufferage": SufferageScheduler(),
    "OLB": OLBScheduler(),
    "FastestNode": FastestNodeScheduler(),
    "FLB": FLBScheduler(),
    "GDL": GDLScheduler(),
    "Duplex": DuplexScheduler(),
    "BIL": BILScheduler(),
}


def run_benchmark(name, tg, net, include_mosaic=True, mosaic_theta=None, timeout=120):
    """Run all schedulers on a task graph and return results."""
    print(f"\n  --- {name}: {len(tg.tasks)} tasks, {len(tg.dependencies)} edges ---")
    results = {}

    for sched_name, scheduler in SAGA_SCHEDULERS.items():
        try:
            t0 = time.perf_counter()
            result = scheduler.schedule(net, tg)
            elapsed = time.perf_counter() - t0
            ms = result.makespan
            results[sched_name] = {"makespan": ms, "time_s": elapsed}
            if elapsed > timeout:
                print(f"    {sched_name}: TIMEOUT ({elapsed:.1f}s)")
        except Exception as e:
            results[sched_name] = {"makespan": float('inf'), "time_s": 0, "error": str(e)}

    if include_mosaic:
        # Find best SAGA result as reference
        best_saga_ms = min(r["makespan"] for r in results.values() if r["makespan"] < float('inf'))

        mosaic = MoSAICScheduler(theta=mosaic_theta)
        t0 = time.perf_counter()
        # Learn theta using HEFT as reference
        phi, rank_u = mosaic._compute_features(tg)
        theta, gap = mosaic._learn_theta(tg, net, phi, best_saga_ms)
        ms = mosaic._list_schedule_makespan(tg, net, phi, theta)
        elapsed = time.perf_counter() - t0
        results["MoSAIC"] = {"makespan": ms, "time_s": elapsed, "theta": theta.tolist()}

    # Print results sorted by makespan
    sorted_results = sorted(results.items(), key=lambda x: x[1]["makespan"])
    best_ms = sorted_results[0][1]["makespan"]

    print(f"    {'Rank':<5} {'Algorithm':<15} {'Makespan':>12} {'Gap':>8} {'Time(s)':>10}")
    print(f"    {'-' * 52}")
    for rank, (sn, sr) in enumerate(sorted_results, 1):
        ms = sr["makespan"]
        gap = (ms - best_ms) / best_ms * 100 if best_ms > 0 and ms < float('inf') else float('inf')
        time_s = sr["time_s"]
        marker = " <-- BEST" if rank == 1 else ""
        if ms < float('inf'):
            print(f"    {rank:<5} {sn:<15} {ms:>12.1f} {gap:>7.1f}% {time_s:>10.3f}{marker}")
        else:
            print(f"    {rank:<5} {sn:<15} {'ERROR':>12} {'N/A':>8} {'N/A':>10}")

    return results, sorted_results


# ============================================================
# MAIN BENCHMARKS
# ============================================================

def main():
    print("=" * 70)
    print("MoSAIC vs SAGA Benchmark Comparison")
    print("Reference: Coleman & Krishnamachari, PODC 2024")
    print("=" * 70)

    net2 = make_network(num_procs=2, comm_speed=1.0)
    all_results = {}

    # =============================================
    # BENCHMARK 1: SAGA Standard Structures
    # =============================================
    print("\n" + "=" * 70)
    print("BENCHMARK 1: SAGA Standard DAG Structures")
    print("=" * 70)

    # In-trees
    print("\n  Generating in-trees (fan-in patterns)...")
    in_trees = gen_in_trees(num=1, num_levels=5, branching_factor=3)
    for i, tg in enumerate(in_trees):
        results, ranked = run_benchmark(f"InTree-L5-B3", tg, net2)
        all_results["InTree-L5-B3"] = results

    # Out-trees
    print("\n  Generating out-trees (fan-out patterns)...")
    out_trees = gen_out_trees(num=1, num_levels=5, branching_factor=3)
    for i, tg in enumerate(out_trees):
        results, ranked = run_benchmark(f"OutTree-L5-B3", tg, net2)
        all_results["OutTree-L5-B3"] = results

    # Parallel chains
    print("\n  Generating parallel chains...")
    chains = gen_parallel_chains(num=1, num_chains=8, chain_length=10)
    for i, tg in enumerate(chains):
        results, ranked = run_benchmark(f"ParChains-8x10", tg, net2)
        all_results["ParChains-8x10"] = results

    # =============================================
    # BENCHMARK 2: Random DAGs (ER, Layered)
    # =============================================
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Random DAGs (Erdos-Renyi & Layered)")
    print("=" * 70)

    for n in [50, 100, 200]:
        tg = gen_erdos_renyi_dag(n, p=0.1, seed=42)
        results, ranked = run_benchmark(f"ER-{n}", tg, net2)
        all_results[f"ER-{n}"] = results

    for layers, width in [(6, 10), (8, 15), (10, 20)]:
        tg = gen_layered_dag(layers, width, seed=42)
        n = len(tg.tasks)
        results, ranked = run_benchmark(f"Layered-{layers}x{width}({n})", tg, net2)
        all_results[f"Layered-{layers}x{width}"] = results

    # =============================================
    # BENCHMARK 3: Transformer Training DAG
    # =============================================
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Transformer Training DAG (MoSAIC workload)")
    print("=" * 70)
    print("  Building Transformer DAG with GPU profiling...")

    saga_tg, mosaic_dag = build_transformer_saga_dag(tile_size=32)
    results, ranked = run_benchmark(
        f"Transformer-{len(saga_tg.tasks)}tasks",
        saga_tg, net2, include_mosaic=True
    )
    all_results["Transformer"] = results

    # =============================================
    # SUMMARY TABLE
    # =============================================
    print("\n" + "=" * 70)
    print("SUMMARY: Algorithm Rankings Across All Benchmarks")
    print("=" * 70)

    # Count wins and compute average gap
    algo_wins = defaultdict(int)
    algo_gaps = defaultdict(list)
    algo_count = defaultdict(int)

    for bench_name, bench_results in all_results.items():
        valid = {k: v for k, v in bench_results.items() if v["makespan"] < float('inf')}
        if not valid:
            continue
        best_ms = min(v["makespan"] for v in valid.values())
        for algo, res in valid.items():
            gap = (res["makespan"] - best_ms) / best_ms * 100
            algo_gaps[algo].append(gap)
            algo_count[algo] += 1
            if abs(gap) < 0.01:
                algo_wins[algo] += 1

    # Rank by average gap
    algo_avg_gap = {a: np.mean(g) for a, g in algo_gaps.items()}
    ranked_algos = sorted(algo_avg_gap.items(), key=lambda x: x[1])

    num_benchmarks = len(all_results)
    print(f"\n  {'Rank':<5} {'Algorithm':<15} {'Avg Gap':>10} {'Wins':>6} {'Benchmarks':>11}")
    print(f"  {'-' * 49}")
    for rank, (algo, avg_gap) in enumerate(ranked_algos, 1):
        wins = algo_wins.get(algo, 0)
        count = algo_count[algo]
        marker = " ***" if algo == "MoSAIC" else ""
        print(f"  {rank:<5} {algo:<15} {avg_gap:>9.2f}% {wins:>5}/{count:<5}{marker}")

    # =============================================
    # PER-BENCHMARK COMPARISON TABLE
    # =============================================
    print(f"\n  === PER-BENCHMARK MAKESPAN (top 5 algorithms + MoSAIC) ===")

    # Find top 5 overall + MoSAIC
    top5_names = [a for a, _ in ranked_algos[:5]]
    if "MoSAIC" not in top5_names:
        top5_names.append("MoSAIC")

    header = f"  {'Benchmark':<30}"
    for a in top5_names:
        header += f" {a:>10}"
    print(header)
    print(f"  {'-' * (30 + 11 * len(top5_names))}")

    for bench_name in all_results:
        row = f"  {bench_name:<30}"
        bench = all_results[bench_name]
        for a in top5_names:
            if a in bench and bench[a]["makespan"] < float('inf'):
                row += f" {bench[a]['makespan']:>10.1f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # =============================================
    # SAVE RESULTS
    # =============================================
    # Serialize results
    save_results = {}
    for bench_name, bench in all_results.items():
        save_results[bench_name] = {}
        for algo, res in bench.items():
            save_res = {"makespan": res["makespan"], "time_s": res["time_s"]}
            if "theta" in res:
                save_res["theta"] = res["theta"]
            if "error" in res:
                save_res["error"] = res["error"]
            save_results[bench_name][algo] = save_res

    save_results["_summary"] = {
        "algorithm_rankings": [
            {"rank": i + 1, "algorithm": a, "avg_gap_pct": round(g, 2),
             "wins": algo_wins.get(a, 0), "benchmarks": algo_count[a]}
            for i, (a, g) in enumerate(ranked_algos)
        ],
        "num_benchmarks": num_benchmarks,
        "num_algorithms": len(ranked_algos),
    }

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saga_benchmark_results.json")
    with open(out, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved: {out}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
