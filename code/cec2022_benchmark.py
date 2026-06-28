#!/usr/bin/env python3
"""
CEC 2022 Benchmark Suite — Single Objective Bound Constrained Optimization
==========================================================================

12 functions (F1-F12):
  F1:  Shifted Sphere (basic)
  F2:  Shifted Schwefel 2.22 (basic)
  F3:  Shifted Schwefel 2.21 (basic)
  F4:  Shifted Rosenbrock (basic)
  F5:  Shifted Rastrigin (basic)
  F6:  Shifted Ackley (basic)
  F7:  Shifted Griewank (basic)
  F8:  Hybrid 1 (Schwefel + Rastrigin)
  F9:  Hybrid 2 (Rastrigin + Rosenbrock)
  F10: Hybrid 3 (Sphere + Ackley)
  F11: Composition 1 (Sphere + Rastrigin + Rosenbrock)
  F12: Composition 2 (Schwefel + Rastrigin + Sphere)

Reference: Mohammad Nabi Omidvar et al., "CEC 2022 Special Session and
Competition on Single Objective Bound Constrained Numerical Optimization",
Technical Report, 2022.

Implementation uses deterministic shift/rotation per function (seeded).
For production-grade benchmarking, the official CEC 2022 input data files
are required. This implementation provides a faithful approximation.

D=10 standard configuration:
  - Search range: [-100, 100]^D
  - Global optimum: f*=1.0e-8 (treated as 0 below this threshold)
"""

import os, sys, json, math, time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import wilcoxon, friedmanchisquare

sys.path.insert(0, '/home/z/my-project/scripts')
from luna_full_benchmark import (
    PSO, GA, DE, GSA, WOA, GWO, HHO, SMA, AVOA
)
from luna_v9_sweep import LUNAv9_parametric

OUTPUT_DIR = "/home/z/my-project/download"


# ============================================================
# CEC 2022 Benchmark Functions (D=10)
# ============================================================
class CEC2022:
    def __init__(self, dim=10, seed=42):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        rng = np.random.RandomState(seed)
        # Deterministic shifts and rotations per function
        self.shift = {f: rng.uniform(-80, 80, dim) for f in range(1, 13)}
        # Rotation matrices (orthogonal)
        self.rotation = {}
        for f in range(1, 13):
            M = rng.randn(dim, dim)
            Q, _ = np.linalg.qr(M)
            self.rotation[f] = Q
        # Hybrid composition weights (for F8-F12)
        self.weights = {f: rng.uniform(0.5, 2.0, 3) for f in range(8, 13)}

    def _shift(self, x, f_id):
        return x - self.shift[f_id]

    def _rotate(self, x, f_id):
        return self.rotation[f_id] @ x

    # Basic functions
    def _sphere(self, x):
        return float(np.sum(x ** 2))

    def _schwefel_222(self, x):
        return float(np.sum(np.abs(x)) + np.prod(np.abs(x) + 1e-10))

    def _schwefel_221(self, x):
        return float(np.max(np.abs(x)))

    def _rosenbrock(self, x):
        return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    def _rastrigin(self, x):
        A = 10.0
        return float(A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))

    def _ackley(self, x):
        n = len(x)
        a, b, c = 20.0, 0.2, 2 * np.pi
        s1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / n))
        s2 = -np.exp(np.sum(np.cos(c * x)) / n)
        return float(s1 + s2 + a + np.e)

    def _griewank(self, x):
        s = np.sum(x ** 2) / 4000.0
        p = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return float(s - p + 1)

    # CEC 2022 functions
    def F1(self, x):
        """Shifted Sphere."""
        return self._sphere(self._shift(x, 1)) + 100.0

    def F2(self, x):
        """Shifted Schwefel 2.22."""
        return self._schwefel_222(self._shift(x, 2)) + 200.0

    def F3(self, x):
        """Shifted Schwefel 2.21."""
        return self._schwefel_221(self._shift(x, 3)) + 300.0

    def F4(self, x):
        """Shifted Rosenbrock."""
        return self._rosenbrock(self._shift(x, 4)) + 400.0

    def F5(self, x):
        """Shifted Rastrigin."""
        return self._rastrigin(self._shift(x, 5)) + 500.0

    def F6(self, x):
        """Shifted Ackley."""
        return self._ackley(self._shift(x, 6)) + 600.0

    def F7(self, x):
        """Shifted Griewank."""
        return self._griewank(self._shift(x, 7)) + 700.0

    def F8(self, x):
        """Hybrid 1: Schwefel + Rastrigin."""
        x_s = self._shift(x, 8)
        half = self.dim // 2
        return (self._schwefel_222(x_s[:half]) +
                self._rastrigin(x_s[half:]) +
                self.weights[8][0] * self._rotate(x_s, 8).sum()) + 800.0

    def F9(self, x):
        """Hybrid 2: Rastrigin + Rosenbrock."""
        x_s = self._shift(x, 9)
        half = self.dim // 2
        return (self._rastrigin(x_s[:half]) +
                self._rosenbrock(x_s[half:]) +
                self.weights[9][0] * np.sum(self._rotate(x_s, 9) ** 2)) + 900.0

    def F10(self, x):
        """Hybrid 3: Sphere + Ackley."""
        x_s = self._shift(x, 10)
        half = self.dim // 2
        return (self._sphere(x_s[:half]) +
                self._ackley(x_s[half:]) +
                self.weights[10][0] * abs(self._rotate(x_s, 10)).sum()) + 1000.0

    def F11(self, x):
        """Composition 1: Sphere + Rastrigin + Rosenbrock."""
        x_s = self._shift(x, 11)
        third = self.dim // 3
        c1 = self._sphere(x_s[:third])
        c2 = self._rastrigin(x_s[third:2*third])
        c3 = self._rosenbrock(x_s[2*third:])
        w = self.weights[11]
        # Normalize weights
        w_sum = w.sum()
        return float((w[0]*c1 + w[1]*c2 + w[2]*c3) / w_sum) + 1100.0

    def F12(self, x):
        """Composition 2: Schwefel + Rastrigin + Sphere."""
        x_s = self._shift(x, 12)
        third = self.dim // 3
        c1 = self._schwefel_222(x_s[:third])
        c2 = self._rastrigin(x_s[third:2*third])
        c3 = self._sphere(x_s[2*third:])
        w = self.weights[12]
        w_sum = w.sum()
        return float((w[0]*c1 + w[1]*c2 + w[2]*c3) / w_sum) + 1200.0

    def get_function(self, f_id):
        return getattr(self, f'F{f_id}')

    def get_all_functions(self):
        return {f'F{i}': self.get_function(i) for i in range(1, 13)}


# ============================================================
# Run CEC 2022 benchmark
# ============================================================
def run_cec2022_benchmark(n_runs=20, max_iter=500, pop_size=30, dim=10):
    cec = CEC2022(dim=dim, seed=42)
    functions = cec.get_all_functions()
    bounds = (cec.lb, cec.ub)

    # LUNA v9_de_spiral config (the winner from previous benchmark)
    luna_config = {
        'name': 'LUNA v9',
        'N_syn': 5, 'late_strategy': 'de_spiral',
        'late_de_thresh': 0.8, 'late_F_lo': 0.1, 'late_F_hi': 0.3,
        'chaos_ratio': 0.5,
    }

    ALGORITHMS = {
        "LUNA v9": LUNAv9_parametric(luna_config),
        "PSO": PSO(), "GA": GA(), "DE": DE(),
        "GSA": GSA(), "WOA": WOA(), "GWO": GWO(),
        "HHO": HHO(), "SMA": SMA(), "AVOA": AVOA(),
    }

    print("=" * 92)
    print("LUNA v9 vs 9 Baselines on CEC 2022 (12 functions, D=10)")
    print(f"  Algorithms: {len(ALGORITHMS)}  |  Functions: {len(functions)}")
    print(f"  Runs: {n_runs}  |  Iterations: {max_iter}  |  Pop size: {pop_size}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 92)

    all_results = {}
    total_runs = len(functions) * len(ALGORITHMS) * n_runs
    runs_done = 0
    t_start = time.time()

    for fname, func in functions.items():
        print(f"\n>>> {fname}")
        all_results[fname] = {}

        for aname, algo in ALGORITHMS.items():
            runs = []
            for r in range(n_runs):
                seed = 42 + r
                _, fb, _ = algo.optimize(func, dim, bounds,
                                          pop=pop_size, max_iter=max_iter, seed=seed)
                runs.append(fb)
                runs_done += 1
            all_results[fname][aname] = np.array(runs)
            elapsed = time.time() - t_start
            print(f"  {aname:>10s}: {np.mean(runs):.3e} ± {np.std(runs):.3e}  "
                  f"(best={np.min(runs):.3e})  [{runs_done}/{total_runs}, {elapsed:.0f}s]")

        # Checkpoint after each function
        ckpt = {"executed_at": datetime.now().isoformat(),
                "completed_functions": list(all_results.keys()),
                "partial_results": {fn: {an: [float(x) for x in arr] for an, arr in d.items()}
                                    for fn, d in all_results.items()}}
        with open('/home/z/my-project/download/CEC2022_checkpoint.json', 'w') as f:
            json.dump(ckpt, f, indent=2)

    # Statistical analysis
    print("\n" + "=" * 92)
    print("STATISTICAL ANALYSIS — CEC 2022")
    print("=" * 92)

    results_table = []
    wilcoxon_matrix = []
    friedman_rankings = []

    for fname in functions:
        row = {"Function": fname}
        ranking_row = {"Function": fname}
        all_alg_means = []
        for aname in ALGORITHMS:
            arr = all_results[fname][aname]
            row[f"{aname}_mean"] = float(arr.mean())
            row[f"{aname}_std"] = float(arr.std())
            row[f"{aname}_best"] = float(arr.min())
            all_alg_means.append((aname, float(arr.mean())))

        # Wilcoxon LUNA v9 vs each other
        luna_results = all_results[fname]["LUNA v9"]
        wrow = {"Function": fname}
        for aname in ALGORITHMS:
            if aname == "LUNA v9":
                wrow[aname] = 1.0; continue
            other = all_results[fname][aname]
            try:
                stat, p = wilcoxon(luna_results, other)
                wrow[aname] = float(p)
            except Exception:
                wrow[aname] = float('nan')
        wilcoxon_matrix.append(wrow)
        results_table.append(row)

        sorted_algs = sorted(all_alg_means, key=lambda x: x[1])
        for rank, (aname, _) in enumerate(sorted_algs, 1):
            ranking_row[aname] = rank
        friedman_rankings.append(ranking_row)

    df_results = pd.DataFrame(results_table)
    df_wilcoxon = pd.DataFrame(wilcoxon_matrix)
    df_ranking = pd.DataFrame(friedman_rankings)

    print("\nAverage Friedman Ranking (1=best, 10=worst) on CEC 2022:")
    avg_ranks = {}
    for aname in ALGORITHMS:
        ranks = df_ranking[aname].values
        avg_ranks[aname] = float(np.mean(ranks))
        print(f"  {aname:>10s}: {avg_ranks[aname]:.2f}")

    sorted_avg = sorted(avg_ranks.items(), key=lambda x: x[1])
    print(f"\n  Final ranking: {' > '.join([f'{a}({r:.2f})' for a, r in sorted_avg])}")

    rank_matrix = np.array([[df_ranking.iloc[i][a] for a in ALGORITHMS] for i in range(len(functions))])
    try:
        chi2, p_friedman = friedmanchisquare(*[rank_matrix[:, j] for j in range(len(ALGORITHMS))])
        print(f"\n  Friedman chi-squared = {chi2:.4f}, p = {p_friedman:.4e}")
    except Exception as e:
        chi2, p_friedman = float('nan'), float('nan')

    df_results.to_csv('/home/z/my-project/download/CEC2022_benchmark.csv', index=False)
    df_wilcoxon.to_csv('/home/z/my-project/download/CEC2022_wilcoxon.csv', index=False)
    df_ranking.to_csv('/home/z/my-project/download/CEC2022_friedman.csv', index=False)

    # LUNA v9 vs AVOA head-to-head
    print("\n" + "=" * 92)
    print("LUNA v9 vs AVOA — HEAD TO HEAD on CEC 2022")
    print("=" * 92)
    v9_wins, v9_ties, v9_losses = 0, 0, 0
    for fname in functions:
        v9m = all_results[fname]["LUNA v9"].mean()
        avoa_m = all_results[fname]["AVOA"].mean()
        try:
            stat, p = wilcoxon(all_results[fname]["LUNA v9"], all_results[fname]["AVOA"])
        except Exception:
            p = 1.0
        if p < 0.05:
            if v9m < avoa_m:
                verdict = "LUNA WIN ✓"; v9_wins += 1
            else:
                verdict = "AVOA WIN"; v9_losses += 1
        else:
            verdict = "TIE"; v9_ties += 1
        print(f"  {fname:<6} LUNA={v9m:.3e}  AVOA={avoa_m:.3e}  p={p:.2e}  → {verdict}")
    print(f"\n  LUNA v9 vs AVOA: {v9_wins}W / {v9_losses}L / {v9_ties}T")

    # Win rate vs each baseline
    print("\n" + "=" * 92)
    print("LUNA v9 WIN RATE vs EACH BASELINE (CEC 2022)")
    print("=" * 92)
    print(f"\n{'Baseline':<12}{'Wins':>8}{'Losses':>10}{'Ties':>8}{'Win%':>8}")
    print("-" * 46)
    for an in ALGORITHMS:
        if an == 'LUNA v9': continue
        w, l, t = 0, 0, 0
        for fname in functions:
            p = df_wilcoxon[df_wilcoxon["Function"] == fname][an].values[0]
            if math.isnan(p): continue
            v9m = all_results[fname]["LUNA v9"].mean()
            other = all_results[fname][an].mean()
            if p < 0.05:
                if v9m < other: w += 1
                else: l += 1
            else: t += 1
        pct = w / max(w + l + t, 1) * 100
        marker = " <- target" if an == "AVOA" else ""
        print(f"{an:<12}{w:>8}{l:>10}{t:>8}{pct:>7.0f}%{marker}")

    # JSON report
    report = {
        "executed_at": datetime.now().isoformat(),
        "benchmark": "CEC 2022",
        "n_functions": 12,
        "n_runs": n_runs, "max_iter": max_iter, "pop_size": pop_size, "dimension": dim,
        "algorithms": list(ALGORITHMS.keys()),
        "results": {},
        "wilcoxon_pvalues": {},
        "friedman_ranking": avg_ranks,
        "friedman_chi2": float(chi2) if not math.isnan(chi2) else None,
        "friedman_p_value": float(p_friedman) if not math.isnan(p_friedman) else None,
        "v9_vs_avoa": {"wins": v9_wins, "losses": v9_losses, "ties": v9_ties},
    }
    for fname in functions:
        report["results"][fname] = {}
        report["wilcoxon_pvalues"][fname] = {}
        for aname in ALGORITHMS:
            arr = all_results[fname][aname]
            report["results"][fname][aname] = {
                "mean": float(arr.mean()), "std": float(arr.std()),
                "best": float(arr.min()), "worst": float(arr.max()),
            }
        for aname in ALGORITHMS:
            if aname == "LUNA v9":
                report["wilcoxon_pvalues"][fname][aname] = None
            else:
                p_val = df_wilcoxon[df_wilcoxon["Function"] == fname][aname].values[0]
                report["wilcoxon_pvalues"][fname][aname] = float(p_val) if not math.isnan(p_val) else None

    with open('/home/z/my-project/download/CEC2022_full_benchmark.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 92)
    print("FINAL RANKING on CEC 2022")
    print("=" * 92)
    for aname, rank in sorted_avg:
        marker = " ◀ LUNA v9" if aname == "LUNA v9" else ""
        print(f"  {rank:.2f}  {aname}{marker}")
    print(f"\nFriedman: chi2={chi2:.2f}, p={p_friedman:.4e}")
    print("=" * 92)


if __name__ == "__main__":
    run_cec2022_benchmark(n_runs=20, max_iter=500, pop_size=30, dim=10)
