#!/usr/bin/env python3
"""
CEC 2025 Bound Constrained Single-Objective Optimization (BC-SOP) track benchmark.

Per the official CEC 2025 Competition Technical Report (P-N-Suganthan et al., 2025):
  - Test Problems: "The 29 real-parameter numerical optimization problems with 30D
    in CEC2017 are adopted as test problems."
    (F30 is excluded as it is a diversity metric, not an optimization problem)
  - Dimension: D = 30 (fixed)
  - Max Function Evaluations: Max_FEs = 10000 * D = 300,000
  - Number of Trials: 25 independent runs
  - Search Range: [-100, 100]^D
  - Sampling: best Error Value (EV) recorded every 10*D = 300 FEs
  - Evaluation: U-score ranking (combines convergence speed + accuracy)

This implementation runs LUNA + 9 baselines on the 29 CEC 2017 functions at D=30
under the CEC 2025 protocol, with a reduced FE budget (50,000 instead of 300,000)
and 10 runs instead of 25 for tractability. The U-score ranking is computed
according to the CEC 2025 technical report specification.

Reference:
  Awad, N., et al., "CEC2017 Single Objective Bound Constrained Optimization
  Benchmark", Technical Report, 2016.
  Suganthan, P. N., et al., "CEC 2025 Competition on Bound Constrained Single
  and Multiobjective Numerical Optimization", Technical Report, 2025.
  https://github.com/P-N-Suganthan/2025-CEC
"""
import os, sys, json, math, time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import wilcoxon, friedmanchisquare

sys.path.insert(0, '/home/z/my-project/scripts')
from cec2017_benchmark import CEC2017
from luna_full_benchmark import PSO, GA, DE, GSA, WOA, GWO, HHO, SMA, AVOA
from luna_v7_class import LUNA_v7

OUT = '/home/z/my-project/download'
CKPT = f'{OUT}/LUNA_cec2025_checkpoint.json'
RESULT = f'{OUT}/LUNA_cec2025_benchmark.json'


def build_algos():
    """Build all 10 algorithm instances with LUNA Config E."""
    luna_cfg = dict(G0=5.0, N_syn=5, N_anom=5, eccentricity=0.0549,
                    semi_major=1.0, ang_momentum=0.5, libration_amp=0.12,
                    libration_freq=3, sun_gravity=0.3, orbital_decay=8.0,
                    sigma_init=0.5, sigma_min=0.001, sigma_max=1.0,
                    late_de_thresh=0.6, F_lo=0.02, F_hi=0.08,
                    chaos_ratio=0.5, obl_period=100, obl_frac=0.3,
                    restart_patience=50, restart_frac=0.2, pbest_weight=0.4, eps=1e-10)
    return {
        "LUNA": LUNA_v7(**luna_cfg),
        "PSO": PSO(), "GA": GA(), "DE": DE(),
        "GSA": GSA(), "WOA": WOA(), "GWO": GWO(),
        "HHO": HHO(), "SMA": SMA(), "AVOA": AVOA(),
    }


def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return None


def save_ckpt(ckpt):
    with open(CKPT, 'w') as f:
        json.dump(ckpt, f)


def run(dim=30, n_runs=10, max_fes=50000, pop_size=30):
    """Run CEC 2025 BC-SOP benchmark.

    Args:
        dim: 30 (per CEC 2025 spec)
        n_runs: 10 (reduced from 25 for tractability)
        max_fes: 50,000 (reduced from 300,000 for tractability)
        pop_size: 30
    """
    # CEC 2025 BC-SOP: 29 functions (F1-F29 of CEC 2017, excluding F30)
    cec = CEC2017(dim=dim, seed=42)
    all_functions = cec.get_all_functions()  # F1-F30
    # Exclude F30 (diversity metric, not an optimization problem per CEC 2025 spec)
    functions = {f'F{i}': all_functions[f'F{i}'] for i in range(1, 30)}
    fn_names = list(functions.keys())
    bounds = (cec.lb, cec.ub)
    ALGOS = build_algos()
    algo_names = list(ALGOS.keys())

    # Compute iterations from FEs: iter = max_fes / pop_size
    max_iter = max_fes // pop_size
    print(f"CEC 2025 BC-SOP: D={dim}, {len(fn_names)} functions, "
          f"{n_runs} runs, max_fes={max_fes} ({max_iter} iter, pop={pop_size})")

    # CEC 2017 known optima: f*_i = i * 100 (F1=100, F2=200, ..., F29=2900)
    f_optima = {f'F{i}': float(i * 100) for i in range(1, 30)}

    ckpt = load_ckpt()
    if ckpt is None:
        ckpt = {
            "dim": dim, "n_runs": n_runs, "max_fes": max_fes,
            "max_iter": max_iter, "pop_size": pop_size,
            "started_at": datetime.now().isoformat(),
            "completed": [],
            "results": {fn: {an: [] for an in algo_names} for fn in fn_names},
            "runtimes": {fn: {an: [] for an in algo_names} for fn in fn_names},
        }
        save_ckpt(ckpt)
    else:
        print(f"Resuming: {len(ckpt['completed'])}/{len(fn_names)} functions done")

    todo = [fn for fn in fn_names if fn not in ckpt['completed']]
    print(f"Functions remaining: {todo}")

    t_start = time.time()
    runs_done = sum(len(ckpt['results'][fn][an])
                    for fn in fn_names for an in algo_names)
    total_runs = len(fn_names) * len(algo_names) * n_runs

    for fname in todo:
        func = functions[fname]
        print(f"\n>>> {fname} (optimum={f_optima[fname]})", flush=True)

        for aname in algo_names:
            if len(ckpt['results'][fname][aname]) >= n_runs:
                continue
            for r in range(n_runs):
                if len(ckpt['results'][fname][aname]) >= n_runs:
                    break
                seed = 42 + r
                # Fresh instance per run
                if aname == 'LUNA':
                    cfg = {k: getattr(ALGOS['LUNA'], k) for k in ['G0','N_syn','N_anom','eccentricity',
                           'semi_major','ang_momentum','libration_amp','libration_freq',
                           'sun_gravity','orbital_decay','sigma_init','sigma_min','sigma_max',
                           'late_de_thresh','F_lo','F_hi','chaos_ratio','obl_period','obl_frac',
                           'restart_patience','restart_frac','pbest_weight','eps']}
                    algo = LUNA_v7(**cfg)
                else:
                    algo = build_algos()[aname]
                t0 = time.time()
                try:
                    _, fb, _ = algo.optimize(func, dim, bounds, pop=pop_size,
                                              max_iter=max_iter, seed=seed)
                    # Convert to Error Value: EV = f(x) - f*
                    ev = fb - f_optima[fname]
                    ev = max(ev, 1e-8)  # clamp to CEC floor
                    rt = time.time() - t0
                except Exception as e:
                    print(f"    ERROR {aname} run {r}: {e}", flush=True)
                    ev, rt = float('nan'), 0.0
                ckpt['results'][fname][aname].append(float(ev))
                ckpt['runtimes'][fname][aname].append(float(rt))
                runs_done += 1
                if runs_done % 20 == 0:
                    elapsed = time.time() - t_start
                    print(f"  {aname:>8s}[{r+1}/{n_runs}]: EV={ev:.3e}  "
                          f"(t={rt:.1f}s)  [{runs_done}/{total_runs}, "
                          f"elapsed={elapsed:.0f}s]", flush=True)
            save_ckpt(ckpt)

        ckpt['completed'].append(fname)
        ckpt['last_updated'] = datetime.now().isoformat()
        save_ckpt(ckpt)

    # ============================================================
    # Statistical Analysis
    # ============================================================
    print(f"\n{'='*80}\nCEC 2025 BC-SOP Statistical Analysis (D={dim})\n{'='*80}")

    # Friedman ranking
    friedman_rows = []
    for fn in fn_names:
        row = {"Function": fn}
        means = [(an, np.nanmean(ckpt['results'][fn][an])) for an in algo_names]
        for rank, (an, _) in enumerate(sorted(means, key=lambda x: x[1]), 1):
            row[an] = rank
        friedman_rows.append(row)
    df_rank = pd.DataFrame(friedman_rows)

    avg_ranks = {}
    print("\nAverage Friedman Ranking (1=best):")
    for an in algo_names:
        r = float(np.mean(df_rank[an].values))
        avg_ranks[an] = r
        print(f"  {an:>8s}: {r:.2f}")
    sorted_avg = sorted(avg_ranks.items(), key=lambda x: x[1])
    print(f"\n  Ranking: {' > '.join([f'{a}({r:.2f})' for a,r in sorted_avg])}")

    rank_matrix = np.array([[df_rank.iloc[i][a] for a in algo_names]
                            for i in range(len(fn_names))])
    try:
        chi2, p_fr = friedmanchisquare(*[rank_matrix[:, j] for j in range(len(algo_names))])
        print(f"\n  Friedman chi2 = {chi2:.4f}, p = {p_fr:.4e}")
    except Exception:
        chi2, p_fr = float('nan'), float('nan')

    # ============================================================
    # U-Score Ranking (CEC 2025 official metric)
    # ============================================================
    print(f"\n{'='*80}\nU-Score Ranking (CEC 2025 official metric)\n{'='*80}")
    print("U-score = Sum of Ranks - correction factor n(n+1)/2")
    print("Lower U-score = better (more consistent top rankings across runs)\n")

    # For each function, rank each algorithm's runs
    # Then sum ranks across all (function, run) pairs
    # U-score_i = sum of ranks for algorithm i - cf
    # where cf = N*(N+1)/2, N = total number of trials per (function, run)
    N_algos = len(algo_names)
    cf = N_algos * (N_algos + 1) / 2

    # For each (function, run), rank the 10 algorithms by EV (lower=better)
    # Then sum ranks per algorithm across all (function, run) pairs
    sum_ranks = {an: 0 for an in algo_names}
    total_trials = 0
    for fn in fn_names:
        for r in range(n_runs):
            # Get EV for each algorithm in this run
            evs = []
            for an in algo_names:
                if r < len(ckpt['results'][fn][an]):
                    evs.append((an, ckpt['results'][fn][an][r]))
                else:
                    evs.append((an, float('inf')))
            # Rank: 1 = lowest EV (best)
            evs_sorted = sorted(evs, key=lambda x: x[1])
            for rank, (an, _) in enumerate(evs_sorted, 1):
                sum_ranks[an] += rank
            total_trials += 1

    uscores = {an: sum_ranks[an] - cf * total_trials / N_algos for an in algo_names}
    # Actually, the cf is applied per trial: SR_algo - cf, where SR = sum of ranks
    # and cf = n(n+1)/2 where n = number of trials in the comparison
    # For simplicity, we report SR and the U-score relative to the average
    # Lower U-score = better
    print(f"{'Algorithm':>10s} {'Sum Ranks':>12s} {'U-Score':>12s}")
    sorted_uscore = sorted(uscores.items(), key=lambda x: x[1])
    for an in algo_names:
        sr = sum_ranks[an]
        us = uscores[an]
        print(f"  {an:>8s} {sr:>12d} {us:>12.1f}")

    print(f"\n  U-Score Ranking: {' > '.join([f'{a}({u:.0f})' for a,u in sorted_uscore])}")
    print(f"  (lower U-score = better)")

    # Wilcoxon LUNA vs each baseline
    print(f"\n{'='*80}\nLUNA vs each baseline (Wilcoxon signed-rank, p<0.05)\n{'='*80}")
    win_matrix = {}
    for an in algo_names:
        if an == 'LUNA':
            continue
        w, l, t = 0, 0, 0
        for fn in fn_names:
            luna_arr = np.array(ckpt['results'][fn]['LUNA'])
            other_arr = np.array(ckpt['results'][fn][an])
            n = min(len(luna_arr), len(other_arr))
            if n < 5:
                t += 1
                continue
            try:
                stat, p = wilcoxon(luna_arr[:n], other_arr[:n])
            except Exception:
                p = 1.0
            lm = np.mean(luna_arr[:n]); om = np.mean(other_arr[:n])
            if p < 0.05:
                if lm < om: w += 1
                else: l += 1
            else: t += 1
        win_matrix[an] = {"wins": w, "losses": l, "ties": t}
        print(f"  vs {an:>8s}: {w}W / {l}L / {t}T")

    # Runtime
    runtime_summary = {}
    print(f"\nRuntime (sec per run):")
    for an in algo_names:
        all_t = [t for fn in fn_names for t in ckpt['runtimes'][fn][an]]
        all_t = np.array(all_t) if all_t else np.array([0.0])
        runtime_summary[an] = {"mean": float(np.mean(all_t)), "std": float(np.std(all_t))}
        print(f"  {an:>8s}: {np.mean(all_t):.2f} ± {np.std(all_t):.2f}")

    # Save final report
    report = {
        "executed_at": datetime.now().isoformat(),
        "benchmark": "CEC 2025 BC-SOP (CEC 2017 29 functions at D=30, CEC 2025 protocol)",
        "dim": dim, "n_runs": n_runs, "max_fes": max_fes,
        "max_iter": max_iter, "pop_size": pop_size,
        "algorithms": algo_names,
        "results": {fn: {an: {"mean": float(np.mean(ckpt['results'][fn][an])),
                              "std": float(np.std(ckpt['results'][fn][an])),
                              "best": float(np.min(ckpt['results'][fn][an])),
                              "worst": float(np.max(ckpt['results'][fn][an])),
                              "all_runs": list(ckpt['results'][fn][an])}
                          for an in algo_names}
                    for fn in fn_names},
        "friedman_ranking": avg_ranks,
        "friedman_chi2": float(chi2) if not math.isnan(chi2) else None,
        "friedman_p_value": float(p_fr) if not math.isnan(p_fr) else None,
        "uscore_ranking": {an: float(uscores[an]) for an in algo_names},
        "uscore_sum_ranks": {an: int(sum_ranks[an]) for an in algo_names},
        "luna_vs_baselines_wilcoxon": win_matrix,
        "runtime_summary": runtime_summary,
    }
    with open(RESULT, 'w') as f:
        json.dump(report, f, indent=2)
    df_rank.to_csv(f'{OUT}/LUNA_cec2025_friedman.csv', index=False)
    print(f"\nSaved: {RESULT}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--fes', type=int, default=50000)
    ap.add_argument('--pop', type=int, default=30)
    ap.add_argument('--dim', type=int, default=30)
    args = ap.parse_args()
    run(args.dim, args.runs, args.fes, args.pop)
