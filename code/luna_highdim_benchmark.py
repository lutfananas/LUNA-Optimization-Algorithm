#!/usr/bin/env python3
"""
High-dimensional CEC 2022 benchmark for LUNA vs baselines.
Supports checkpoint/resume so it can survive sandbox timeouts.

Usage:
  python luna_highdim_benchmark.py --dim 50 --runs 10 --iter 200
  python luna_highdim_benchmark.py --dim 100 --runs 5 --iter 150
"""
import os, sys, json, math, time, argparse, signal
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import wilcoxon, friedmanchisquare

sys.path.insert(0, '/home/z/my-project/scripts')
from cec2022_benchmark import CEC2022
from luna_full_benchmark import PSO, GA, DE, GSA, WOA, GWO, HHO, SMA, AVOA
from luna_v7_class import LUNA_v7

OUT = '/home/z/my-project/download'
CKPT_TEMPLATE = '{out}/LUNA_highdim_D{dim}_checkpoint.json'
RESULT_TEMPLATE = '{out}/LUNA_highdim_D{dim}_benchmark.json'


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


def load_checkpoint(dim):
    path = CKPT_TEMPLATE.format(out=OUT, dim=dim)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_checkpoint(dim, ckpt):
    path = CKPT_TEMPLATE.format(out=OUT, dim=dim)
    with open(path, 'w') as f:
        json.dump(ckpt, f, indent=2)


def run_single(algo, func, dim, bounds, pop, max_iter, seed):
    """Run one algorithm once and return (best, runtime)."""
    # Fresh instance to reset internal state
    if isinstance(algo, LUNA_v7):
        cfg = {k: getattr(algo, k) for k in ['G0','N_syn','N_anom','eccentricity',
               'semi_major','ang_momentum','libration_amp','libration_freq',
               'sun_gravity','orbital_decay','sigma_init','sigma_min','sigma_max',
               'late_de_thresh','F_lo','F_hi','chaos_ratio','obl_period','obl_frac',
               'restart_patience','restart_frac','pbest_weight','eps']}
        algo = LUNA_v7(**cfg)
    t0 = time.time()
    _, fb, _ = algo.optimize(func, dim, bounds, pop=pop, max_iter=max_iter, seed=seed)
    return float(fb), time.time() - t0


def run(dim, n_runs, max_iter, pop_size):
    cec = CEC2022(dim=dim, seed=42)
    functions = cec.get_all_functions()
    bounds = (cec.lb, cec.ub)
    fn_names = list(functions.keys())
    ALGOS = build_algos()
    algo_names = list(ALGOS.keys())

    # Load checkpoint
    ckpt = load_checkpoint(dim)
    if ckpt is None:
        ckpt = {
            "dimension": dim,
            "n_runs": n_runs, "max_iter": max_iter, "pop_size": pop_size,
            "started_at": datetime.now().isoformat(),
            "completed": [],
            "results": {fn: {an: [] for an in algo_names} for fn in fn_names},
            "runtimes": {fn: {an: [] for an in algo_names} for fn in fn_names},
        }
        save_checkpoint(dim, ckpt)
    else:
        print(f"Resuming D={dim}: {len(ckpt['completed'])}/{len(fn_names)} functions done")

    # Find next function to do
    todo = [fn for fn in fn_names if fn not in ckpt['completed']]
    if not todo:
        print(f"All functions done for D={dim}; building final report.")
    else:
        print(f"Functions remaining for D={dim}: {todo}")

    total_runs = len(fn_names) * len(algo_names) * n_runs
    runs_done = sum(len(ckpt['results'][fn][an])
                    for fn in fn_names for an in algo_names)
    t_start = time.time()

    for fname in todo:
        func = functions[fname]
        print(f"\n>>> D={dim} {fname}", flush=True)

        for aname in algo_names:
            # Skip if already has enough runs
            if len(ckpt['results'][fname][aname]) >= n_runs:
                continue
            algo = build_algos()[aname]  # fresh instance per algorithm
            for r in range(n_runs):
                if len(ckpt['results'][fname][aname]) >= n_runs:
                    break
                seed = 42 + r
                try:
                    fb, rt = run_single(algo, func, dim, bounds, pop_size, max_iter, seed)
                except Exception as e:
                    print(f"    ERROR {aname} run {r}: {e}", flush=True)
                    fb, rt = float('nan'), 0.0
                ckpt['results'][fname][aname].append(fb)
                ckpt['runtimes'][fname][aname].append(rt)
                runs_done += 1
                elapsed = time.time() - t_start
                print(f"  {aname:>8s}[{r+1}/{n_runs}]: {fb:.3e}  "
                      f"(t={rt:.1f}s)  [{runs_done}/{total_runs}, "
                      f"elapsed={elapsed:.0f}s]", flush=True)
            # Save checkpoint after each algo finishes
            save_checkpoint(dim, ckpt)

        ckpt['completed'].append(fname)
        ckpt['last_updated'] = datetime.now().isoformat()
        save_checkpoint(dim, ckpt)

    # Build final report
    print(f"\n{'='*80}\nSTATISTICAL ANALYSIS - CEC 2022 D={dim}\n{'='*80}")

    # Friedman rankings
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

    # LUNA vs each baseline (Wilcoxon)
    print("\nLUNA vs each baseline (Wilcoxon signed-rank, p<0.05):")
    win_matrix = {}
    for an in algo_names:
        if an == 'LUNA':
            continue
        w, l, t = 0, 0, 0
        for fn in fn_names:
            luna_arr = np.array(ckpt['results'][fn]['LUNA'])
            other_arr = np.array(ckpt['results'][fn][an])
            # Truncate to same length
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

    # Runtime summary
    print("\nRuntime (sec per run):")
    runtime_summary = {}
    for an in algo_names:
        all_t = [t for fn in fn_names for t in ckpt['runtimes'][fn][an]]
        all_t = np.array(all_t) if all_t else np.array([0.0])
        runtime_summary[an] = {"mean": float(np.mean(all_t)),
                                "std": float(np.std(all_t))}
        print(f"  {an:>8s}: {np.mean(all_t):.2f} ± {np.std(all_t):.2f}")

    # Save final
    report = {
        "executed_at": datetime.now().isoformat(),
        "algorithm_type": "astronomy_embedded_v7",
        "dimension": dim,
        "n_runs": n_runs, "max_iter": max_iter, "pop_size": pop_size,
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
        "luna_vs_baselines_wilcoxon": win_matrix,
        "runtime_summary": runtime_summary,
    }
    with open(RESULT_TEMPLATE.format(out=OUT, dim=dim), 'w') as f:
        json.dump(report, f, indent=2)
    df_rank.to_csv(f'{OUT}/LUNA_highdim_D{dim}_friedman.csv', index=False)
    print(f"\nSaved: {RESULT_TEMPLATE.format(out=OUT, dim=dim)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, required=True)
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--iter', type=int, default=200)
    ap.add_argument('--pop', type=int, default=30)
    args = ap.parse_args()
    run(args.dim, args.runs, args.iter, args.pop)
