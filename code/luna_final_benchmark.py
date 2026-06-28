#!/usr/bin/env python3
"""Full CEC 2022 benchmark with LUNA v7 Config E (astronomy-embedded operators)."""
import os, sys, json, math, time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import wilcoxon, friedmanchisquare

sys.path.insert(0, '/home/z/my-project/scripts')
from cec2022_benchmark import CEC2022
from luna_full_benchmark import PSO, GA, DE, GSA, WOA, GWO, HHO, SMA, AVOA
from luna_v7_class import LUNA_v7

OUT = '/home/z/my-project/download'

def run(dim=10, n_runs=20, max_iter=500, pop_size=30):
    cec = CEC2022(dim=dim, seed=42)
    functions = cec.get_all_functions()
    bounds = (cec.lb, cec.ub)

    # Config E: best from tuning
    luna_cfg = dict(G0=5.0, N_syn=5, N_anom=5, eccentricity=0.0549,
                    semi_major=1.0, ang_momentum=0.5, libration_amp=0.12,
                    libration_freq=3, sun_gravity=0.3, orbital_decay=8.0,
                    sigma_init=0.5, sigma_min=0.001, sigma_max=1.0,
                    late_de_thresh=0.6, F_lo=0.02, F_hi=0.08,
                    chaos_ratio=0.5, obl_period=100, obl_frac=0.3,
                    restart_patience=50, restart_frac=0.2, pbest_weight=0.4, eps=1e-10)

    ALGOS = {
        "LUNA": LUNA_v7(**luna_cfg),
        "PSO": PSO(), "GA": GA(), "DE": DE(),
        "GSA": GSA(), "WOA": WOA(), "GWO": GWO(),
        "HHO": HHO(), "SMA": SMA(), "AVOA": AVOA(),
    }

    print("=" * 92)
    print(f"LUNA (Astronomy-Embedded) vs 9 Baselines on CEC 2022 (D={dim})")
    print(f"  Algorithms: {len(ALGOS)} | Functions: {len(functions)}")
    print(f"  Runs: {n_runs} | Iterations: {max_iter} | Pop: {pop_size}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 92)

    all_results = {}
    all_runtimes = {}
    total_runs = len(functions) * len(ALGOS) * n_runs
    runs_done = 0
    t_start = time.time()

    for fname, func in functions.items():
        print(f"\n>>> {fname}")
        all_results[fname] = {}
        all_runtimes[fname] = {}

        for aname, algo in ALGOS.items():
            runs = []; rts = []
            for r in range(n_runs):
                seed = 42 + r
                t0 = time.time()
                _, fb, _ = algo.optimize(func, dim, bounds,
                                          pop=pop_size, max_iter=max_iter, seed=seed)
                elapsed = time.time() - t0
                runs.append(fb); rts.append(elapsed)
                runs_done += 1
            all_results[fname][aname] = np.array(runs)
            all_runtimes[fname][aname] = np.array(rts)
            elapsed_total = time.time() - t_start
            print(f"  {aname:>8s}: {np.mean(runs):.3e} +/- {np.std(runs):.3e}  "
                  f"(best={np.min(runs):.3e}, t={np.mean(rts):.1f}s)  "
                  f"[{runs_done}/{total_runs}, {elapsed_total:.0f}s]")

        # Checkpoint
        ckpt = {"executed_at": datetime.now().isoformat(), "dimension": dim,
                "completed_functions": list(all_results.keys()),
                "partial_results": {fn: {an: [float(x) for x in arr] for an, arr in d.items()}
                                    for fn, d in all_results.items()},
                "partial_runtimes": {fn: {an: [float(x) for x in arr] for an, arr in d.items()}
                                     for fn, d in all_runtimes.items()}}
        with open(f'{OUT}/LUNA_final_D{dim}_checkpoint.json', 'w') as f:
            json.dump(ckpt, f, indent=2)

    # Stats
    print("\n" + "=" * 92)
    print(f"STATISTICAL ANALYSIS - CEC 2022 D={dim}")
    print("=" * 92)

    friedman_rankings = []
    for fname in functions:
        ranking_row = {"Function": fname}
        all_alg_means = [(aname, all_results[fname][aname].mean()) for aname in ALGOS]
        sorted_algs = sorted(all_alg_means, key=lambda x: x[1])
        for rank, (aname, _) in enumerate(sorted_algs, 1):
            ranking_row[aname] = rank
        friedman_rankings.append(ranking_row)

    df_ranking = pd.DataFrame(friedman_rankings)
    avg_ranks = {}
    print("\nAverage Friedman Ranking (1=best):")
    for aname in ALGOS:
        ranks = df_ranking[aname].values
        avg_ranks[aname] = float(np.mean(ranks))
        print(f"  {aname:>8s}: {avg_ranks[aname]:.2f}")

    sorted_avg = sorted(avg_ranks.items(), key=lambda x: x[1])
    print(f"\n  Ranking: {' > '.join([f'{a}({r:.2f})' for a, r in sorted_avg])}")

    rank_matrix = np.array([[df_ranking.iloc[i][a] for a in ALGOS] for i in range(len(functions))])
    try:
        chi2, p_friedman = friedmanchisquare(*[rank_matrix[:, j] for j in range(len(ALGOS))])
        print(f"\n  Friedman chi2 = {chi2:.4f}, p = {p_friedman:.4e}")
    except:
        chi2, p_friedman = float('nan'), float('nan')

    # Runtime
    print("\nRUNTIME (seconds per run):")
    print(f"{'Algorithm':>10s} {'Avg':>8s} {'Std':>8s}")
    runtime_summary = {}
    for aname in ALGOS:
        all_t = []
        for fn in functions:
            all_t.extend(all_runtimes[fn][aname])
        all_t = np.array(all_t)
        runtime_summary[aname] = {"mean": float(np.mean(all_t)), "std": float(np.std(all_t))}
        print(f"  {aname:>8s} {np.mean(all_t):>8.2f} {np.std(all_t):>8.2f}")

    # LUNA vs AVOA
    print("\nLUNA vs AVOA:")
    wins, ties, losses = 0, 0, 0
    for fname in functions:
        lm = all_results[fname]["LUNA"].mean()
        am = all_results[fname]["AVOA"].mean()
        try:
            stat, p = wilcoxon(all_results[fname]["LUNA"], all_results[fname]["AVOA"])
        except:
            p = 1.0
        if p < 0.05:
            if lm < am: verdict = "LUNA WIN"; wins += 1
            else: verdict = "AVOA WIN"; losses += 1
        else: verdict = "TIE"; ties += 1
        print(f"  {fname:<6} LUNA={lm:.3e}  AVOA={am:.3e}  p={p:.2e}  -> {verdict}")
    print(f"\n  LUNA vs AVOA: {wins}W / {losses}L / {ties}T")

    # Win rate vs all baselines
    print("\nWIN RATE vs EACH BASELINE:")
    print(f"{'Baseline':>10s} {'Wins':>6s} {'Loss':>6s} {'Ties':>6s} {'Win%':>6s}")
    for an in ALGOS:
        if an == 'LUNA': continue
        w, l, t = 0, 0, 0
        for fn in functions:
            p = None
            try:
                stat, p = wilcoxon(all_results[fn]["LUNA"], all_results[fn][an])
            except: p = 1.0
            lm = all_results[fn]["LUNA"].mean(); om = all_results[fn][an].mean()
            if p < 0.05:
                if lm < om: w += 1
                else: l += 1
            else: t += 1
        print(f"  {an:>8s} {w:>6d} {l:>6d} {t:>6d} {w/(w+l+t)*100:>5.0f}%")

    # Save
    report = {
        "executed_at": datetime.now().isoformat(),
        "algorithm_type": "astronomy_embedded",
        "dimension": dim,
        "luna_config": luna_cfg,
        "algorithms": list(ALGOS.keys()),
        "results": {},
        "friedman_ranking": avg_ranks,
        "friedman_chi2": float(chi2) if not math.isnan(chi2) else None,
        "friedman_p_value": float(p_friedman) if not math.isnan(p_friedman) else None,
        "runtime_summary": runtime_summary,
        "luna_vs_avoa": {"wins": wins, "losses": losses, "ties": ties},
    }
    for fname in functions:
        report["results"][fname] = {}
        for aname in ALGOS:
            arr = all_results[fname][aname]
            report["results"][fname][aname] = {
                "mean": float(arr.mean()), "std": float(arr.std()),
                "best": float(arr.min()), "worst": float(arr.max()),
            }

    with open(f'{OUT}/LUNA_final_D{dim}_benchmark.json', 'w') as f:
        json.dump(report, f, indent=2)

    df_results = pd.DataFrame([{**{"Function": fn},
                                **{f"{an}_mean": all_results[fn][an].mean() for an in ALGOS}}
                               for fn in functions])
    df_results.to_csv(f'{OUT}/LUNA_final_D{dim}_benchmark.csv', index=False)
    df_ranking.to_csv(f'{OUT}/LUNA_final_D{dim}_friedman.csv', index=False)

    print(f"\nSaved: {OUT}/LUNA_final_D{dim}_benchmark.json")

if __name__ == "__main__":
    run(dim=10, n_runs=20, max_iter=500, pop_size=30)
