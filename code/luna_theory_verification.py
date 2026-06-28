#!/usr/bin/env python3
"""
Empirical verification of theoretical predictions:
1. Measure empirical hitting time on CEC 2022 D=10 (12 functions, 20 runs)
2. Compute effect size (Cliff's delta) for LUNA vs DE
3. Plot hitting-time probability vs iteration
"""
import os, sys, json, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
from scipy.stats import rankdata

fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

sys.path.insert(0, '/home/z/my-project/scripts')
from cec2022_benchmark import CEC2022
from luna_v7_class import LUNA_v7
from luna_full_benchmark import PSO, GA, DE, GSA, WOA, GWO, HHO, SMA, AVOA

OUT = '/home/z/my-project/download'

BASE_CFG = dict(G0=5.0, N_syn=5, N_anom=5, eccentricity=0.0549,
                semi_major=1.0, ang_momentum=0.5, libration_amp=0.12,
                libration_freq=3, sun_gravity=0.3, orbital_decay=8.0,
                sigma_init=0.5, sigma_min=0.001, sigma_max=1.0,
                late_de_thresh=0.6, F_lo=0.02, F_hi=0.08,
                chaos_ratio=0.5, obl_period=100, obl_frac=0.3,
                restart_patience=50, restart_frac=0.2, pbest_weight=0.4, eps=1e-10)


# ============================================================
# PART 1: Hitting Time Experiments
# ============================================================
def run_hitting_time(dim=10, n_runs=20, max_iter=500, pop_size=30,
                     epsilons=None):
    """Measure empirical hitting time for various epsilon thresholds.

    For each (function, run), record the iteration at which best-so-far
    first enters the epsilon-basin: f(x_best) <= f_opt + epsilon, where
    f_opt is the known optimum for CEC 2022 (100, 200, ..., 1200).
    """
    if epsilons is None:
        # CEC 2022 optima are 100, 200, ..., 1200
        epsilons = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    cec = CEC2022(dim=dim, seed=42)
    functions = cec.get_all_functions()
    bounds = (cec.lb, cec.ub)
    f_optima = {f'F{i}': float(i * 100) for i in range(1, 13)}

    # Track best-so-far at each iteration
    # We'll modify LUNA to record full history
    ckpt_path = f'{OUT}/LUNA_hitting_time_checkpoint.json'
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            results = json.load(f)
        print(f"Loaded checkpoint: {len(results.get('functions_done', []))} functions done")
    else:
        results = {
            'dim': dim, 'n_runs': n_runs, 'max_iter': max_iter,
            'epsilons': epsilons,
            'functions_done': [],
            'histories': {},  # histories[fn][run] = list of best-so-far
            'hitting_times': {},  # hitting_times[fn][run][eps] = iteration
        }

    fn_names = list(functions.keys())
    todo = [fn for fn in fn_names if fn not in results['functions_done']]
    print(f"Todo: {todo}")

    t_start = time.time()
    for fname in todo:
        func = functions[fname]
        f_opt = f_optima[fname]
        print(f"\n>>> {fname} (optimum={f_opt})", flush=True)

        results['histories'][fname] = {}
        results['hitting_times'][fname] = {}

        for r in range(n_runs):
            seed = 42 + r
            algo = LUNA_v7(**BASE_CFG)
            _, fb, history = algo.optimize(func, dim, bounds,
                                            pop=pop_size, max_iter=max_iter, seed=seed)
            # history[i] = best-so-far at iteration i
            # Compute hitting time for each epsilon
            ht_dict = {}
            for eps in epsilons:
                threshold = f_opt + eps
                hit_t = None
                for t, v in enumerate(history):
                    if v <= threshold:
                        hit_t = t
                        break
                ht_dict[str(eps)] = hit_t  # None means never hit
            results['histories'][fname][str(r)] = history
            results['hitting_times'][fname][str(r)] = ht_dict
            if (r + 1) % 5 == 0:
                elapsed = time.time() - t_start
                print(f"  Run {r+1}/{n_runs} done (elapsed={elapsed:.0f}s)", flush=True)

        results['functions_done'].append(fname)
        with open(ckpt_path, 'w') as f:
            json.dump(results, f)

    # Compute statistics
    print(f"\n{'='*80}\nHitting Time Statistics (D={dim})\n{'='*80}")
    print(f"{'Function':<8}", end='')
    for eps in epsilons:
        print(f"  eps={eps:<8g}", end='')
    print()

    summary = {}
    for fname in fn_names:
        summary[fname] = {}
        print(f"{fname:<8}", end='')
        for eps in epsilons:
            times = []
            for r in range(n_runs):
                ht = results['hitting_times'][fname][str(r)].get(str(eps))
                if ht is not None:
                    times.append(ht)
            if times:
                mean_ht = np.mean(times)
                hit_prob = len(times) / n_runs
                summary[fname][str(eps)] = {
                    'mean': mean_ht, 'std': np.std(times),
                    'hit_prob': hit_prob, 'n_hit': len(times)
                }
                print(f"  {mean_ht:>5.1f}({int(hit_prob*100):>2d}%)", end='')
            else:
                summary[fname][str(eps)] = {'mean': None, 'hit_prob': 0.0}
                print(f"  never(0%)", end='')
        print()

    # Overall mean hitting time across all functions
    print(f"\nOverall (across {len(fn_names)} functions):")
    overall = {}
    for eps in epsilons:
        means = []
        hit_probs = []
        for fn in fn_names:
            s = summary[fn].get(str(eps), {})
            if s.get('mean') is not None:
                means.append(s['mean'])
            hit_probs.append(s.get('hit_prob', 0))
        overall[str(eps)] = {
            'mean_hitting_time': float(np.mean(means)) if means else None,
            'mean_hit_prob': float(np.mean(hit_probs)),
        }
        print(f"  eps={eps:<8g}: mean HT={overall[str(eps)]['mean_hitting_time']}, "
              f"avg hit prob={overall[str(eps)]['mean_hit_prob']:.3f}")

    # Save final summary
    final = {
        'executed_at': datetime.now().isoformat(),
        'dim': dim, 'n_runs': n_runs, 'max_iter': max_iter,
        'epsilons': epsilons,
        'per_function': summary,
        'overall': overall,
    }
    with open(f'{OUT}/LUNA_hitting_time.json', 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved: {OUT}/LUNA_hitting_time.json")
    return final


# ============================================================
# PART 2: Plot Hitting Time vs Iteration Probability
# ============================================================
def plot_hitting_time_probability(hitting_data):
    """Plot fraction of runs that have entered the epsilon-basin by iteration t."""
    epsilons = hitting_data['epsilons']
    fn_names = list(hitting_data['per_function'].keys())
    n_runs = hitting_data['n_runs']
    max_iter = hitting_data['max_iter']

    # Load full histories
    with open(f'{OUT}/LUNA_hitting_time_checkpoint.json') as f:
        ckpt = json.load(f)

    f_optima = {f'F{i}': float(i * 100) for i in range(1, 13)}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes = axes.flatten()
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(epsilons)))

    for ax_idx, eps in enumerate(epsilons):
        ax = axes[ax_idx]
        # For each iteration t, compute fraction of (function, run) pairs
        # where best-so-far at t is <= f_opt + eps
        all_curves = []  # one curve per function
        for fn in fn_names:
            curve = np.zeros(max_iter + 1)
            f_opt = f_optima[fn]
            threshold = f_opt + eps
            for r in range(n_runs):
                hist = ckpt['histories'][fn][str(r)]
                # Pad history if needed
                if len(hist) < max_iter + 1:
                    hist = hist + [hist[-1]] * (max_iter + 1 - len(hist))
                for t in range(max_iter + 1):
                    if hist[t] <= threshold:
                        curve[t:] += 1
                        break
            curve = curve / n_runs
            all_curves.append(curve)

        # Average across functions
        mean_curve = np.mean(all_curves, axis=0)
        # Plot individual function curves (faint)
        for fn_idx, c in enumerate(all_curves):
            ax.plot(range(max_iter + 1), c, alpha=0.2, color='gray', linewidth=0.7)
        # Plot mean curve
        ax.plot(range(max_iter + 1), mean_curve, color=colors[ax_idx],
                linewidth=2.5, label=f'eps={eps:g} (mean)')
        ax.set_xlabel('Iteration $t$', fontsize=11)
        ax.set_ylabel(r'$P(\tau_\varepsilon \leq t)$', fontsize=11)
        ax.set_title(f'$\\varepsilon = {eps:g}$', fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color='red', linestyle=':', alpha=0.5, linewidth=1)

    fig.suptitle('Empirical Hitting-Time Probability on CEC 2022 ($D=10$, 20 runs × 12 functions)\n'
                 'Each curve: fraction of runs that entered the $\\varepsilon$-basin by iteration $t$',
                 fontsize=13, fontweight='bold')
    plt.savefig(f'{OUT}/fig_hitting_time.png', dpi=200)
    plt.close()
    print(f"Saved: fig_hitting_time.png")


# ============================================================
# PART 3: Cliff's Delta Effect Size for LUNA vs DE
# ============================================================
def compute_cliffs_delta():
    """Compute Cliff's delta effect size for LUNA vs DE on D=10."""
    # Load raw run data from checkpoint
    with open(f'{OUT}/LUNA_final_D10_checkpoint.json') as f:
        d = json.load(f)

    results = d['partial_results']
    fn_names = list(results.keys())

    print(f"\n{'='*80}\nCliff's Delta Effect Size: LUNA vs DE\n{'='*80}")
    print(f"Cliff's delta interpretation:")
    print(f"  |delta| < 0.147: negligible")
    print(f"  |delta| < 0.33:  small")
    print(f"  |delta| < 0.474: medium")
    print(f"  |delta| >= 0.474: large")
    print()

    deltas = []
    print(f"{'Function':<8} {'Cliff δ':>10} {'Interpretation':<20} {'LUNA mean':>12} {'DE mean':>12}")
    for fn in fn_names:
        luna = np.array(results[fn]['LUNA'])
        de = np.array(results[fn]['DE'])
        n_luna = len(luna)
        n_de = len(de)
        # Cliff's delta: #(luna < de) - #(luna > de) / (n_luna * n_de)
        # Note: lower fitness = better
        count_luna_better = 0  # luna fitness < de fitness (luna wins)
        count_de_better = 0
        for l in luna:
            for dval in de:
                if l < dval:
                    count_luna_better += 1
                elif l > dval:
                    count_de_better += 1
        delta = (count_luna_better - count_de_better) / (n_luna * n_de)
        deltas.append(delta)
        abs_d = abs(delta)
        if abs_d < 0.147:
            interp = 'negligible'
        elif abs_d < 0.33:
            interp = 'small'
        elif abs_d < 0.474:
            interp = 'medium'
        else:
            interp = 'large'
        print(f"{fn:<8} {delta:>10.3f} {interp:<20} {np.mean(luna):>12.3e} {np.mean(de):>12.3e}")

    mean_delta = float(np.mean(deltas))
    abs_md = abs(mean_delta)
    if abs_md < 0.147:
        interp = 'negligible'
    elif abs_md < 0.33:
        interp = 'small'
    elif abs_md < 0.474:
        interp = 'medium'
    else:
        interp = 'large'

    print(f"\nMean Cliff's delta: {mean_delta:.3f} ({interp})")
    print(f"  Positive delta = LUNA tends to outperform DE")
    print(f"  Negative delta = DE tends to outperform LUNA")

    # Save
    report = {
        'comparison': 'LUNA vs DE',
        'method': "Cliff's delta (non-parametric effect size)",
        'per_function': [
            {'function': fn, 'delta': float(deltas[i]),
             'luna_mean': float(np.mean(results[fn]['LUNA'])),
             'de_mean': float(np.mean(results[fn]['DE']))}
            for i, fn in enumerate(fn_names)
        ],
        'mean_delta': mean_delta,
        'interpretation': interp,
        'interpretation_scale': {
            'negligible': '|delta| < 0.147',
            'small': '0.147 <= |delta| < 0.33',
            'medium': '0.33 <= |delta| < 0.474',
            'large': '|delta| >= 0.474',
        },
    }
    with open(f'{OUT}/LUNA_cliffs_delta.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: LUNA_cliffs_delta.json")
    return report


# ============================================================
# PART 4: Comparison plot of AUC values (precise)
# ============================================================
def plot_auc_precise():
    """Plot AUC values with more precision to show LUNA vs GA difference."""
    # Recompute AUC precisely
    with open(f'{OUT}/LUNA_final_D10_checkpoint.json') as f:
        ckpt = json.load(f)
    raw = ckpt['partial_results']
    algos = list(next(iter(raw.values())).keys())
    functions = list(raw.keys())

    best_per_fn = {fn: min(v for alg in algos for v in raw[fn][alg]) for fn in functions}

    ratios = {alg: [] for alg in algos}
    for fn in functions:
        for alg in algos:
            for v in raw[fn][alg]:
                ratios[alg].append(max(v, 1e-300) / max(best_per_fn[fn], 1e-300))

    tau_grid = np.logspace(0, np.log10(1000), 2000)  # finer grid
    ecdf = {alg: np.array([np.mean(np.array(ratios[alg]) <= t) for t in tau_grid])
            for alg in algos}

    # Trapezoidal AUC
    auc = {alg: float(np.trapz(ecdf[alg], tau_grid)) for alg in algos}

    # Sort
    sorted_auc = sorted(auc.items(), key=lambda x: -x[1])
    print(f"\nPrecise AUC values (finer grid):")
    for alg, a in sorted_auc:
        print(f"  {alg:>8s}: {a:.4f}")

    return auc


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', type=str, default='all',
                    choices=['all', 'hitting', 'cliffs', 'auc'])
    args = ap.parse_args()

    if args.mode in ['all', 'hitting']:
        ht_data = run_hitting_time(dim=10, n_runs=20, max_iter=500)
        plot_hitting_time_probability(ht_data)

    if args.mode in ['all', 'cliffs']:
        compute_cliffs_delta()

    if args.mode in ['all', 'auc']:
        plot_auc_precise()
