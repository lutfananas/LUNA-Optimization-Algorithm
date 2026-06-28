#!/usr/bin/env python3
"""
Run D=30, D=50 CEC 2022 benchmark + Parameter sensitivity + Post-hoc + CD diagram.
All in one script for efficiency.
"""
import os, sys, json, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
from scipy.stats import wilcoxon, friedmanchisquare
from itertools import combinations

fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

sys.path.insert(0, '/home/z/my-project/scripts')
from cec2022_benchmark import CEC2022
from luna_full_benchmark import PSO, GA, DE, GSA, WOA, GWO, HHO, SMA, AVOA
from luna_v7_class import LUNA_v7

OUT = '/home/z/my-project/download'

# ============================================================
# PART 1: D=30 and D=50 (reduced runs for time)
# ============================================================
def run_dimension(dim, n_runs=10, max_iter=500, pop_size=30):
    cec = CEC2022(dim=dim, seed=42)
    functions = cec.get_all_functions()
    bounds = (cec.lb, cec.ub)

    luna_cfg = dict(G0=5.0, N_syn=5, N_anom=5, eccentricity=0.0549,
                    semi_major=1.0, ang_momentum=0.5, libration_amp=0.12,
                    libration_freq=3, sun_gravity=0.3, orbital_decay=8.0,
                    sigma_init=0.5, sigma_min=0.001, sigma_max=1.0,
                    late_de_thresh=0.6, F_lo=0.02, F_hi=0.08,
                    chaos_ratio=0.5, obl_period=100, obl_frac=0.3,
                    restart_patience=50, restart_frac=0.2, pbest_weight=0.4, eps=1e-10)

    ALGOS = {"LUNA": LUNA_v7(**luna_cfg), "DE": DE(), "GA": GA(), "AVOA": AVOA(),
             "WOA": WOA(), "GWO": GWO(), "HHO": HHO(), "PSO": PSO()}

    print(f"\n{'='*60}\nD={dim} Benchmark ({n_runs} runs, {max_iter} iter)\n{'='*60}")

    all_results = {}
    for fname, func in functions.items():
        print(f"  {fname}...", end="", flush=True)
        all_results[fname] = {}
        for aname, algo in ALGOS.items():
            runs = [algo.optimize(func, dim, bounds, pop=pop_size, max_iter=max_iter, seed=42+r)[1]
                    for r in range(n_runs)]
            all_results[fname][aname] = np.array(runs)
        print(" done")

    # Friedman
    rankings = []
    for fn in functions:
        means = [(an, all_results[fn][an].mean()) for an in ALGOS]
        means.sort()
        row = {}
        for rank, (an, _) in enumerate(means, 1):
            row[an] = rank
        rankings.append(row)

    df_rank = pd.DataFrame(rankings)
    avg_ranks = {an: float(np.mean(df_rank[an].values)) for an in ALGOS}
    sorted_r = sorted(avg_ranks.items(), key=lambda x: x[1])
    print(f"\n  D={dim} Friedman Ranking:")
    for an, r in sorted_r:
        print(f"    {an:>8s}: {r:.2f}")

    # Save
    report = {"dimension": dim, "friedman_ranking": avg_ranks, "results": {}}
    for fn in functions:
        report["results"][fn] = {an: {"mean": float(all_results[fn][an].mean()),
                                       "best": float(all_results[fn][an].min())}
                                  for an in ALGOS}
    with open(f'{OUT}/LUNA_final_D{dim}_benchmark.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: LUNA_final_D{dim}_benchmark.json")
    return avg_ranks


# ============================================================
# PART 2: Parameter Sensitivity
# ============================================================
def run_sensitivity():
    print(f"\n{'='*60}\nParameter Sensitivity Analysis\n{'='*60}")

    cec = CEC2022(dim=10, seed=42)
    # Use 4 representative functions
    test_funcs = {f'F{i}': cec.get_function(i) for i in [1, 5, 8, 11]}
    bounds = (cec.lb, cec.ub)

    base_cfg = dict(G0=5.0, N_syn=5, N_anom=5, eccentricity=0.0549,
                    semi_major=1.0, ang_momentum=0.5, libration_amp=0.12,
                    libration_freq=3, sun_gravity=0.3, orbital_decay=8.0,
                    sigma_init=0.5, sigma_min=0.001, sigma_max=1.0,
                    late_de_thresh=0.6, F_lo=0.02, F_hi=0.08,
                    chaos_ratio=0.5, obl_period=100, obl_frac=0.3,
                    restart_patience=50, restart_frac=0.2, pbest_weight=0.4, eps=1e-10)

    # Parameters to sweep
    sweeps = {
        'N_syn': [1, 3, 5, 7, 10],
        'orbital_decay': [2, 4, 6, 8, 10, 15],
        'G0': [1, 2, 5, 10, 20],
        'sigma_init': [0.1, 0.3, 0.5, 0.7, 1.0],
        'late_de_thresh': [0.4, 0.5, 0.6, 0.7, 0.8],
    }

    sensitivity_results = {}

    for param_name, values in sweeps.items():
        print(f"\n  Sweeping {param_name}: {values}")
        param_results = {}
        for val in values:
            cfg = {**base_cfg, param_name: val}
            algo = LUNA_v7(**cfg)
            means = []
            for fn_name, fn in test_funcs.items():
                runs = [algo.optimize(fn, 10, bounds, pop=30, max_iter=300, seed=42+r)[1]
                        for r in range(3)]
                means.append(np.mean(runs))
            avg_mean = np.mean(means)
            param_results[val] = avg_mean
            print(f"    {param_name}={val}: avg_fitness={avg_mean:.4e}")
        sensitivity_results[param_name] = param_results

    # Save
    with open(f'{OUT}/LUNA_sensitivity.json', 'w') as f:
        json.dump(sensitivity_results, f, indent=2, default=str)

    # Generate heatmap
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    for idx, (param, results) in enumerate(sensitivity_results.items()):
        ax = axes[idx]
        vals = list(results.keys())
        fitness = list(results.values())
        norm_fitness = [f / max(fitness) for f in fitness]  # normalize
        colors = ['#2E7D32' if f == min(fitness) else '#1F4E79' for f in fitness]
        ax.bar(range(len(vals)), norm_fitness, color=colors, edgecolor='white')
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([str(v) for v in vals], fontsize=9)
        ax.set_title(param, fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Fitness' if idx == 0 else '')
        ax.grid(True, alpha=0.3, axis='y')
        # Mark best
        best_idx = fitness.index(min(fitness))
        ax.text(best_idx, norm_fitness[best_idx] + 0.02, '★', ha='center', fontsize=14, color='red')
    fig.suptitle('Parameter Sensitivity Analysis (CEC 2022 F1, F5, F8, F11)', fontsize=14, fontweight='bold')
    plt.savefig(f'{OUT}/fig_sensitivity.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Sensitivity figure saved: fig_sensitivity.png")
    return sensitivity_results


# ============================================================
# PART 3: Post-hoc Holm/Bonferroni + CD Diagram
# ============================================================
def run_posthoc():
    print(f"\n{'='*60}\nPost-hoc Analysis (Holm/Bonferroni + CD Diagram)\n{'='*60}")

    # Load D=10 results
    with open(f'{OUT}/LUNA_final_D10_benchmark.json') as f:
        data = json.load(f)

    funcs = sorted(data['results'].keys(), key=lambda x: int(x[1:]))
    algos = list(data['friedman_ranking'].keys())
    n_funcs = len(funcs)
    n_algos = len(algos)

    # Get per-function ranks
    ranks_per_func = []
    for fn in funcs:
        means = [(an, data['results'][fn][an]['mean']) for an in algos]
        means.sort()
        rank_dict = {an: r+1 for r, (an, _) in enumerate(means)}
        ranks_per_func.append(rank_dict)

    # Average ranks
    avg_ranks = {an: np.mean([r[an] for r in ranks_per_func]) for an in algos}
    sorted_algos = sorted(avg_ranks.items(), key=lambda x: x[1])

    # Friedman chi2
    rank_matrix = np.array([[r[an] for an in algos] for r in ranks_per_func])
    chi2, p_friedman = friedmanchisquare(*[rank_matrix[:, j] for j in range(n_algos)])

    print(f"  Friedman chi2={chi2:.2f}, p={p_friedman:.4e}")
    print(f"  Average ranks: {dict(sorted(avg_ranks.items(), key=lambda x: x[1]))}")

    # Holm post-hoc: compare LUNA (control) vs each other
    # Using simplified Holm: compute p-values from Wilcoxon, then adjust
    control = 'LUNA'
    p_values = {}
    for an in algos:
        if an == control:
            continue
        # Use mean values as proxy (we don't have raw data for Wilcoxon in JSON)
        # Instead use rank-based p-value approximation
        diff = avg_ranks[control] - avg_ranks[an]
        # z-score approximation
        z = diff / math.sqrt(n_algos * (n_algos + 1) / (6 * n_funcs))
        from scipy.stats import norm
        p = 2 * (1 - norm.cdf(abs(z)))
        p_values[an] = p

    # Holm adjustment
    m = len(p_values)
    sorted_p = sorted(p_values.items(), key=lambda x: x[1])
    holm_results = []
    for i, (an, p) in enumerate(sorted_p):
        adjusted_p = min(p * (m - i), 1.0)
        holm_results.append({
            'algorithm': an,
            'raw_p': p,
            'holm_p': adjusted_p,
            'significant': adjusted_p < 0.05,
            'rank_diff': avg_ranks[an] - avg_ranks[control],
        })

    print(f"\n  Holm Post-hoc (control: LUNA):")
    print(f"  {'Algorithm':>10s} {'Rank Diff':>10s} {'Raw p':>12s} {'Holm p':>12s} {'Sig':>5s}")
    for r in holm_results:
        sig = '***' if r['significant'] else 'ns'
        print(f"  {r['algorithm']:>10s} {r['rank_diff']:>10.2f} {r['raw_p']:>12.4e} {r['holm_p']:>12.4e} {sig:>5s}")

    # CD Diagram
    # Critical Difference = q_alpha * sqrt(k(k+1)/(6N))
    k = n_algos
    N = n_funcs
    # q_alpha for alpha=0.05, k=10 (from Nemenyi table) ≈ 3.714
    q_alpha = 3.714  # approximation for k=10
    CD = q_alpha * math.sqrt(k * (k + 1) / (6 * N))

    print(f"\n  Critical Difference (CD) = {CD:.2f}")

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    # Plot ranks
    y_pos = range(len(sorted_algos))
    ax.barh(y_pos, [r for _, r in sorted_algos], color=['#1F4E79' if a == 'LUNA' else '#BDBDBD'
                                                          for a, _ in sorted_algos],
            edgecolor='white', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{a} ({r:.2f})' for a, r in sorted_algos], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Average Friedman Rank', fontsize=12)
    ax.set_title(f'Critical Difference Diagram (CD={CD:.2f}, Nemenyi test, alpha=0.05)', fontsize=13, fontweight='bold')

    # Draw CD line
    best_rank = sorted_algos[0][1]
    ax.plot([best_rank, best_rank + CD], [-0.3, -0.3], 'r-', linewidth=3)
    ax.text(best_rank + CD/2, -0.5, f'CD={CD:.2f}', ha='center', fontsize=10, color='red', fontweight='bold')

    # Connect algorithms that are NOT significantly different (within CD)
    for i in range(len(sorted_algos)):
        for j in range(i+1, len(sorted_algos)):
            if sorted_algos[j][1] - sorted_algos[i][1] < CD:
                # Not significantly different - draw line
                ax.plot([sorted_algos[i][1], sorted_algos[j][1]], [i, j], 'k--', alpha=0.3, linewidth=1)

    ax.grid(True, alpha=0.3, axis='x')
    plt.savefig(f'{OUT}/fig_cd_diagram.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  CD diagram saved: fig_cd_diagram.png")

    # Save post-hoc results
    posthoc_report = {
        'friedman_chi2': float(chi2),
        'friedman_p': float(p_friedman),
        'critical_difference': float(CD),
        'avg_ranks': {an: float(r) for an, r in avg_ranks.items()},
        'holm_posthoc': holm_results,
    }
    with open(f'{OUT}/LUNA_posthoc.json', 'w') as f:
        json.dump(posthoc_report, f, indent=2)
    print(f"  Post-hoc results saved: LUNA_posthoc.json")


# ============================================================
# PART 4: Convergence + Diversity + Orbital evolution figures
# ============================================================
def run_extra_figures():
    print(f"\n{'='*60}\nExtra Figures (Convergence, Diversity, Orbital)\n{'='*60}")

    cec = CEC2022(dim=10, seed=42)
    func = cec.get_function(5)  # F5 (Rastrigin)
    bounds = (cec.lb, cec.ub)

    luna_cfg = dict(G0=5.0, N_syn=5, N_anom=5, eccentricity=0.0549,
                    semi_major=1.0, ang_momentum=0.5, libration_amp=0.12,
                    libration_freq=3, sun_gravity=0.3, orbital_decay=8.0,
                    sigma_init=0.5, sigma_min=0.001, sigma_max=1.0,
                    late_de_thresh=0.6, F_lo=0.02, F_hi=0.08,
                    chaos_ratio=0.5, obl_period=100, obl_frac=0.3,
                    restart_patience=50, restart_frac=0.2, pbest_weight=0.4, eps=1e-10)

    # Run LUNA with history tracking
    algo = LUNA_v7(**luna_cfg)
    _, fb, history = algo.optimize(func, 10, bounds, pop=30, max_iter=500, seed=42)

    # Run baselines for convergence comparison
    conv_data = {'LUNA': history}
    for aname, algo_cls in [('DE', DE()), ('AVOA', AVOA()), ('GWO', GWO()), ('WOA', WOA())]:
        _, _, h = algo_cls.optimize(func, 10, bounds, pop=30, max_iter=500, seed=42)
        conv_data[aname] = h

    # Fig: Convergence curves
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = {'LUNA': '#1F4E79', 'DE': '#66A61E', 'AVOA': '#FF7F00', 'GWO': '#666666', 'WOA': '#A6761D'}
    for aname, h in conv_data.items():
        h_plot = [max(v, 1e-300) for v in h]
        ax.semilogy(h_plot, label=aname, color=colors.get(aname, '#333'),
                    linewidth=2.5 if aname == 'LUNA' else 1.3,
                    alpha=1.0 if aname == 'LUNA' else 0.75)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Fitness (log scale)', fontsize=12)
    ax.set_title('Convergence Curves on CEC 2022 F5 (Rastrigin)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, which='both')
    plt.savefig(f'{OUT}/fig_convergence.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: fig_convergence.png")

    # Fig: Orbital parameter evolution
    T = 500
    t = np.arange(T)
    theta_p = 2 * np.pi * 5 * t / T
    I = (1 - np.cos(theta_p)) / 2
    theta_a = 2 * np.pi * 5 * t / T
    D = 1 + 0.0549 * np.cos(theta_a)
    mu_t = 5.0 * np.exp(-8.0 * t / T)
    v = np.sqrt(mu_t * np.maximum(2.0/D - 1.0/1.0, 1e-10))
    vf = v / (v + 1)
    lib = 0.12 * np.sin(2 * np.pi * 3 * t / T)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes[0,0].plot(t, I, color='#1F4E79', linewidth=2)
    axes[0,0].set_title('(a) Illumination I(t)', fontweight='bold')
    axes[0,0].set_xlabel('Iteration'); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(t, mu_t, color='#C62828', linewidth=2)
    axes[0,1].set_title('(b) Orbital Energy mu(t) (dissipation)', fontweight='bold')
    axes[0,1].set_xlabel('Iteration'); axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot(t, vf, color='#2E7D32', linewidth=2)
    axes[1,0].set_title('(c) Vis-Viva Factor v_f(t)', fontweight='bold')
    axes[1,0].set_xlabel('Iteration'); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(t, lib, color='#FF7F00', linewidth=2)
    axes[1,1].set_title('(d) Libration lambda(t)', fontweight='bold')
    axes[1,1].set_xlabel('Iteration'); axes[1,1].grid(True, alpha=0.3)

    fig.suptitle('Lunar Orbital Parameter Evolution Over 500 Iterations', fontsize=14, fontweight='bold')
    plt.savefig(f'{OUT}/fig_orbital_evolution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: fig_orbital_evolution.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Part 1: D=30 and D=50
    r30 = run_dimension(30, n_runs=10, max_iter=500, pop_size=30)
    r50 = run_dimension(50, n_runs=10, max_iter=500, pop_size=30)

    # Part 2: Sensitivity
    sens = run_sensitivity()

    # Part 3: Post-hoc + CD
    run_posthoc()

    # Part 4: Extra figures
    run_extra_figures()

    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print("=" * 60)
