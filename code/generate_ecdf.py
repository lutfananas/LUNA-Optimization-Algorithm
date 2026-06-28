#!/usr/bin/env python3
"""Generate ECDF (Empirical Cumulative Distribution Function) plot.

Following Dolan & Moré (2002) benchmarking methodology:
- For each algorithm, plot fraction of (function, run) pairs solved to within
  factor tau of the best result for that function.
- ECDF aggregates performance across all 12 CEC 2022 functions × 20 runs = 240
  problem instances per algorithm.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

OUT = '/home/z/my-project/download'


def load_d10_raw():
    """Load raw D=10 results from checkpoint."""
    with open(f'{OUT}/LUNA_final_D10_checkpoint.json') as f:
        d = json.load(f)
    # partial_results[fn][algo] = list of 20 final fitness values
    return d['partial_results']


def ecdf_plot(raw_data, tau_max=1000, n_tau=200):
    """Generate ECDF plot.

    For each algorithm:
      For each (function, run):
        ratio = fitness / best_fitness_for_that_function
        solved if ratio <= tau
      ECDF(tau) = fraction of (function, run) pairs with ratio <= tau
    """
    algos = list(next(iter(raw_data.values())).keys())
    functions = list(raw_data.keys())

    # Compute best per function
    best_per_fn = {}
    for fn in functions:
        all_vals = []
        for alg in algos:
            all_vals.extend(raw_data[fn][alg])
        best_per_fn[fn] = min(all_vals)

    # Compute ratios per (alg, fn, run)
    ratios = {alg: [] for alg in algos}
    for fn in functions:
        for alg in algos:
            for v in raw_data[fn][alg]:
                # Use ratio fitness/best; subtract optimal constant shift if all are positive
                # CEC 2022 functions have known offsets (100, 200, ..., 1200)
                # so ratios may be misleading; better to normalize by best
                ratio = max(v, 1e-300) / max(best_per_fn[fn], 1e-300)
                ratios[alg].append(ratio)

    # ECDF curves
    tau_grid = np.logspace(0, np.log10(tau_max), n_tau)
    ecdf = {alg: np.array([np.mean(np.array(ratios[alg]) <= t) for t in tau_grid])
            for alg in algos}

    # Sort by area under curve (AUC) for legend order
    auc = {alg: np.trapz(ecdf[alg], tau_grid) for alg in algos}
    sorted_algos = sorted(auc.items(), key=lambda x: -x[1])
    print("AUC (higher = better):")
    for alg, a in sorted_algos:
        print(f"  {alg:>8s}: {a:.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(algos)))
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', 'h', '<']

    # Plot LUNA thicker
    for i, (alg, _) in enumerate(sorted_algos):
        idx = algos.index(alg)
        lw = 3.5 if alg == 'LUNA' else 1.4
        alpha = 1.0 if alg == 'LUNA' else 0.75
        ax.step(tau_grid, ecdf[alg], where='post',
                label=f'{alg} (AUC={auc[alg]:.1f})',
                color=colors[idx], linewidth=lw, alpha=alpha)

    ax.set_xscale('log')
    ax.set_xlabel(r'Performance ratio $\tau = f/f_{\mathrm{best}}$ (log scale)', fontsize=12)
    ax.set_ylabel(r'Fraction of problems solved ($\leq \tau$)', fontsize=12)
    ax.set_title('Empirical Cumulative Distribution Function (Dolan-Moré)\n'
                 'CEC 2022, D=10, 12 functions × 20 runs = 240 problem instances per algorithm',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, tau_max)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='lower right', fontsize=10, ncol=2)
    plt.savefig(f'{OUT}/fig_ecdf.png', dpi=200)
    plt.close()
    print(f"\nSaved: fig_ecdf.png")

    # Save ECDF data for reference
    ecdf_report = {
        'description': 'ECDF (Dolan-More 2002) on CEC 2022 D=10. For each algorithm, fraction of (function,run) pairs whose final fitness is within factor tau of the best result for that function.',
        'n_instances': len(functions) * len(raw_data[functions[0]][algos[0]]),
        'auc': {alg: float(auc[alg]) for alg in algos},
        'sorted_by_auc': [alg for alg, _ in sorted_algos],
    }
    with open(f'{OUT}/LUNA_ecdf.json', 'w') as f:
        json.dump(ecdf_report, f, indent=2)
    print(f"Saved: LUNA_ecdf.json")


if __name__ == "__main__":
    raw = load_d10_raw()
    ecdf_plot(raw)
