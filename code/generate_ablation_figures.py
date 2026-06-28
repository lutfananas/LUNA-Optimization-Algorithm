#!/usr/bin/env python3
"""Generate ablation Wilcoxon heatmap and summary once ablation completes."""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

OUT = '/home/z/my-project/download'


def main():
    with open(f'{OUT}/LUNA_ablation_wilcoxon.json') as f:
        d = json.load(f)

    variants = d['variants']
    n = len(variants)

    # Build win/loss matrix
    win_matrix = np.zeros((n, n), dtype=int)
    loss_matrix = np.zeros((n, n), dtype=int)
    tie_matrix = np.zeros((n, n), dtype=int)
    sig_matrix = np.zeros((n, n), dtype=int)
    p_matrix = np.ones((n, n))

    for i, v1 in enumerate(variants):
        for j, v2 in enumerate(variants):
            if i == j:
                continue
            key = f"{v1}_vs_{v2}"
            if key in d['pairwise_wilcoxon_with_holm']:
                ps = d['pairwise_wilcoxon_with_holm'][key]
                win_matrix[i, j] = ps['wins_v1']
                loss_matrix[i, j] = ps['wins_v2']
                tie_matrix[i, j] = ps['ties']
                sig_matrix[i, j] = 1 if ps['any_significant'] else 0
                p_matrix[i, j] = ps['min_holm_p']

    # Plot 1: Win matrix heatmap
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.imshow(win_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=12)
    ax.set_xticks(range(n)); ax.set_xticklabels(variants, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(variants, fontsize=10)
    ax.set_title('Ablation Win Matrix (Wilcoxon, $p<0.05$ after Holm)', fontsize=13, fontweight='bold')
    for i in range(n):
        for j in range(n):
            if i != j:
                color = 'white' if win_matrix[i, j] > 6 else 'black'
                ax.text(j, i, str(win_matrix[i, j]), ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)
            else:
                ax.text(j, i, '-', ha='center', va='center', fontsize=10, color='gray')
    plt.colorbar(im, ax=ax, label='Wins (out of 12 functions)')
    plt.savefig(f'{OUT}/fig_ablation_wilcoxon.png', dpi=200)
    plt.close()
    print(f"Saved: fig_ablation_wilcoxon.png")

    # Plot 2: Mean fitness comparison
    means = d['overall_mean_fitness']
    sorted_variants = sorted(means.items(), key=lambda x: x[1])
    names = [v for v, _ in sorted_variants]
    values = [m for _, m in sorted_variants]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    colors = ['#1F4E79' if 'full' in v else
              '#C62828' if 'astronomy' in v else
              '#66A61E' if 'lateDE' in v else
              '#FF7F00' if 'pbest' in v else
              '#6A3D9A' if 'restart' in v else
              '#BDBDBD' for v in names]
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xscale('log')
    ax.set_xlabel('Mean Fitness across 12 CEC 2022 Functions (log scale)', fontsize=11)
    ax.set_title('Ablation: Overall Mean Fitness per Variant (lower = better)', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    for i, (v, m) in enumerate(zip(names, values)):
        ax.text(m * 1.05, i, f'{m:.2e}', va='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    plt.savefig(f'{OUT}/fig_ablation_fitness.png', dpi=200)
    plt.close()
    print(f"Saved: fig_ablation_fitness.png")

    # Print summary
    print(f"\n=== Ablation Summary ===")
    print(f"{'Variant':<22} {'Mean Fitness':>15} {'Rank':>5}")
    for i, (v, m) in enumerate(sorted_variants, 1):
        print(f"  {v:<20} {m:>15.3e} {i:>5}")

    print(f"\n=== Pairwise Significance (Holm-corrected p < 0.05) ===")
    for i, v1 in enumerate(variants):
        for j, v2 in enumerate(variants):
            if i < j:
                key = f"{v1}_vs_{v2}"
                if key in d['pairwise_wilcoxon_with_holm']:
                    ps = d['pairwise_wilcoxon_with_holm'][key]
                    sig = '***' if ps['any_significant'] else 'ns'
                    print(f"  {v1} vs {v2}: v1_wins={ps['wins_v1']}, v2_wins={ps['wins_v2']}, "
                          f"ties={ps['ties']}, min_holm_p={ps['min_holm_p']:.2e} {sig}")


if __name__ == "__main__":
    main()
