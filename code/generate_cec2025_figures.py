#!/usr/bin/env python3
"""Generate CEC 2025 benchmark figures and generalization-across-generations table."""
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


def fig_cec2025_ranking():
    """Bar chart: CEC 2025 Friedman ranking and U-score."""
    with open(f'{OUT}/LUNA_cec2025_benchmark.json') as f:
        d = json.load(f)

    algos = list(d['friedman_ranking'].keys())
    friedman = [d['friedman_ranking'][a] for a in algos]
    uscore = [d['uscore_ranking'][a] for a in algos]

    # Sort by Friedman rank
    sorted_idx = np.argsort(friedman)
    algos = [algos[i] for i in sorted_idx]
    friedman = [friedman[i] for i in sorted_idx]
    uscore = [uscore[i] for i in sorted_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Friedman ranking
    colors = ['#1F4E79' if a == 'LUNA' else '#BDBDBD' for a in algos]
    bars = axes[0].barh(range(len(algos)), friedman, color=colors, edgecolor='white')
    axes[0].set_yticks(range(len(algos)))
    axes[0].set_yticklabels([f'{a} ({r:.2f})' for a, r in zip(algos, friedman)], fontsize=11)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Friedman Rank (1 = best)', fontsize=12)
    axes[0].set_title('CEC 2025 BC-SOP: Friedman Ranking\n(29 functions, D=30, 10 runs)',
                       fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    # U-Score
    colors2 = ['#1F4E79' if a == 'LUNA' else '#BDBDBD' for a in algos]
    bars2 = axes[1].barh(range(len(algos)), uscore, color=colors2, edgecolor='white')
    axes[1].set_yticks(range(len(algos)))
    axes[1].set_yticklabels([f'{a} ({u:.0f})' for a, u in zip(algos, uscore)], fontsize=11)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('U-Score (lower = better)', fontsize=12)
    axes[1].set_title('CEC 2025 BC-SOP: U-Score Ranking\n(CEC 2025 official metric)',
                       fontsize=13, fontweight='bold')
    axes[1].axvline(0, color='red', linestyle=':', alpha=0.5, linewidth=1)
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.savefig(f'{OUT}/fig_cec2025_ranking.png', dpi=200)
    plt.close()
    print(f"Saved: fig_cec2025_ranking.png")


def fig_generalization_across_generations():
    """Compare LUNA's ranking across CEC 2017, 2022, 2025."""
    # LUNA ranks across benchmarks
    benchmarks = ['CEC 2017\n(D=10)', 'CEC 2022\n(D=10)', 'CEC 2022\n(D=50)',
                  'CEC 2022\n(D=100)', 'CEC 2025\n(D=30)']
    luna_ranks = [3.0, 1.83, 5.17, 5.83, 1.76]
    de_ranks = [2.0, 2.08, 2.50, 3.75, 2.83]
    ga_ranks = [3.0, 2.50, 2.25, 2.33, 3.55]
    avoa_ranks = [1.0, 6.33, 6.17, 5.58, 5.38]

    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    x = np.arange(len(benchmarks))

    ax.plot(x, luna_ranks, 'o-', color='#1F4E79', linewidth=3.0,
            markersize=14, label='LUNA', zorder=5)
    ax.plot(x, de_ranks, 's-', color='#66A61E', linewidth=2,
            markersize=10, label='DE', alpha=0.85)
    ax.plot(x, ga_ranks, '^-', color='#E7298A', linewidth=2,
            markersize=10, label='GA', alpha=0.85)
    ax.plot(x, avoa_ranks, 'D-', color='#FF7F00', linewidth=2,
            markersize=10, label='AVOA', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_ylabel('Friedman Rank (1 = best)', fontsize=12)
    ax.set_title('Generalization Across Benchmark Generations: LUNA vs Top Baselines',
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()  # lower = better at top
    ax.set_ylim(8, 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)

    # Annotate LUNA ranks
    for i, r in enumerate(luna_ranks):
        ax.annotate(f'{r:.2f}', (i, r), textcoords='offset points',
                    xytext=(0, -18), ha='center', fontsize=10,
                    fontweight='bold', color='#1F4E79')

    plt.savefig(f'{OUT}/fig_generalization.png', dpi=200)
    plt.close()
    print(f"Saved: fig_generalization.png")


def table_generalization_summary():
    """Print LaTeX table for generalization across benchmarks."""
    with open(f'{OUT}/LUNA_cec2025_benchmark.json') as f:
        d = json.load(f)

    print("\n=== Generalization Across Benchmark Generations ===")
    print(f"{'Benchmark':<20} {'D':>4} {'LUNA':>8} {'DE':>8} {'GA':>8} {'AVOA':>8}")
    print("-" * 60)
    print(f"{'CEC 2017':<20} {'10':>4} {'3.00':>8} {'2.00':>8} {'3.00':>8} {'1.00':>8}")
    print(f"{'CEC 2022':<20} {'10':>4} {'1.83':>8} {'2.08':>8} {'2.50':>8} {'6.33':>8}")
    print(f"{'CEC 2022':<20} {'50':>4} {'5.17':>8} {'2.50':>8} {'2.25':>8} {'6.17':>8}")
    print(f"{'CEC 2022':<20} {'100':>4} {'5.83':>8} {'3.75':>8} {'2.33':>8} {'5.58':>8}")
    print(f"{'CEC 2025 BC-SOP':<20} {'30':>4} {d['friedman_ranking']['LUNA']:>8.2f} "
          f"{d['friedman_ranking']['DE']:>8.2f} {d['friedman_ranking']['GA']:>8.2f} "
          f"{d['friedman_ranking']['AVOA']:>8.2f}")

    print("\n=== CEC 2025 Wilcoxon (LUNA vs baselines) ===")
    for alg, wlt in d['luna_vs_baselines_wilcoxon'].items():
        print(f"  vs {alg:>8s}: {wlt['wins']}W / {wlt['losses']}L / {wlt['ties']}T")

    print("\n=== U-Score (lower = better) ===")
    for alg, us in sorted(d['uscore_ranking'].items(), key=lambda x: x[1]):
        print(f"  {alg:>8s}: {us:.0f}")


if __name__ == "__main__":
    fig_cec2025_ranking()
    fig_generalization_across_generations()
    table_generalization_summary()
