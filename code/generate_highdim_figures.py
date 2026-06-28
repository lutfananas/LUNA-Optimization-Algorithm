#!/usr/bin/env python3
"""Generate figures for high-dimensional benchmarks and other paper updates."""
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


def load(dim):
    with open(f'{OUT}/LUNA_highdim_D{dim}_benchmark.json') as f:
        return json.load(f)


def fig_multidim_ranking():
    """Bar chart: LUNA rank across D=10, 50, 100."""
    d10 = {'LUNA': 1.83, 'DE': 2.08, 'GA': 2.50, 'WOA': 4.25, 'HHO': 5.83,
           'GWO': 6.33, 'AVOA': 6.33, 'PSO': 7.00, 'SMA': 9.00, 'GSA': 9.83}
    d50 = load(50)['friedman_ranking']
    d100 = load(100)['friedman_ranking']

    algos = ['LUNA', 'DE', 'GA', 'GWO', 'WOA', 'HHO', 'AVOA', 'PSO', 'SMA', 'GSA']
    d10_r = [d10[a] for a in algos]
    d50_r = [d50[a] for a in algos]
    d100_r = [d100[a] for a in algos]

    x = np.arange(len(algos))
    width = 0.27

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    b1 = ax.bar(x - width, d10_r, width, label='D=10', color='#1F4E79', edgecolor='white')
    b2 = ax.bar(x,         d50_r, width, label='D=50', color='#C62828', edgecolor='white')
    b3 = ax.bar(x + width, d100_r, width, label='D=100', color='#2E7D32', edgecolor='white')

    # Highlight LUNA bars
    for bar in [b1[0], b2[0], b3[0]]:
        bar.set_edgecolor('gold')
        bar.set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=11)
    ax.set_ylabel('Friedman Rank (1 = best)', fontsize=12)
    ax.set_title('LUNA vs Baselines: Friedman Ranking across Dimensions', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Annotate LUNA bars
    ax.text(0 - width, d10_r[0] + 0.15, f'{d10_r[0]:.2f}', ha='center', fontsize=9, fontweight='bold', color='#1F4E79')
    ax.text(0,         d50_r[0] + 0.15, f'{d50_r[0]:.2f}', ha='center', fontsize=9, fontweight='bold', color='#C62828')
    ax.text(0 + width, d100_r[0] + 0.15, f'{d100_r[0]:.2f}', ha='center', fontsize=9, fontweight='bold', color='#2E7D32')

    plt.savefig(f'{OUT}/fig_multidim_ranking.png', dpi=200)
    plt.close()
    print(f"Saved: fig_multidim_ranking.png")


def fig_winrate_heatmap():
    """Heatmap: LUNA wins/losses/ties vs each baseline at D=10, 50, 100."""
    # D=10 from existing data
    d10_wins = {'GSA': 12, 'HHO': 12, 'SMA': 12, 'WOA': 11, 'GWO': 11, 'AVOA': 11,
                'PSO': 10, 'GA': 9, 'DE': 3}
    d10_loss = {'GSA': 0, 'HHO': 0, 'SMA': 0, 'WOA': 0, 'GWO': 0, 'AVOA': 0,
                'PSO': 0, 'GA': 2, 'DE': 6}

    d50 = load(50)['luna_vs_baselines_wilcoxon']
    d100 = load(100)['luna_vs_baselines_wilcoxon']

    algos = ['PSO', 'GA', 'DE', 'GSA', 'WOA', 'GWO', 'HHO', 'SMA', 'AVOA']
    win_matrix = np.array([
        [d10_wins[a], d50[a]['wins'], d100[a]['wins']] for a in algos
    ])
    loss_matrix = np.array([
        [d10_loss[a], d50[a]['losses'], d100[a]['losses']] for a in algos
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Wins heatmap
    im1 = axes[0].imshow(win_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=12)
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(['D=10', 'D=50', 'D=100'])
    axes[0].set_yticks(range(len(algos))); axes[0].set_yticklabels(algos)
    axes[0].set_title('LUNA Wins', fontsize=13, fontweight='bold')
    for i in range(len(algos)):
        for j in range(3):
            axes[0].text(j, i, str(win_matrix[i, j]), ha='center', va='center',
                         fontsize=11, fontweight='bold',
                         color='white' if win_matrix[i, j] > 6 else 'black')
    plt.colorbar(im1, ax=axes[0])

    # Losses heatmap
    im2 = axes[1].imshow(loss_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=12)
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(['D=10', 'D=50', 'D=100'])
    axes[1].set_yticks(range(len(algos))); axes[1].set_yticklabels(algos)
    axes[1].set_title('LUNA Losses', fontsize=13, fontweight='bold')
    for i in range(len(algos)):
        for j in range(3):
            axes[1].text(j, i, str(loss_matrix[i, j]), ha='center', va='center',
                         fontsize=11, fontweight='bold',
                         color='white' if loss_matrix[i, j] > 6 else 'black')
    plt.colorbar(im2, ax=axes[1])

    fig.suptitle('LUNA Wilcoxon Win/Loss Counts (12 functions, α=0.05)',
                 fontsize=14, fontweight='bold')
    plt.savefig(f'{OUT}/fig_multidim_winrate.png', dpi=200)
    plt.close()
    print(f"Saved: fig_multidim_winrate.png")


def fig_runtime_scaling():
    """Runtime vs dimension for LUNA vs baselines."""
    # D=10 runtime from existing data
    d10_rt = {'LUNA': 0.73, 'GA': 0.68, 'DE': 0.42, 'GWO': 0.37, 'WOA': 0.32,
              'HHO': 0.32, 'SMA': 0.37, 'AVOA': 0.22, 'GSA': 0.18, 'PSO': 0.14}
    d50_rt = {a: r['mean'] for a, r in load(50)['runtime_summary'].items()}
    d100_rt = {a: r['mean'] for a, r in load(100)['runtime_summary'].items()}

    algos = ['LUNA', 'DE', 'GA', 'GWO', 'WOA', 'HHO', 'SMA', 'AVOA', 'GSA', 'PSO']
    dims = [10, 50, 100]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', 'h', '<']
    colors = plt.cm.tab10(np.linspace(0, 1, len(algos)))

    for i, a in enumerate(algos):
        y = [d10_rt[a], d50_rt[a], d100_rt[a]]
        lw = 3 if a == 'LUNA' else 1.5
        ms = 12 if a == 'LUNA' else 8
        ax.plot(dims, y, marker=markers[i], color=colors[i], label=a,
                linewidth=lw, markersize=ms, alpha=1.0 if a == 'LUNA' else 0.75)

    ax.set_xlabel('Dimension D', fontsize=12)
    ax.set_ylabel('Runtime per run (sec)', fontsize=12)
    ax.set_title('Runtime Scaling: LUNA vs Baselines', fontsize=14, fontweight='bold')
    ax.set_xticks(dims)
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.savefig(f'{OUT}/fig_runtime_scaling.png', dpi=200)
    plt.close()
    print(f"Saved: fig_runtime_scaling.png")


def table_highdim_summary():
    """Generate LaTeX table for D=50, 100 results."""
    d50 = load(50)
    d100 = load(100)

    # Sort by D=50 rank
    algos = sorted(d50['friedman_ranking'].items(), key=lambda x: x[1])
    algos = [a[0] for a in algos]

    rows = []
    for a in algos:
        rows.append({
            'Algo': a,
            'D50_rank': d50['friedman_ranking'][a],
            'D50_LUNA_vs': (d50['luna_vs_baselines_wilcoxon'][a]['wins']
                            if a != 'LUNA' else '-'),
            'D100_rank': d100['friedman_ranking'][a],
            'D100_LUNA_vs': (d100['luna_vs_baselines_wilcoxon'][a]['wins']
                             if a != 'LUNA' else '-'),
        })

    df = pd.DataFrame(rows)
    print("\nHigh-D Summary:")
    print(df.to_string(index=False))
    df.to_csv(f'{OUT}/LUNA_highdim_summary.csv', index=False)
    print(f"\nSaved: LUNA_highdim_summary.csv")


if __name__ == "__main__":
    fig_multidim_ranking()
    fig_winrate_heatmap()
    fig_runtime_scaling()
    table_highdim_summary()
