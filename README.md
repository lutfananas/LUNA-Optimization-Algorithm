# 🌙 LUNA Algorithm (Lunar-inspired Update & Navigation Algorithm)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Benchmark: CEC 2017/2022](https://img.shields.io/badge/benchmark-CEC%202017%20%7C%202022-green.svg)](https://www.cs.cuvic.edu.au/)

LUNA is a novel population-based metaheuristic optimization algorithm grounded in **authentic lunar astronomical mechanics**. Unlike existing nature-inspired algorithms that use metaphors superficially, LUNA mathematically encodes three real celestial phenomena into its search dynamics.

---

## 🌟 Key Results

| Benchmark | LUNA Rank | vs AVOA | vs PSO | vs GWO | vs HHO | vs SMA |
|-----------|-----------|---------|--------|--------|--------|--------|
| **CEC 2022** (12 functions) | **#1** (tied DE) | 11/12 wins | ✅ | ✅ | ✅ | ✅ |
| **CEC 2017** (30 functions) | **#3** | 28/30 wins | 28/30 wins | 28/30 wins | 28/30 wins | 30/30 wins |
| Classical (6 functions) | #4 | Win Rosenbrock | — | — | — | — |

- 🏆 **Rank #1 on CEC 2022** (Friedman rank 2.00, tied with DE)
- 📊 **Outperforms AVOA on 28/30 CEC 2017 functions** (zero losses)
- 🎯 **Reaches global optimum on 5/12 CEC 2022 functions**
- 📈 **Friedman test highly significant** (p < 10⁻¹⁶)

---

## 🌙 Lunar Astronomical Foundation

LUNA's search dynamics are governed by three real astronomical cycles:

### 1. Synodic Month (Phase Cycle)
```python
θ_p(t) = 2π · N_syn · t / T
I(θ_p) = (1 - cos θ_p) / 2  ∈ [0, 1]
```
- **New Moon** (I < 0.2): Pure exploration (Lévy flights / Gaussian)
- **Full Moon** (I ≥ 0.8): Maximum exploitation (4 hybrid strategies)
- **Quarter phases**: Smooth transitional mix

### 2. Anomalistic Month (Distance Cycle)
```python
D(t) = 1 + e · cos(θ_a)     # e = 0.0549 (real lunar eccentricity)
G_eff(t) = G₀ / D(t)²        # Newton's inverse-square law
```
The gravitational pull varies with the Moon's orbital distance — strongest at perigee, weakest at apogee.

### 3. Tidal Force (Spring-Neap Cycle)
```python
T(θ_p) = (1 + cos 2θ_p) / 2  ∈ [0, 1]
G(t) = G_eff(t) · (0.3 + 0.7 · T(θ_p))
```
Tidal force peaks at new AND full moons (spring tides), creating a bi-modal exploitation pattern.

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/lutfananas/LUNA-Optimization-Algorithm.git
cd LUNA-Optimization-Algorithm

# Run demo
python code/luna.py
```

### Basic Usage

```python
from code.luna import LUNA
import numpy as np

# Define objective function
def sphere(x):
    return float(np.sum(x ** 2))

# Initialize and optimize
luna = LUNA()
best_x, best_f, history = luna.optimize(
    f=sphere,
    dim=10,
    bounds=(-100, 100),
    pop=30,
    max_iter=500,
    seed=42
)

print(f"Best fitness: {best_f:.6e}")
```

### Custom Objective Function

```python
def my_objective(x):
    """Your optimization problem here."""
    cost = compute_cost(x)
    penalty = compute_constraint_violation(x)
    return cost + penalty

luna = LUNA(N_syn=5, chaos_ratio=0.5, late_de_thresh=0.8)
best_x, best_f, history = luna.optimize(
    my_objective, dim=20, bounds=(0, 1), pop=50, max_iter=1000
)
```

---

## ⚙️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `G0` | 2.5 | Baseline gravitational constant |
| `N_syn` | 5 | Number of synodic cycles per run |
| `N_anom` | 5 | Number of anomalistic cycles per run |
| `eccentricity` | 0.0549 | Lunar orbital eccentricity (real Moon) |
| `sigma_init` | 0.5 | Initial exploration step size |
| `sigma_min` | 0.001 | Minimum step size floor |
| `late_de_thresh` | 0.8 | Late DE convergence threshold (t/T) |
| `late_F_lo` | 0.1 | Late DE scaling factor lower bound |
| `late_F_hi` | 0.3 | Late DE scaling factor upper bound |
| `chaos_ratio` | 0.5 | Fraction of chaotic init population |
| `p_crossover` | 0.4 | DE crossover probability |
| `obl_period` | 100 | Periodic OBL interval |
| `restart_patience` | 50 | Stagnation restart threshold |

---

## 📁 Repository Structure

```
📦 LUNA-Optimization-Algorithm
├── 📜 README.md                    → This file
├── 📜 LICENSE                      → MIT License
├── 📂 code/
│   ├── luna.py                     → Main LUNA algorithm (standalone)
│   ├── luna_initial.py             → Original prototype (v1)
│   └── luna_benchmark.py           → Benchmark suite runner
├── 📂 paper/
│   ├── LUNA_Paper_Q1.pdf           → Full Q1-grade paper (15 pages)
│   └── LUNA_Paper_Q1.tex           → LaTeX source
├── 📂 figures/
│   ├── fig1_lunar_parameters.png   → Lunar astronomical parameters
│   ├── fig2_cec2017_ranking.png    → CEC 2017 Friedman ranking
│   ├── fig3_cec2022_ranking.png    → CEC 2022 Friedman ranking
│   ├── fig4_cec2017_perfunction.png→ CEC 2017 per-function results
│   ├── fig5_cec2022_perfunction.png→ CEC 2022 per-function results
│   ├── fig6_winrate_heatmap.png    → Win rate vs baselines
│   ├── fig7_ablation_evolution.png → Ablation study
│   ├── fig8_applications.png       → Real-world applications
│   ├── fig9_convergence_cec2017.png→ Convergence curves
│   └── fig10_classical_ranking.png → Classical benchmark ranking
├── 📂 benchmark_data/
│   ├── CEC2017_full_benchmark.json → CEC 2017 complete results
│   ├── CEC2022_full_benchmark.json → CEC 2022 complete results
│   ├── H3_applications_report.json → EVCS + load forecasting results
│   └── ablation_sweep.json         → 96-config parameter sweep
├── 📂 docs/
│   ├── LUNA_Workflow.pdf           → Algorithm workflow diagram
│   └── LUNA_Mathematical_Formalization.pdf → Formal math document
├── 📂 data/
│   └── synthetic_city.csv          → Synthetic EVCS data
└── 📂 results/
    └── README.md                   → Results documentation
```

---

## 📊 Benchmark Results

### CEC 2022 (12 functions, D=10, 20 runs)

| Rank | Algorithm | Friedman Score |
|------|-----------|---------------|
| **1** | **LUNA** | **2.00** |
| 1 | DE | 2.00 |
| 3 | GA | 2.42 |
| 4 | WOA | 4.25 |
| 5 | HHO | 5.83 |
| 6 | GWO | 6.33 |
| 6 | AVOA | 6.33 |
| 8 | PSO | 7.00 |
| 9 | SMA | 9.00 |
| 10 | GSA | 9.83 |

### CEC 2017 (30 functions, D=10, 20 runs)

| Rank | Algorithm | Friedman Score | LUNA Win Rate |
|------|-----------|---------------|---------------|
| 1 | DE | 1.93 | 13% |
| 2 | GA | 2.30 | 47% |
| **3** | **LUNA** | **2.33** | — |
| 4 | WOA | 4.43 | 90% |
| 5 | HHO | 5.60 | 93% |
| 6 | GWO | 6.10 | 93% |
| 7 | AVOA | 6.57 | 93% |
| 8 | PSO | 7.03 | 93% |
| 9 | SMA | 8.77 | 100% |
| 10 | GSA | 9.93 | 97% |

---

## 🔬 Algorithm Design

LUNA combines lunar astronomy with modern optimization techniques:

1. **Chaotic Initialization** — Logistic map (μ=4) + OBL for superior search space coverage
2. **Lunar Phase Strategy** — Exploration/exploitation balance driven by illumination I(θ)
3. **Four Exploitation Strategies** (randomly selected at full moon):
   - S1: Triple-best gravitational pull (GWO-inspired)
   - S2: Spiral update (WOA-inspired)
   - S3: Lévy flight around best (HHO-inspired)
   - S4: DE/current-to-best/1 (DE-inspired)
4. **Late-Stage DE Convergence** — Pure DE with small F for precision (t/T > 0.8)
5. **Adaptive σ** — Rechenberg's 1/5-success rule from Evolution Strategies
6. **Periodic OBL** — Opposition-based learning every 100 iterations
7. **Boundary Reflection** — Prevents boundary clustering
8. **Stagnation Restart** — Escapes deep local optima

---

## 📝 Citation

If you use LUNA in your research, please cite:

```bibtex
@article{luna2025,
  title={LUNA: A Lunar-Inspired Metaheuristic Optimization Algorithm with Real Astronomical Mechanics for Numerical Optimization},
  author={Zahir, Lutfan Anas and Wardani, Nikken Kusuma},
  year={2025},
  publisher={LUNA Project},
  url={https://github.com/lutfananas/LUNA-Optimization-Algorithm}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Lutfan Anas Zahir** & **Nikken Kusuma Wardani**
- 📧 Contact: [lutfananas@gmail.com](mailto:lutfananas@gmail.com)
- 🌐 GitHub: [lutfananas/LUNA-Optimization-Algorithm](https://github.com/lutfananas/LUNA-Optimization-Algorithm)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to fork the repo and submit a pull request.

---

## 🙏 Acknowledgments

This research was supported by the LUNA Project. We thank the metaheuristic optimization community for the foundational algorithms (PSO, GA, DE, GSA, WOA, GWO, HHO, SMA, AVOA) that served as baselines in our evaluation.
