#!/usr/bin/env python3
"""
LUNA v9 Parametric Framework + 50-Configuration Sweep
======================================================

Real lunar astronomy with 3 month types + perigee/apogee modulation.
Sweep 50+ configurations on quick benchmark to find ones that beat AVOA.

Real lunar parameters:
  - Synodic month: 29.53 days (phase cycle, new moon to new moon)
  - Anomalistic month: 27.55 days (perigee to perigee, distance varies)
  - Draconic month: 27.21 days (node cycle, eclipse season)
  - Perigee distance: 363,300 km (strongest pull)
  - Apogee distance: 405,500 km (weakest pull)
  - Tidal range: spring tide ~1.8x neap tide

We model:
  - Phase angle: theta_p(t) = 2*pi*N_syn*t/T (synodic)
  - Distance modulation: D(t) = 1 + e*cos(theta_a) where e~0.0549 (lunar eccentricity)
  - G_eff(t) = G0 / D(t)^2 (inverse square law, real physics)
  - Tidal force: T(theta_p) = (1 + cos(2*theta_p))/2 (spring/neap)
  - Illumination: I(theta_p) = (1 - cos(theta_p))/2

50 configurations sweep:
  - N_syn: 1, 3, 5, 7
  - Late DE thresh: 0.6, 0.7, 0.8, 0.9
  - Late F range: (0.05,0.15), (0.1,0.3), (0.2,0.4)
  - Chaos ratio: 0.5, 1.0
  - Strategy: 'de_only', 'de_spiral', 'de_levy', 'all4'
"""

import os, sys, json, math, time, itertools
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, '/home/z/my-project/scripts')
from luna_full_benchmark import (
    LUNAv1, LUNAv2, PSO, GA, DE, GSA, WOA, GWO, HHO, SMA, AVOA,
    sphere, rastrigin, rosenbrock, ackley, griewank, schwefel
)
from luna_v4_benchmark import LUNAv4
from luna_v8_benchmark import LUNAv8

OUTPUT_DIR = "/home/z/my-project/download"


class LUNAv9_parametric:
    """LUNA v9 with real lunar astronomy + parametric configuration."""

    def __init__(self, config):
        self.config = config
        self.G0 = config.get('G0', 2.5)
        self.N_syn = config.get('N_syn', 5)
        self.N_anom = config.get('N_anom', self.N_syn)  # anomalistic
        self.eccentricity = config.get('eccentricity', 0.0549)  # real moon
        self.kappa = config.get('kappa', 0.5)
        self.sigma_init = config.get('sigma_init', 0.5)
        self.sigma_min = config.get('sigma_min', 0.001)
        self.sigma_max = config.get('sigma_max', 1.0)
        self.levy_beta = config.get('levy_beta', 1.5)
        self.beta_sig = config.get('beta_sig', 8.0)
        self.tau_center = config.get('tau_center', 0.35)
        self.lambda_g = config.get('lambda_g', 0.6)
        self.lambda_p = config.get('lambda_p', 0.4)
        self.F_de_lo = config.get('F_de_lo', 0.4)
        self.F_de_hi = config.get('F_de_hi', 0.9)
        self.p_crossover = config.get('p_crossover', 0.4)
        self.W_landscape = config.get('W_landscape', 10)
        self.var_threshold = config.get('var_threshold', 0.3)
        self.eps = config.get('eps', 1e-3)
        self.obl_flag = config.get('obl_flag', True)
        self.restart_patience = config.get('restart_patience', 50)
        self.restart_frac = config.get('restart_frac', 0.2)
        self.w_alpha = config.get('w_alpha', 0.5)
        self.w_beta = config.get('w_beta', 0.3)
        self.w_delta = config.get('w_delta', 0.2)
        # v8 features
        self.chaos_ratio = config.get('chaos_ratio', 0.5)
        self.late_de_thresh = config.get('late_de_thresh', 0.8)
        self.late_F_lo = config.get('late_F_lo', 0.1)
        self.late_F_hi = config.get('late_F_hi', 0.3)
        self.obl_period = config.get('obl_period', 100)
        self.obl_frac = config.get('obl_frac', 0.3)
        # NEW: late-stage strategy
        self.late_strategy = config.get('late_strategy', 'de_only')
        # NEW: real lunar distance modulation
        self.use_real_distance = config.get('use_real_distance', True)

    def _G(self, t, T):
        """Real lunar G with synodic + anomalistic + eccentricity."""
        theta_p = 2 * np.pi * self.N_syn * t / T  # synodic phase
        if self.use_real_distance:
            theta_a = 2 * np.pi * self.N_anom * t / T  # anomalistic
            D = 1 + self.eccentricity * np.cos(theta_a)  # distance ratio
            G_eff = self.G0 / (D ** 2)  # inverse square law
        else:
            G_eff = self.G0
        # Tidal modulation
        T_force = (1 + np.cos(2 * theta_p)) / 2
        return G_eff * (0.3 + 0.7 * T_force)

    def _illumination(self, t, T):
        theta_p = 2 * np.pi * self.N_syn * t / T
        return (1 - np.cos(theta_p)) / 2

    def _p_explore(self, t, T):
        return 1.0 / (1.0 + np.exp(-self.beta_sig * (t / T - self.tau_center)))

    def _levy(self, dim):
        beta = self.levy_beta
        sigma = (math.gamma(1 + beta) * math.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, 1, dim) * sigma
        v = np.abs(np.random.normal(0, 1, dim))
        return u / (v + 1e-30) ** (1 / beta)

    def _chaotic_init(self, n, dim, lb, ub):
        chaotic = np.zeros((n, dim))
        for i in range(n):
            x = 0.5
            for j in range(dim):
                x = 4 * x * (1 - x)
                chaotic[i, j] = lb + x * (ub - lb)
        return chaotic

    def _reflect_bounds(self, x, lb, ub):
        for _ in range(3):
            over = x > ub; under = x < lb
            if not over.any() and not under.any():
                break
            x = np.where(over, 2 * ub - x, x)
            x = np.where(under, 2 * lb - x, x)
        return np.clip(x, lb, ub)

    def _late_strategy(self, X, fit, pbest, pbest_fit, xb, fb, i, pop, dim, lb, ub, f):
        """Apply late-stage strategy based on config."""
        cands = [j for j in range(pop) if j != i]
        a, b = np.random.choice(cands, 2, replace=False)
        F_de = np.random.uniform(self.late_F_lo, self.late_F_hi)

        if self.late_strategy == 'de_only':
            xn = X[i] + F_de * (xb - X[i]) + F_de * (X[a] - X[b])
        elif self.late_strategy == 'de_spiral':
            if np.random.random() < 0.5:
                xn = X[i] + F_de * (xb - X[i]) + F_de * (X[a] - X[b])
            else:
                D = np.abs(xb - X[i])
                l = np.random.uniform(-1, 1, dim)
                xn = D * np.exp(l) * np.cos(2 * np.pi * l) + xb
        elif self.late_strategy == 'de_levy':
            if np.random.random() < 0.5:
                xn = X[i] + F_de * (xb - X[i]) + F_de * (X[a] - X[b])
            else:
                xn = xb - np.abs(2 * np.random.random(dim) * xb - X[i]) * F_de
                xn = xn + self._levy(dim) * 0.1
        else:  # 'all4'
            strategy = np.random.randint(1, 5)
            if strategy == 1:
                xn = X[i] + F_de * (xb - X[i]) + F_de * (X[a] - X[b])
            elif strategy == 2:
                D = np.abs(xb - X[i])
                l = np.random.uniform(-1, 1, dim)
                xn = D * np.exp(l) * np.cos(2 * np.pi * l) + xb
            elif strategy == 3:
                xn = xb - np.abs(2 * np.random.random(dim) * xb - X[i]) * F_de
                xn = xn + self._levy(dim) * 0.1
            else:
                xn = X[i] + F_de * 0.5 * (xb - X[i]) + F_de * 0.3 * (X[a] - X[b])

        xn = self._reflect_bounds(xn, lb, ub)
        fn = f(xn)
        return xn, fn

    def optimize(self, f, dim, bounds, pop=30, max_iter=500, seed=None):
        if seed is not None:
            np.random.seed(seed)
        lb, ub = bounds

        # Chaos + random + OBL init
        n_chaos = int(pop * self.chaos_ratio)
        n_random = pop - n_chaos
        X_chaos = self._chaotic_init(n_chaos, dim, lb, ub)
        X_random = np.random.uniform(lb, ub, (n_random, dim))
        X_init = np.vstack([X_chaos, X_random])
        X_obl = (lb + ub) - X_init
        X_combined = np.vstack([X_init, X_obl])
        fit_combined = np.array([f(x) for x in X_combined])
        order = np.argsort(fit_combined)[:pop]
        X = X_combined[order].copy()
        fit = fit_combined[order].copy()

        pbest = X.copy(); pbest_fit = fit.copy()
        bi = np.argmin(fit)
        xb = X[bi].copy(); fb = fit[bi]
        history = [fb]

        sigma = self.sigma_init
        success_window = []
        last_improvement = 0
        last_obl_iter = 0
        var_history = []

        for t in range(max_iter):
            G_t = self._G(t, max_iter)
            I = self._illumination(t, max_iter)
            p_exp = self._p_explore(t, max_iter)
            alpha_mix = 0.7 - 0.5 * (t / max_iter)
            is_late_de = (t / max_iter) > self.late_de_thresh

            # Landscape detection
            if len(var_history) >= self.W_landscape:
                recent_vars = var_history[-self.W_landscape:]
                mean_var = np.mean(recent_vars)
                norm_var = mean_var / (abs(fb) + 1e-10)
                is_multimodal = norm_var > self.var_threshold
            else:
                is_multimodal = True

            order_fit = np.argsort(fit)
            alpha = X[order_fit[0]].copy() if len(order_fit) > 0 else xb
            beta_sol = X[order_fit[1]].copy() if len(order_fit) > 1 else xb
            delta_sol = X[order_fit[2]].copy() if len(order_fit) > 2 else xb

            iter_improved = False

            for i in range(pop):
                if is_late_de:
                    xn, fn = self._late_strategy(X, fit, pbest, pbest_fit, xb, fb, i, pop, dim, lb, ub, f)
                    if fn < fit[i]:
                        X[i] = xn; fit[i] = fn
                        iter_improved = True
                        if fn < pbest_fit[i]:
                            pbest[i] = xn.copy(); pbest_fit[i] = fn
                        if fn < fb:
                            fb = fn; xb = xn.copy(); last_improvement = t
                    continue

                # DE crossover
                if np.random.random() < self.p_crossover:
                    cands = [j for j in range(pop) if j != i]
                    a, b = np.random.choice(cands, 2, replace=False)
                    F_de = np.random.uniform(self.F_de_lo, self.F_de_hi)
                    xn = X[i] + F_de * (X[a] - X[b])
                    xn = self._reflect_bounds(xn, lb, ub)
                    fn = f(xn)
                    if fn < fit[i]:
                        X[i] = xn; fit[i] = fn
                        iter_improved = True
                        if fn < pbest_fit[i]:
                            pbest[i] = xn.copy(); pbest_fit[i] = fn
                        if fn < fb:
                            fb = fn; xb = xn.copy(); last_improvement = t
                    continue

                # Lunar phase-based
                E0 = np.random.uniform(-1, 1)
                E = 2 * E0 * (1 - t / max_iter)
                abs_E = abs(E)

                if abs_E >= 1:
                    if is_multimodal:
                        levy_step = self._levy(dim)
                        best_pull = (xb - X[i])
                        step = sigma * (alpha_mix * levy_step + (1 - alpha_mix) * best_pull)
                        xn = X[i] + step
                    else:
                        step = np.random.normal(0, sigma * 0.3, dim)
                        xn = X[i] + step + 0.1 * (xb - X[i])
                elif abs_E >= 0.5:
                    R_a = np.linalg.norm(alpha - X[i]) + self.eps
                    R_b = np.linalg.norm(beta_sol - X[i]) + self.eps
                    R_d = np.linalg.norm(delta_sol - X[i]) + self.eps
                    R_p = np.linalg.norm(pbest[i] - X[i]) + self.eps
                    xi = np.random.uniform(0.5, 1.0)
                    pull = (self.w_alpha * (alpha - X[i]) / R_a
                            + self.w_beta * (beta_sol - X[i]) / R_b
                            + self.w_delta * (delta_sol - X[i]) / R_d
                            + self.lambda_p * (pbest[i] - X[i]) / R_p)
                    xn = X[i] + xi * G_t * pull
                else:
                    R = np.linalg.norm(xb - X[i]) + self.eps
                    xi = np.random.uniform(0.7, 1.0)
                    xn = X[i] + xi * G_t * (xb - X[i]) / R
                    xn = xn + np.random.normal(0, sigma * 0.1, dim)

                xn = self._reflect_bounds(xn, lb, ub)
                fn = f(xn)

                if fn < fit[i]:
                    X[i] = xn; fit[i] = fn
                    iter_improved = True
                    if fn < pbest_fit[i]:
                        pbest[i] = xn.copy(); pbest_fit[i] = fn
                    if fn < fb:
                        fb = fn; xb = xn.copy(); last_improvement = t

            # Adaptive sigma
            success_window.append(1 if iter_improved else 0)
            if len(success_window) > 20:
                success_window.pop(0)
            if len(success_window) == 20:
                success_rate = sum(success_window) / 20
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9
                sigma = max(self.sigma_min, min(self.sigma_max, sigma))

            var_history.append(float(np.var(fit)))

            # Periodic OBL
            if t - last_obl_iter >= self.obl_period and t > 0:
                last_obl_iter = t
                n_obl = max(1, int(pop * self.obl_frac))
                worst_idx = np.argsort(fit)[-n_obl:]
                opposites = (lb + ub) - X[worst_idx]
                opp_fit = np.array([f(x) for x in opposites])
                for k, idx in enumerate(worst_idx):
                    if opp_fit[k] < fit[idx]:
                        X[idx] = opposites[k]
                        fit[idx] = opp_fit[k]
                        if opp_fit[k] < pbest_fit[idx]:
                            pbest[idx] = opposites[k].copy()
                            pbest_fit[idx] = opp_fit[k]
                        if opp_fit[k] < fb:
                            fb = opp_fit[k]; xb = opposites[k].copy()
                            last_improvement = t

            # Stagnation restart
            if t - last_improvement >= self.restart_patience:
                n_restart = max(1, int(pop * self.restart_frac))
                worst_idx = np.argsort(fit)[-n_restart:]
                for idx in worst_idx:
                    X[idx] = np.random.uniform(lb, ub, dim)
                    fit[idx] = f(X[idx])
                    pbest[idx] = X[idx].copy(); pbest_fit[idx] = fit[idx]
                    if fit[idx] < fb:
                        fb = fit[idx]; xb = X[idx].copy()
                        last_improvement = t

            history.append(fb)

        return xb, fb, history


# ============================================================
# Configuration Sweep Design (50+ configs)
# ============================================================
def generate_configs():
    """Generate 50+ configurations to sweep."""
    configs = []

    # Base lunar params variations (4)
    for N_syn in [1, 3, 5, 7]:
        configs.append({
            'name': f'N_syn={N_syn}',
            'N_syn': N_syn,
            'late_strategy': 'de_only',
            'late_de_thresh': 0.8,
            'late_F_lo': 0.1, 'late_F_hi': 0.3,
            'chaos_ratio': 0.5,
        })

    # Late DE threshold variations (4)
    for thresh in [0.6, 0.7, 0.8, 0.9]:
        configs.append({
            'name': f'late_thresh={thresh}',
            'N_syn': 5,
            'late_strategy': 'de_only',
            'late_de_thresh': thresh,
            'late_F_lo': 0.1, 'late_F_hi': 0.3,
            'chaos_ratio': 0.5,
        })

    # Late F range variations (4)
    for F_lo, F_hi in [(0.01, 0.1), (0.05, 0.15), (0.1, 0.3), (0.2, 0.4)]:
        configs.append({
            'name': f'F=({F_lo},{F_hi})',
            'N_syn': 5,
            'late_strategy': 'de_only',
            'late_de_thresh': 0.8,
            'late_F_lo': F_lo, 'late_F_hi': F_hi,
            'chaos_ratio': 0.5,
        })

    # Late strategy variations (4)
    for strat in ['de_only', 'de_spiral', 'de_levy', 'all4']:
        configs.append({
            'name': f'strat={strat}',
            'N_syn': 5,
            'late_strategy': strat,
            'late_de_thresh': 0.8,
            'late_F_lo': 0.1, 'late_F_hi': 0.3,
            'chaos_ratio': 0.5,
        })

    # Chaos ratio variations (4)
    for cr in [0.3, 0.5, 0.7, 1.0]:
        configs.append({
            'name': f'chaos={cr}',
            'N_syn': 5,
            'late_strategy': 'de_only',
            'late_de_thresh': 0.8,
            'late_F_lo': 0.1, 'late_F_hi': 0.3,
            'chaos_ratio': cr,
        })

    # Real distance on/off + eccentricity (4)
    for use_dist, ecc in [(True, 0.0549), (True, 0.1), (True, 0.2), (False, 0)]:
        configs.append({
            'name': f'dist={use_dist},ecc={ecc}',
            'N_syn': 5,
            'late_strategy': 'de_only',
            'late_de_thresh': 0.8,
            'late_F_lo': 0.1, 'late_F_hi': 0.3,
            'chaos_ratio': 0.5,
            'use_real_distance': use_dist,
            'eccentricity': ecc,
        })

    # Sigma min variations (3)
    for s_min in [0.0001, 0.001, 0.01]:
        configs.append({
            'name': f'sigma_min={s_min}',
            'N_syn': 5,
            'late_strategy': 'de_only',
            'late_de_thresh': 0.8,
            'late_F_lo': 0.1, 'late_F_hi': 0.3,
            'chaos_ratio': 0.5,
            'sigma_min': s_min,
        })

    # OBL period variations (3)
    for obl_p in [50, 100, 150]:
        configs.append({
            'name': f'obl_period={obl_p}',
            'N_syn': 5,
            'late_strategy': 'de_only',
            'late_de_thresh': 0.8,
            'late_F_lo': 0.1, 'late_F_hi': 0.3,
            'chaos_ratio': 0.5,
            'obl_period': obl_p,
        })

    # Restart patience (3)
    for rp in [30, 50, 70]:
        configs.append({
            'name': f'restart={rp}',
            'N_syn': 5,
            'late_strategy': 'de_only',
            'late_de_thresh': 0.8,
            'late_F_lo': 0.1, 'late_F_hi': 0.3,
            'chaos_ratio': 0.5,
            'restart_patience': rp,
        })

    # Tau center (3)
    for tc in [0.25, 0.35, 0.45]:
        configs.append({
            'name': f'tau={tc}',
            'N_syn': 5,
            'late_strategy': 'de_only',
            'late_de_thresh': 0.8,
            'late_F_lo': 0.1, 'late_F_hi': 0.3,
            'chaos_ratio': 0.5,
            'tau_center': tc,
        })

    # Aggressive combos (10)
    combos = [
        {'N_syn': 3, 'late_de_thresh': 0.6, 'late_F_lo': 0.05, 'late_F_hi': 0.15, 'chaos_ratio': 1.0, 'late_strategy': 'all4'},
        {'N_syn': 7, 'late_de_thresh': 0.7, 'late_F_lo': 0.01, 'late_F_hi': 0.1, 'chaos_ratio': 0.7, 'late_strategy': 'de_levy'},
        {'N_syn': 5, 'late_de_thresh': 0.9, 'late_F_lo': 0.02, 'late_F_hi': 0.08, 'chaos_ratio': 1.0, 'late_strategy': 'de_only'},
        {'N_syn': 3, 'late_de_thresh': 0.8, 'late_F_lo': 0.1, 'late_F_hi': 0.2, 'chaos_ratio': 0.5, 'late_strategy': 'de_spiral', 'sigma_min': 0.0001},
        {'N_syn': 5, 'late_de_thresh': 0.7, 'late_F_lo': 0.05, 'late_F_hi': 0.2, 'chaos_ratio': 0.5, 'late_strategy': 'all4', 'obl_period': 50},
        {'N_syn': 7, 'late_de_thresh': 0.6, 'late_F_lo': 0.1, 'late_F_hi': 0.3, 'chaos_ratio': 0.7, 'late_strategy': 'de_only', 'tau_center': 0.25},
        {'N_syn': 3, 'late_de_thresh': 0.8, 'late_F_lo': 0.05, 'late_F_hi': 0.15, 'chaos_ratio': 1.0, 'late_strategy': 'de_only', 'use_real_distance': True, 'eccentricity': 0.2},
        {'N_syn': 5, 'late_de_thresh': 0.85, 'late_F_lo': 0.03, 'late_F_hi': 0.1, 'chaos_ratio': 0.5, 'late_strategy': 'de_only', 'restart_patience': 30},
        {'N_syn': 5, 'late_de_thresh': 0.8, 'late_F_lo': 0.1, 'late_F_hi': 0.3, 'chaos_ratio': 0.5, 'late_strategy': 'de_only', 'sigma_min': 0.0001, 'obl_period': 50, 'obl_frac': 0.5},
        {'N_syn': 3, 'late_de_thresh': 0.75, 'late_F_lo': 0.05, 'late_F_hi': 0.2, 'chaos_ratio': 0.7, 'late_strategy': 'all4', 'tau_center': 0.3, 'restart_patience': 40},
    ]
    for i, c in enumerate(combos):
        c['name'] = f'combo_{i+1}'
        configs.append(c)

    # Add config index
    for i, c in enumerate(configs):
        c['index'] = i + 1
    return configs


# ============================================================
# Quick benchmark for sweep
# ============================================================
def quick_benchmark(algo, n_runs=3, max_iter=100, dim=5):
    """Quick benchmark on 3 functions, D=5, 100 iter, 3 runs."""
    funcs = [
        ('Sphere', sphere, (-5, 5)),
        ('Rastrigin', rastrigin, (-5.12, 5.12)),
        ('Rosenbrock', rosenbrock, (-2.048, 2.048)),
        ('Ackley', ackley, (-32, 32)),
        ('Schwefel', schwefel, (-500, 500)),
    ]
    results = {}
    for fname, fn, bnds in funcs:
        runs = []
        for seed in range(n_runs):
            _, fb, _ = algo.optimize(fn, dim, bnds, pop=20, max_iter=max_iter, seed=seed)
            runs.append(fb)
        results[fname] = {'mean': float(np.mean(runs)), 'best': float(np.min(runs))}
    return results


def config_score(results, avoa_results):
    """Score = number of functions where config beats AVOA + avg improvement."""
    wins = 0
    improvements = []
    for fname in results:
        if results[fname]['mean'] < avoa_results[fname]['mean']:
            wins += 1
        if avoa_results[fname]['mean'] > 0:
            imp = (avoa_results[fname]['mean'] - results[fname]['mean']) / avoa_results[fname]['mean']
            improvements.append(imp)
    avg_imp = float(np.mean(improvements)) if improvements else 0
    return wins, avg_imp


def run_sweep():
    print("=" * 92)
    print("LUNA v9 Parametric Sweep — 50+ Configurations")
    print("=" * 92)

    configs = generate_configs()
    print(f"\nTotal configurations: {len(configs)}")

    # Baseline: AVOA quick benchmark
    print("\nRunning AVOA baseline (quick benchmark)...")
    avoa = AVOA()
    avoa_results = quick_benchmark(avoa, n_runs=3, max_iter=100, dim=5)
    print(f"AVOA results:")
    for fname, r in avoa_results.items():
        print(f"  {fname}: mean={r['mean']:.4e}, best={r['best']:.4e}")

    # Sweep all configs
    print(f"\nSweeping {len(configs)} configurations...")
    sweep_results = []
    t_start = time.time()

    for i, config in enumerate(configs):
        algo = LUNAv9_parametric(config)
        results = quick_benchmark(algo, n_runs=3, max_iter=100, dim=5)
        wins, avg_imp = config_score(results, avoa_results)
        sweep_results.append({
            'index': config['index'],
            'name': config['name'],
            'config': config,
            'results': results,
            'wins_vs_avoa': wins,
            'avg_improvement': avg_imp,
        })
        elapsed = time.time() - t_start
        marker = " ★" if wins > 0 else ""
        print(f"  [{i+1}/{len(configs)}] {config['name']:<35} wins={wins}/5, avg_imp={avg_imp:+.2f}{marker}  [{elapsed:.0f}s]")

    # Sort by wins (desc) then by avg_improvement (desc)
    sweep_results.sort(key=lambda x: (x['wins_vs_avoa'], x['avg_improvement']), reverse=True)

    print("\n" + "=" * 92)
    print("TOP 10 CONFIGURATIONS (by wins vs AVOA, then avg improvement)")
    print("=" * 92)
    for i, r in enumerate(sweep_results[:10]):
        marker = " ★★★" if r['wins_vs_avoa'] >= 3 else (" ★★" if r['wins_vs_avoa'] >= 2 else (" ★" if r['wins_vs_avoa'] >= 1 else ""))
        print(f"\n  #{i+1} (idx {r['index']}): {r['name']}{marker}")
        print(f"     Wins vs AVOA: {r['wins_vs_avoa']}/5, Avg improvement: {r['avg_improvement']:+.2f}")
        for fname, res in r['results'].items():
            avoa_m = avoa_results[fname]['mean']
            cmp = "WIN" if res['mean'] < avoa_m else "lose"
            print(f"     {fname:<12}: mean={res['mean']:.3e} (AVOA={avoa_m:.3e}) {cmp}")

    # Save sweep results
    with open('/home/z/my-project/download/LUNA_v9_sweep_results.json', 'w') as f:
        json.dump({
            'executed_at': datetime.now().isoformat(),
            'n_configs': len(configs),
            'avoa_baseline': avoa_results,
            'sweep_results': sweep_results,
        }, f, indent=2, default=str)
    print(f"\nSweep results saved: /home/z/my-project/download/LUNA_v9_sweep_results.json")

    return sweep_results, avoa_results


if __name__ == "__main__":
    run_sweep()
