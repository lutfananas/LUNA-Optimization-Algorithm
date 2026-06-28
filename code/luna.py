#!/usr/bin/env python3
"""
LUNA: Lunar-inspired Update & Navigation Algorithm
===================================================
A novel population-based metaheuristic optimization algorithm grounded in
authentic lunar astronomical mechanics.

Three real astronomical cycles drive the algorithm:
  1. Synodic month (phase cycle) — governs exploration/exploitation balance
  2. Anomalistic month (distance cycle) — governs gravitational strength
  3. Tidal force (spring-neap cycle) — secondary gravity modulation

Additional features:
  - Chaotic initialization (Logistic map μ=4) + OBL
  - Hybrid 4-strategy exploitation (triple-best, spiral, Lévy, DE)
  - Late-stage DE convergence for precision
  - Adaptive σ (1/5-success rule)
  - Periodic OBL + stagnation restart + boundary reflection

Author: Lutfan Anas Zahir, Nikken Kusuma Wardani
License: MIT
Repository: https://github.com/lutfananas/LUNA-Optimization-Algorithm
"""

import math
import numpy as np


class LUNA:
    """LUNA optimizer.

    Parameters:
        G0              : baseline gravitational constant (default 2.5)
        N_syn           : number of synodic cycles per run (default 5)
        N_anom          : number of anomalistic cycles per run (default 5)
        eccentricity    : lunar orbital eccentricity (default 0.0549, real Moon)
        sigma_init      : initial exploration step size (default 0.5)
        sigma_min       : minimum step size floor (default 0.001)
        sigma_max       : maximum step size ceiling (default 1.0)
        levy_beta       : Lévy stability parameter (default 1.5)
        late_de_thresh  : iteration fraction for late DE (default 0.8)
        late_F_lo       : late DE scaling factor lower bound (default 0.1)
        late_F_hi       : late DE scaling factor upper bound (default 0.3)
        chaos_ratio     : fraction of chaotic init population (default 0.5)
        p_crossover     : DE crossover probability (default 0.4)
        obl_period      : periodic OBL interval (default 100)
        obl_frac        : OBL fraction of worst population (default 0.3)
        restart_patience: stagnation restart threshold (default 50)
        restart_frac    : restart fraction (default 0.2)
        w_alpha/beta/delta : triple-best weights (0.5, 0.3, 0.2)
        eps             : numerical safety (default 1e-3)
    """

    def __init__(self, G0=2.5, N_syn=5, N_anom=5, eccentricity=0.0549,
                 sigma_init=0.5, sigma_min=0.001, sigma_max=1.0,
                 levy_beta=1.5, late_de_thresh=0.8,
                 late_F_lo=0.1, late_F_hi=0.3,
                 chaos_ratio=0.5, p_crossover=0.4,
                 obl_period=100, obl_frac=0.3,
                 restart_patience=50, restart_frac=0.2,
                 w_alpha=0.5, w_beta=0.3, w_delta=0.2,
                 eps=1e-3):
        self.G0 = G0
        self.N_syn = N_syn
        self.N_anom = N_anom
        self.eccentricity = eccentricity
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.levy_beta = levy_beta
        self.late_de_thresh = late_de_thresh
        self.late_F_lo = late_F_lo
        self.late_F_hi = late_F_hi
        self.chaos_ratio = chaos_ratio
        self.p_crossover = p_crossover
        self.obl_period = obl_period
        self.obl_frac = obl_frac
        self.restart_patience = restart_patience
        self.restart_frac = restart_frac
        self.w_alpha = w_alpha
        self.w_beta = w_beta
        self.w_delta = w_delta
        self.eps = eps

    # === Lunar Astronomy ===

    def _phase_angle(self, t, T):
        """Synodic phase angle: θ_p = 2π · N_syn · t / T"""
        return 2 * np.pi * self.N_syn * t / T

    def _illumination(self, theta):
        """I(θ) = (1 - cos θ) / 2  ∈ [0, 1]"""
        return (1 - np.cos(theta)) / 2

    def _tidal_force(self, theta):
        """T(θ) = (1 + cos 2θ) / 2  ∈ [0, 1]"""
        return (1 + np.cos(2 * theta)) / 2

    def _G(self, t, T):
        """Combined gravitational constant with inverse-square law + tidal modulation."""
        theta_p = self._phase_angle(t, T)
        theta_a = 2 * np.pi * self.N_anom * t / T
        D = 1 + self.eccentricity * np.cos(theta_a)
        G_eff = self.G0 / (D ** 2)  # inverse-square law
        T_force = self._tidal_force(theta_p)
        return G_eff * (0.3 + 0.7 * T_force)

    # === Utility Functions ===

    def _levy(self, dim):
        """Mantegna's Lévy flight sampler."""
        beta = self.levy_beta
        sigma = (math.gamma(1 + beta) * math.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, 1, dim) * sigma
        v = np.abs(np.random.normal(0, 1, dim))
        return u / (v + 1e-30) ** (1 / beta)

    def _chaotic_init(self, n, dim, lb, ub):
        """Logistic map (μ=4) chaotic initialization."""
        chaotic = np.zeros((n, dim))
        for i in range(n):
            x = 0.5
            for j in range(dim):
                x = 4 * x * (1 - x)
                chaotic[i, j] = lb + x * (ub - lb)
        return chaotic

    def _reflect_bounds(self, x, lb, ub):
        """Boundary reflection instead of clipping."""
        for _ in range(3):
            over = x > ub
            under = x < lb
            if not over.any() and not under.any():
                break
            x = np.where(over, 2 * ub - x, x)
            x = np.where(under, 2 * lb - x, x)
        return np.clip(x, lb, ub)

    # === Main Optimization ===

    def optimize(self, f, dim, bounds, pop=30, max_iter=500, seed=None):
        """
        Optimize objective function f.

        Parameters:
            f        : objective function f(x) -> float
            dim      : problem dimension
            bounds   : (lb, ub) tuple
            pop      : population size (default 30)
            max_iter : maximum iterations (default 500)
            seed     : random seed for reproducibility

        Returns:
            (best_x, best_f, history) tuple
        """
        if seed is not None:
            np.random.seed(seed)

        lb, ub = bounds

        # === Chaotic + Random + OBL Initialization ===
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

        pbest = X.copy()
        pbest_fit = fit.copy()
        bi = np.argmin(fit)
        xb = X[bi].copy()
        fb = fit[bi]
        history = [fb]

        sigma = self.sigma_init
        success_window = []
        last_improvement = 0
        last_obl_iter = 0
        var_history = []

        for t in range(max_iter):
            # Lunar parameters
            theta_p = self._phase_angle(t, max_iter)
            I = self._illumination(theta_p)
            G_t = self._G(t, max_iter)
            is_late_de = (t / max_iter) > self.late_de_thresh

            # Landscape detection
            if len(var_history) >= 10:
                mean_var = np.mean(var_history[-10:])
                is_multimodal = mean_var / (abs(fb) + 1e-10) > 0.3
            else:
                is_multimodal = True

            # Top-3 solutions
            order_fit = np.argsort(fit)
            alpha = X[order_fit[0]].copy()
            beta_sol = X[order_fit[1]].copy() if len(order_fit) > 1 else xb
            delta_sol = X[order_fit[2]].copy() if len(order_fit) > 2 else xb

            iter_improved = False

            for i in range(pop):
                # === Late-stage DE convergence ===
                if is_late_de:
                    cands = [j for j in range(pop) if j != i]
                    a, b = np.random.choice(cands, 2, replace=False)
                    F_de = np.random.uniform(self.late_F_lo, self.late_F_hi)
                    if np.random.random() < 0.5:
                        xn = X[i] + F_de * (xb - X[i]) + F_de * (X[a] - X[b])
                    else:
                        D = np.abs(xb - X[i])
                        l = np.random.uniform(-1, 1, dim)
                        xn = D * np.exp(l) * np.cos(2 * np.pi * l) + xb
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

                # === DE crossover ===
                if np.random.random() < self.p_crossover:
                    cands = [j for j in range(pop) if j != i]
                    a, b = np.random.choice(cands, 2, replace=False)
                    F_de = np.random.uniform(0.4, 0.9)
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

                # === Lunar phase-based strategy ===
                if I < 0.2:
                    # NEW MOON: pure exploration
                    if is_multimodal:
                        step = sigma * self._levy(dim)
                    else:
                        step = np.random.normal(0, sigma * 0.3, dim)
                    xn = X[i] + step

                elif I >= 0.8:
                    # FULL MOON: hybrid 4-strategy exploitation
                    R_a = np.linalg.norm(alpha - X[i]) + self.eps
                    R_b = np.linalg.norm(beta_sol - X[i]) + self.eps
                    R_d = np.linalg.norm(delta_sol - X[i]) + self.eps
                    R_p = np.linalg.norm(pbest[i] - X[i]) + self.eps
                    xi = np.random.uniform(0.7, 1.0)
                    strategy = np.random.randint(1, 5)

                    if strategy == 1:
                        # S1: Triple-best pull
                        pull = (self.w_alpha * (alpha - X[i]) / R_a
                                + self.w_beta * (beta_sol - X[i]) / R_b
                                + self.w_delta * (delta_sol - X[i]) / R_d
                                + 0.4 * (pbest[i] - X[i]) / R_p)
                        xn = X[i] + xi * G_t * pull
                        xn = xn + np.random.normal(0, sigma * 0.05, dim)
                    elif strategy == 2:
                        # S2: Spiral (WOA-style)
                        D = np.abs(alpha - X[i])
                        l = np.random.uniform(-1, 1, dim)
                        xn = D * np.exp(l) * np.cos(2 * np.pi * l) + alpha
                    elif strategy == 3:
                        # S3: Lévy around best (HHO-style)
                        xn = alpha - np.abs(2 * np.random.random(dim) * alpha - X[i]) * (xi * G_t * 0.1)
                        xn = xn + sigma * self._levy(dim) * 0.5
                    else:
                        # S4: DE/current-to-best/1
                        cands = [j for j in range(pop) if j != i]
                        a, b = np.random.choice(cands, 2, replace=False)
                        F_de = np.random.uniform(0.3, 0.6)
                        xn = X[i] + F_de * (alpha - X[i]) + F_de * (X[a] - X[b])

                else:
                    # TRANSITIONAL: weighted mix
                    w_explore = 1 - I
                    w_exploit = I
                    if is_multimodal:
                        exp_step = sigma * self._levy(dim) * 0.5
                    else:
                        exp_step = np.random.normal(0, sigma * 0.3, dim) * 0.5
                    R_a = np.linalg.norm(alpha - X[i]) + self.eps
                    R_b = np.linalg.norm(beta_sol - X[i]) + self.eps
                    R_d = np.linalg.norm(delta_sol - X[i]) + self.eps
                    R_p = np.linalg.norm(pbest[i] - X[i]) + self.eps
                    xi = np.random.uniform(0.5, 1.0)
                    pull = (self.w_alpha * (alpha - X[i]) / R_a
                            + self.w_beta * (beta_sol - X[i]) / R_b
                            + self.w_delta * (delta_sol - X[i]) / R_d
                            + 0.4 * (pbest[i] - X[i]) / R_p)
                    exploit_step = xi * G_t * pull
                    xn = X[i] + w_explore * exp_step + w_exploit * exploit_step

                xn = self._reflect_bounds(xn, lb, ub)
                fn = f(xn)

                if fn < fit[i]:
                    X[i] = xn; fit[i] = fn
                    iter_improved = True
                    if fn < pbest_fit[i]:
                        pbest[i] = xn.copy(); pbest_fit[i] = fn
                    if fn < fb:
                        fb = fn; xb = xn.copy(); last_improvement = t

            # Adaptive sigma (1/5-success rule)
            success_window.append(1 if iter_improved else 0)
            if len(success_window) > 20:
                success_window.pop(0)
            if len(success_window) == 20:
                sr = sum(success_window) / 20
                if sr > 0.2:
                    sigma *= 1.1
                elif sr < 0.2:
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
                    pbest[idx] = X[idx].copy()
                    pbest_fit[idx] = fit[idx]
                    if fit[idx] < fb:
                        fb = fit[idx]; xb = X[idx].copy()
                        last_improvement = t

            history.append(fb)

        return xb, fb, history


# === Quick Demo ===
if __name__ == "__main__":
    # Example: optimize Sphere function
    def sphere(x):
        return float(np.sum(x ** 2))

    luna = LUNA()
    best_x, best_f, history = luna.optimize(
        sphere, dim=10, bounds=(-100, 100),
        pop=30, max_iter=500, seed=42
    )
    print(f"LUNA Optimization Result:")
    print(f"  Best fitness: {best_f:.6e}")
    print(f"  Best solution: {best_x[:5]}... (showing first 5 dims)")
    print(f"  Converged from {history[0]:.4e} to {history[-1]:.4e}")
