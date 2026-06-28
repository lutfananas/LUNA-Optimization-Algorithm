#!/usr/bin/env python3
"""
Ablation study with Wilcoxon signed-rank tests between variants.

Variants:
  LUNA_full      : Full algorithm (Config E)
  LUNA_no_chaos  : Chaotic init replaced with uniform random
  LUNA_no_OBL    : No opposition-based learning
  LUNA_no_restart: No restart mechanism
  LUNA_no_lateDE : No late-stage DE-like phase (always use astronomical ops)
  LUNA_no_pbest  : No personal-best memory
  LUNA_no_astronomy : All astronomy variables set to constants (the "kill test")

Runs each variant on 12 CEC 2022 functions × 20 runs (D=10, 500 iter),
then computes pairwise Wilcoxon signed-rank tests with Holm correction.
"""
import os, sys, json, math, time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import wilcoxon
from itertools import combinations

sys.path.insert(0, '/home/z/my-project/scripts')
from cec2022_benchmark import CEC2022
from luna_v7_class import LUNA_v7

OUT = '/home/z/my-project/download'
CKPT = f'{OUT}/LUNA_ablation_checkpoint.json'
RESULT = f'{OUT}/LUNA_ablation_wilcoxon.json'

# Full config (Config E)
BASE_CFG = dict(G0=5.0, N_syn=5, N_anom=5, eccentricity=0.0549,
                semi_major=1.0, ang_momentum=0.5, libration_amp=0.12,
                libration_freq=3, sun_gravity=0.3, orbital_decay=8.0,
                sigma_init=0.5, sigma_min=0.001, sigma_max=1.0,
                late_de_thresh=0.6, F_lo=0.02, F_hi=0.08,
                chaos_ratio=0.5, obl_period=100, obl_frac=0.3,
                restart_patience=50, restart_frac=0.2, pbest_weight=0.4, eps=1e-10)


class LUNA_no_chaos(LUNA_v7):
    """Replace chaotic init with pure uniform random."""
    def _ci(self, n, d, lb, ub):
        return np.random.uniform(lb, ub, (n, d))


class LUNA_no_OBL(LUNA_v7):
    """Disable OBL by setting obl_period to a huge number."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.obl_period = 10**9


class LUNA_no_restart(LUNA_v7):
    """Disable restart by setting patience huge."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.restart_patience = 10**9


class LUNA_no_lateDE(LUNA_v7):
    """Disable late-stage DE: never enter 'is_late' regime."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.late_de_thresh = 2.0  # ratio never exceeds 1.0, so always False


class LUNA_no_pbest(LUNA_v7):
    """Disable personal-best memory."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.pbest_weight = 0.0


class LUNA_no_astronomy(LUNA_v7):
    """Set all astronomical variables to constants (the kill test).

    - I = 0.5 (constant illumination)
    - D = 1.0 (constant distance, no anomalistic cycle)
    - lib = 0 (no libration)
    - tidal = constant
    - v_factor depends on D and mu only, but mu still decays.

    This isolates the contribution of the astronomical cycles.
    """
    def optimize(self, f, dim, bounds, pop=30, max_iter=500, seed=None):
        if seed is not None:
            np.random.seed(seed)
        lb, ub = bounds
        G0 = self.G0; od = self.orbital_decay
        si = self.sigma_init; sm = self.sigma_min; sx = self.sigma_max
        ldt = self.late_de_thresh; Fl = self.F_lo; Fh = self.F_hi
        cr = self.chaos_ratio; op = self.obl_period; of = self.obl_frac
        rp = self.restart_patience; rf = self.restart_frac
        pw = self.pbest_weight; eps = self.eps

        # ASTRONOMY KILLED: I=0.5, D=1.0, lib=0, kr=0
        I_const = 0.5
        D_const = 1.0
        lib_const = 0.0
        Gt_const = G0  # no D modulation
        # v_factor: vis-viva still uses D=1, but energy = 2/1 - 1/1 = 1
        # so v = sqrt(G0_t), vf = v/(v+1)
        # But this still depends on G0_t decay.
        # For "pure kill": set v_factor = 1.0 (constant)
        vf_const = 1.0
        kr_const = 0.0  # no angular rate
        tidal_const = G0  # no sun gravity

        nc = int(pop * cr); nr = pop - nc
        Xc = self._ci(nc, dim, lb, ub); Xr = np.random.uniform(lb, ub, (nr, dim))
        Xi = np.vstack([Xc, Xr]); Xo = (lb + ub) - Xi
        Xc2 = np.vstack([Xi, Xo]); fc = np.array([f(x) for x in Xc2])
        order = np.argsort(fc)[:pop]; X = Xc2[order].copy(); fit = fc[order].copy()
        pb = X.copy(); pbf = fit.copy()
        bi = np.argmin(fit); xb = X[bi].copy(); fb = fit[bi]; hh = [fb]
        sigma = si; sw = []; li = 0; lo = 0

        for t in range(max_iter):
            ratio = t / max_iter; is_late = ratio > ldt
            G0_t = G0 * math.exp(-od * ratio)
            # All astronomical quantities replaced by constants
            we = I_const; wx = 1 - I_const; ii = False

            for i in range(pop):
                diff = xb - X[i]; R = np.linalg.norm(diff) + eps; d = diff / R
                dp_v = pb[i] - X[i]; Rp = np.linalg.norm(dp_v) + eps; dp = dp_v / Rp
                if is_late:
                    F = np.random.uniform(Fl, Fh) * vf_const
                    cands = [j for j in range(pop) if j != i]
                    a2, b2 = np.random.choice(cands, 2, replace=False)
                    diff_ab = X[a2] - X[b2]
                    xn = X[i] + F * diff + F * Gt_const * diff_ab + pw * F * dp_v
                    if np.random.random() < 0.5:
                        D_abs = np.abs(xb - X[i]); l = np.random.uniform(-1, 1, dim)
                        # kr_const = 0 means kr2 = 0, exp(0)=1, cos(0)=1
                        xn = D_abs * l * 0 + xb  # degenerates to xb
                elif I_const < 0.2:  # = False since I_const=0.5
                    rv = np.random.randn(dim); orth = rv - np.dot(rv, d) * d
                    orth /= (np.linalg.norm(orth) + eps)
                    xn = X[i] + lib_const * R * orth + sigma * np.random.randn(dim)
                elif I_const >= 0.8:  # = False
                    pass  # never reached
                else:  # main branch (I_const=0.5)
                    cands = [j for j in range(pop) if j != i]
                    a2, b2 = np.random.choice(cands, 2, replace=False)
                    diff_ab = X[a2] - X[b2]; sa = sigma * vf_const
                    exploit = sa * diff + sigma * Gt_const * diff_ab
                    rv = np.random.randn(dim); orth = rv - np.dot(rv, d) * d
                    orth /= (np.linalg.norm(orth) + eps)
                    explore = lib_const * R * orth + sigma * np.random.randn(dim) * 0.5
                    xn = X[i] + we * exploit + wx * explore
                xn = self._rb(xn, lb, ub); fn = f(xn)
                if fn < fit[i]:
                    X[i] = xn; fit[i] = fn; ii = True
                    if fn < pbf[i]: pb[i] = xn.copy(); pbf[i] = fn
                    if fn < fb: fb = fn; xb = xn.copy(); li = t

            if not is_late:
                sw.append(1 if ii else 0)
                if len(sw) > 20: sw.pop(0)
                if len(sw) == 20:
                    sr = sum(sw) / 20
                    if sr > 0.2: sigma *= 1.1
                    elif sr < 0.2: sigma *= 0.9
                    sigma = max(sm, min(sx, sigma))
                if t - lo >= op and t > 0:
                    lo = t; n_obl = max(1, int(pop * of))
                    worst = np.argsort(fit)[-n_obl:]; opp = (lb + ub) - X[worst]
                    of3 = np.array([f(x) for x in opp])
                    for k, idx in enumerate(worst):
                        if of3[k] < fit[idx]:
                            X[idx] = opp[k]; fit[idx] = of3[k]
                            if of3[k] < pbf[idx]: pb[idx] = opp[k].copy(); pbf[idx] = of3[k]
                            if of3[k] < fb: fb = of3[k]; xb = opp[k].copy(); li = t
                if t - li >= rp:
                    n_r = max(1, int(pop * rf)); worst = np.argsort(fit)[-n_r:]
                    for idx in worst:
                        X[idx] = np.random.uniform(lb, ub, dim); fit[idx] = f(X[idx])
                        pb[idx] = X[idx].copy(); pbf[idx] = fit[idx]
                        if fit[idx] < fb: fb = fit[idx]; xb = X[idx].copy(); li = t
            hh.append(fb)
        return xb, fb, hh


VARIANTS = {
    "LUNA_full":       lambda: LUNA_v7(**BASE_CFG),
    "LUNA_no_chaos":   lambda: LUNA_no_chaos(**BASE_CFG),
    "LUNA_no_OBL":     lambda: LUNA_no_OBL(**BASE_CFG),
    "LUNA_no_restart": lambda: LUNA_no_restart(**BASE_CFG),
    "LUNA_no_lateDE":  lambda: LUNA_no_lateDE(**BASE_CFG),
    "LUNA_no_pbest":   lambda: LUNA_no_pbest(**BASE_CFG),
    "LUNA_no_astronomy": lambda: LUNA_no_astronomy(**BASE_CFG),
}


def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return None


def save_ckpt(ckpt):
    with open(CKPT, 'w') as f:
        json.dump(ckpt, f, indent=2)


def run(n_runs=20, max_iter=500, pop_size=30, dim=10):
    cec = CEC2022(dim=dim, seed=42)
    functions = cec.get_all_functions()
    bounds = (cec.lb, cec.ub)
    fn_names = list(functions.keys())
    var_names = list(VARIANTS.keys())

    ckpt = load_ckpt()
    if ckpt is None:
        ckpt = {
            "dim": dim, "n_runs": n_runs, "max_iter": max_iter, "pop_size": pop_size,
            "started_at": datetime.now().isoformat(),
            "completed": [],
            "results": {fn: {v: [] for v in var_names} for fn in fn_names},
        }
        save_ckpt(ckpt)
    else:
        print(f"Resuming ablation: {len(ckpt['completed'])}/{len(fn_names)} functions done")

    todo = [fn for fn in fn_names if fn not in ckpt['completed']]
    print(f"Variants: {var_names}")
    print(f"Functions remaining: {todo}")

    t_start = time.time()
    runs_done = sum(len(ckpt['results'][fn][v]) for fn in fn_names for v in var_names)
    total_runs = len(fn_names) * len(var_names) * n_runs

    for fname in todo:
        func = functions[fname]
        print(f"\n>>> {fname}", flush=True)

        for vname in var_names:
            if len(ckpt['results'][fname][vname]) >= n_runs:
                continue
            for r in range(n_runs):
                if len(ckpt['results'][fname][vname]) >= n_runs:
                    break
                seed = 42 + r
                try:
                    algo = VARIANTS[vname]()
                    _, fb, _ = algo.optimize(func, dim, bounds, pop=pop_size,
                                              max_iter=max_iter, seed=seed)
                except Exception as e:
                    print(f"    ERROR {vname} run {r}: {e}", flush=True)
                    fb = float('nan')
                ckpt['results'][fname][vname].append(float(fb))
                runs_done += 1
                if runs_done % 10 == 0:
                    elapsed = time.time() - t_start
                    print(f"  {vname:>20s}[{r+1}/{n_runs}]: {fb:.3e}  "
                          f"[{runs_done}/{total_runs}, elapsed={elapsed:.0f}s]",
                          flush=True)
            save_ckpt(ckpt)

        ckpt['completed'].append(fname)
        ckpt['last_updated'] = datetime.now().isoformat()
        save_ckpt(ckpt)

    # Compute pairwise Wilcoxon tests
    print(f"\n{'='*80}\nPairwise Wilcoxon Signed-Rank Tests (with Holm correction)\n{'='*80}")
    p_matrix = {v1: {v2: None for v2 in var_names} for v1 in var_names}

    # For each function, compute Wilcoxon between each pair, then aggregate
    all_p_values = []  # list of (v1, v2, fn, p)
    for fn in fn_names:
        for v1, v2 in combinations(var_names, 2):
            arr1 = np.array(ckpt['results'][fn][v1])
            arr2 = np.array(ckpt['results'][fn][v2])
            n = min(len(arr1), len(arr2))
            if n < 5:
                p = 1.0
            else:
                try:
                    _, p = wilcoxon(arr1[:n], arr2[:n])
                except Exception:
                    p = 1.0
            all_p_values.append((v1, v2, fn, float(p)))

    # Aggregate: count wins/losses/ties per pair
    pair_summary = {}
    for v1, v2 in combinations(var_names, 2):
        wins_v1, wins_v2, ties = 0, 0, 0
        p_vals = []
        for fn in fn_names:
            arr1 = np.array(ckpt['results'][fn][v1])
            arr2 = np.array(ckpt['results'][fn][v2])
            m1, m2 = np.mean(arr1), np.mean(arr2)
            # find p for this pair
            p = next((pv for (a, b, f, pv) in all_p_values
                      if a == v1 and b == v2 and f == fn), 1.0)
            p_vals.append(p)
            if p < 0.05:
                if m1 < m2: wins_v1 += 1
                else: wins_v2 += 1
            else: ties += 1
        # Holm correction across the 12 functions for this pair
        sorted_p = sorted(p_vals)
        holm_p = [min(sorted_p[i] * (len(p_vals) - i), 1.0) for i in range(len(p_vals))]
        min_holm_p = min(holm_p)

        pair_summary[f"{v1}_vs_{v2}"] = {
            "wins_v1": wins_v1, "wins_v2": wins_v2, "ties": ties,
            "min_raw_p": float(min(p_vals)),
            "min_holm_p": float(min_holm_p),
            "any_significant": bool(min_holm_p < 0.05),
        }
        print(f"  {v1} vs {v2}: v1_wins={wins_v1}, v2_wins={wins_v2}, ties={ties}, "
              f"min_p={min(p_vals):.2e}, holm_p={min_holm_p:.2e}, "
              "sig" if min_holm_p < 0.05 else "ns")

    # Mean fitness per variant per function
    means = {fn: {v: float(np.mean(ckpt['results'][fn][v])) for v in var_names}
             for fn in fn_names}

    # Overall rank
    avg_fitness = {v: float(np.mean([means[fn][v] for fn in fn_names])) for v in var_names}
    print(f"\nOverall mean fitness (lower = better):")
    for v, m in sorted(avg_fitness.items(), key=lambda x: x[1]):
        print(f"  {v:>22s}: {m:.4e}")

    # Save
    report = {
        "executed_at": datetime.now().isoformat(),
        "dim": dim, "n_runs": n_runs, "max_iter": max_iter, "pop_size": pop_size,
        "variants": var_names,
        "per_function_means": means,
        "overall_mean_fitness": avg_fitness,
        "pairwise_wilcoxon_with_holm": pair_summary,
        "raw_results": {fn: {v: list(ckpt['results'][fn][v]) for v in var_names}
                         for fn in fn_names},
    }
    with open(RESULT, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {RESULT}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--iter', type=int, default=500)
    ap.add_argument('--pop', type=int, default=30)
    ap.add_argument('--dim', type=int, default=10)
    args = ap.parse_args()
    run(args.runs, args.iter, args.pop, args.dim)
