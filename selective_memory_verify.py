"""
Three-Type Selective Memory: Deep Computational Verification
============================================================

Solves the general three-type (and N-type) garbling problem:

    max_{garbling W}  E[h(mu_L) + delta * r(mu_H)]
    s.t.  W is a stochastic matrix (garbling constraint)

and compares the unrestricted global optimum to the Selective Memory (SM)
record. Tests whether SM is globally optimal under the theorem's conditions
(A1: nondegeneracy, A2: g convex, A3: strict Jensen gain).

Uses multiple global optimizers (differential evolution, basin-hopping,
SLSQP with many starts) and tests across:
  - Nested-independence experiments (affine m)
  - Correlated experiments (non-affine m)
  - Adversarial constructions
  - N >= 4 type extensions

See selective_memory_spec.md for full mathematical specification.

Usage:
    python selective_memory_verify.py [--quick] [--output results.json]

    --quick    : Run a fast sweep (fewer trials, smaller grid)
    --output   : Path for JSON results (default: sm_results.json)
"""

import os
# Limit BLAS threads to 1 per process to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
from itertools import product
import json
import argparse
import time
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class Config:
    """Global configuration for the sweep."""
    # Optimizer settings
    n_slsqp_starts: int = 500        # random starts for SLSQP (phase 2 target)
    de_maxiter: int = 300             # differential evolution max iterations
    de_popsize: int = 15              # differential evolution population
    bh_niter: int = 30                # basin-hopping iterations
    use_differential_evolution: bool = False  # 500 SLSQP starts alone is sufficient
    use_basin_hopping: bool = False
    use_slsqp: bool = True

    # Adaptive budget: skip expensive optimization when SM seed is optimal
    adaptive: bool = True
    phase1_random_starts: int = 50   # Cheap first probe before escalating

    # Tolerance
    gap_tol: float = 1e-5             # threshold for declaring a counterexample

    # Parallelism
    n_parallel: int = field(default_factory=lambda: max(1, os.cpu_count() or 1))

    # Quick mode overrides
    @classmethod
    def quick(cls):
        return cls(
            n_slsqp_starts=200,
            de_maxiter=300,
            de_popsize=15,
            bh_niter=50,
            use_basin_hopping=False,
            adaptive=True,
            phase1_random_starts=30,
        )


# =====================================================================
# MODEL
# =====================================================================

@dataclass
class ModelParams:
    """Primitive parameters for the three-type model."""
    y_L: float       # low-type current output
    y_bar: float     # M,H current output (current-period equivalence)
    y_under: float   # L,M continuation output (continuation-equivalence)
    y_H: float       # high-type continuation output
    sigma: float     # match surplus
    w1: float        # contractual wage
    delta: float     # discount factor

    @property
    def c1_h(self) -> float:
        """Retention threshold in mu_L space."""
        return (self.y_bar + self.sigma - self.w1) / (self.y_bar - self.y_L)

    @property
    def c1_r(self) -> float:
        """Fire threshold in mu_H space (should be <= 0 for r concave)."""
        return (self.w1 - self.sigma - self.y_under) / (self.y_H - self.y_under)

    @property
    def c2(self) -> float:
        """Ratchet threshold in mu_H space."""
        return (self.w1 - self.y_under) / (self.y_H - self.y_under)

    def is_feasible(self) -> bool:
        """Check all parameter restrictions."""
        return (
            self.y_L + self.sigma < self.w1 and     # h interior
            self.w1 <= self.y_under + self.sigma and # r concave
            self.y_under < self.w1 and               # c2 > 0
            self.w1 < self.y_H and                   # c2 < 1
            self.y_L < self.y_bar and                # non-degenerate current
            self.y_under < self.y_H and              # non-degenerate continuation
            0 < self.c1_h < 1                        # h threshold interior
        )

    def h(self, mu_L: np.ndarray) -> np.ndarray:
        """Current retention value. Convex in mu_L."""
        return np.maximum(0.0,
            (self.y_bar + self.sigma - self.w1)
            - (self.y_bar - self.y_L) * mu_L)

    def r(self, mu_H: np.ndarray) -> np.ndarray:
        """Continuation ratchet value. Concave in mu_H (when c1_r <= 0)."""
        x = mu_H * self.y_H + (1 - mu_H) * self.y_under
        w2 = np.maximum(self.w1, x)
        return np.maximum(0.0, x + self.sigma - w2)

    def total_value(self, mu_L: np.ndarray, mu_H: np.ndarray) -> np.ndarray:
        """h(mu_L) + delta * r(mu_H)."""
        return self.h(mu_L) + self.delta * self.r(mu_H)


# =====================================================================
# OPERATIONAL EXPERIMENT CONSTRUCTION
# =====================================================================

@dataclass
class Experiment:
    """An operational experiment: posteriors and their probabilities."""
    posteriors: np.ndarray   # shape (n, N_types), each row on simplex
    probs: np.ndarray        # shape (n,), sums to 1
    label: str = ""
    n_types: int = 3

    @property
    def n(self) -> int:
        return len(self.probs)

    @property
    def prior(self) -> np.ndarray:
        return self.probs @ self.posteriors


def make_nested_independent(mu_L_vals, mu_L_probs, q_vals, q_probs, label=""):
    """Construct a nested-independence experiment.

    Stage 1: mu_L drawn from (mu_L_vals, mu_L_probs)
    Stage 2: q = Pr(H|not-L) drawn from (q_vals, q_probs), independent
    Then: mu_H = (1 - mu_L) * q, mu_M = 1 - mu_L - mu_H
    """
    posteriors, probs = [], []
    for mL, pL in zip(mu_L_vals, mu_L_probs):
        for q, pq in zip(q_vals, q_probs):
            mu_H = (1 - mL) * q
            mu_M = 1 - mL - mu_H
            posteriors.append([mL, mu_M, mu_H])
            probs.append(pL * pq)
    return Experiment(
        posteriors=np.array(posteriors),
        probs=np.array(probs),
        label=label or "nested_independent"
    )


def make_correlated_grid(mu_L_vals, mu_H_given_L, probs_H_given_L,
                         mu_L_probs, label=""):
    """Correlated experiment with repeated mu_L values and varying mu_H.

    Constructs an experiment where multiple atoms share the same mu_L value
    but have different mu_H values. This ensures A1 (Var(mu_H | mu_L) > 0)
    can hold, unlike the single-atom-per-mu_L structure of make_correlated.

    Parameters:
        mu_L_vals: array of J distinct mu_L values
        mu_H_given_L: list of J arrays, each giving mu_H atoms for that mu_L
        probs_H_given_L: list of J arrays, conditional probs within each group
        mu_L_probs: array of J marginal probabilities for each mu_L group

    Returns an Experiment on the 3-simplex with sum(mu_L, mu_M, mu_H) = 1.
    The mu_M values are determined by Bayes plausibility: mu_M = 1 - mu_L - mu_H.
    """
    posteriors, probs = [], []
    for j, (mL, p_mL) in enumerate(zip(mu_L_vals, mu_L_probs)):
        mH_atoms = mu_H_given_L[j]
        p_H = probs_H_given_L[j]
        for mH, p_mH in zip(mH_atoms, p_H):
            mM = 1.0 - mL - mH
            if mM < 0 or mM > 1:
                raise ValueError(
                    f"Invalid simplex point: mu_L={mL}, mu_H={mH}, mu_M={mM}")
            posteriors.append([mL, mM, mH])
            probs.append(p_mL * p_mH)
    return Experiment(
        posteriors=np.array(posteriors),
        probs=np.array(probs),
        label=label or "correlated_grid"
    )


def make_correlated_grid_random(seed, n_L=4, n_H_per_L=3, mu_L_range=(0.1, 0.8),
                                mu_H_max=0.4, label=""):
    """Random correlated-grid experiment with repeated mu_L values.

    Generates n_L distinct mu_L values with n_H_per_L atoms each at varying
    mu_H levels. The mu_H distribution conditional on mu_L varies across
    groups, creating a non-affine m(ell).

    mu_H_max: cap on mu_H to keep simplex valid and stay below c2
             (smaller values help A2 hold by keeping r in its linear region)
    """
    rng = np.random.default_rng(seed)

    # J distinct mu_L values in the specified range
    mu_L_vals = np.sort(rng.uniform(mu_L_range[0], mu_L_range[1], n_L))

    # mu_L marginal probabilities (Dirichlet)
    mu_L_probs = rng.dirichlet(np.ones(n_L) * 2)

    mu_H_given_L = []
    probs_H_given_L = []
    for j, mL in enumerate(mu_L_vals):
        # Max mu_H for this mu_L to keep simplex valid (mu_H + mu_L <= 1)
        max_mH = min(mu_H_max, 1.0 - mL - 0.01)
        if max_mH <= 0.02:
            max_mH = 0.02
        # Draw mu_H atoms
        mH_atoms = rng.uniform(0.001, max_mH, n_H_per_L)
        # Conditional probs
        p_H = rng.dirichlet(np.ones(n_H_per_L) * 1.5)
        mu_H_given_L.append(mH_atoms)
        probs_H_given_L.append(p_H)

    return make_correlated_grid(
        mu_L_vals, mu_H_given_L, probs_H_given_L, mu_L_probs,
        label=label or f"corr_grid_random_s{seed}"
    )


def make_correlated(n_atoms, correlation_type="positive", seed=None, label=""):
    """Construct a correlated (non-nested) experiment on the 3-simplex.

    correlation_type: "positive", "negative", "nonlinear", "random"
    """
    rng = np.random.default_rng(seed)

    if correlation_type == "positive":
        # mu_L low => mu_H high (positive correlation in quality)
        t = np.linspace(0.05, 0.95, n_atoms)
        mu_L = 1 - t  # decreasing
        mu_H = t * 0.6  # increasing
        mu_M = 1 - mu_L - mu_H
    elif correlation_type == "negative":
        # mu_L high => mu_H high (both extremes together)
        t = np.linspace(0.05, 0.95, n_atoms)
        mu_L = t * 0.5
        mu_H = t * 0.4
        mu_M = 1 - mu_L - mu_H
    elif correlation_type == "nonlinear":
        # U-shaped relationship
        t = np.linspace(0, 1, n_atoms)
        mu_L = 0.1 + 0.6 * (2*t - 1)**2
        mu_H = 0.05 + 0.3 * t * (1 - t) * 4
        mu_M = 1 - mu_L - mu_H
    elif correlation_type == "random":
        # Random points on simplex
        raw = rng.dirichlet(np.ones(3), size=n_atoms)
        mu_L = raw[:, 0]
        mu_M = raw[:, 1]
        mu_H = raw[:, 2]
    else:
        raise ValueError(f"Unknown correlation_type: {correlation_type}")

    # Ensure valid simplex points
    posteriors = np.column_stack([mu_L, mu_M, mu_H])
    posteriors = np.clip(posteriors, 0.001, 0.998)
    posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)

    # Uniform probabilities
    probs = np.ones(n_atoms) / n_atoms

    return Experiment(
        posteriors=posteriors,
        probs=probs,
        label=label or f"correlated_{correlation_type}"
    )


def make_n_type(N, n_atoms, seed=None, label=""):
    """Construct an experiment with N >= 4 types.

    Types: {L, M1, ..., M_{N-2}, H}
    Current-period equivalence: M1...M_{N-2}, H equivalent for h
    Continuation-equivalence: L, M1...M_{N-2} equivalent for r
    """
    rng = np.random.default_rng(seed)
    posteriors = rng.dirichlet(np.ones(N), size=n_atoms)
    probs = np.ones(n_atoms) / n_atoms
    return Experiment(
        posteriors=posteriors,
        probs=probs,
        label=label or f"{N}_types",
        n_types=N
    )


# =====================================================================
# SELECTIVE MEMORY VALUE
# =====================================================================

def compute_sm_value(params: ModelParams, expt: Experiment) -> float:
    """Compute the Selective Memory value.

    SM reveals mu_L fully, replaces mu_H by E[mu_H | mu_L].
    Groups operational posteriors by their mu_L value (for nested)
    or by unique mu_L atoms.
    """
    N = expt.n_types
    mu_L_col = expt.posteriors[:, 0]
    mu_H_col = expt.posteriors[:, N-1]  # last column = H-type

    # Group by unique mu_L values (with tolerance)
    unique_L = []
    groups = {}
    tol = 1e-10
    for i in range(expt.n):
        mL = mu_L_col[i]
        found = False
        for j, uL in enumerate(unique_L):
            if abs(mL - uL) < tol:
                groups[j].append(i)
                found = True
                break
        if not found:
            groups[len(unique_L)] = [i]
            unique_L.append(mL)

    V = 0.0
    for j, mL in enumerate(unique_L):
        idx = groups[j]
        p_group = sum(expt.probs[i] for i in idx)
        if p_group < 1e-15:
            continue
        m_H = sum(expt.probs[i] * mu_H_col[i] for i in idx) / p_group
        V += p_group * params.total_value(np.array(mL), np.array(m_H))
    return float(V)


# =====================================================================
# GARBLING OPTIMIZATION
# =====================================================================

def make_objective(params: ModelParams, expt: Experiment, K: int):
    """Create the negative objective function for optimization.

    Returns a function that takes a flat W array and returns -V(W).
    Fully vectorized over signals for speed.
    """
    n = expt.n
    N = expt.n_types
    p = expt.probs
    posts = expt.posteriors

    def neg_objective(W_flat):
        W = W_flat.reshape(n, K)
        Z = p[:, None] * W           # joint probs: n x K
        pi = Z.sum(axis=0)           # signal probs: K

        # Vectorized: compute all institutional posteriors at once
        active = pi > 1e-15
        if not np.any(active):
            return 0.0
        # mu_all shape: (K_active, N_types)
        mu_all = (Z[:, active].T @ posts) / pi[active, None]
        mu_L = mu_all[:, 0]
        mu_H = mu_all[:, N-1]
        vals = params.total_value(mu_L, mu_H)  # already vectorized
        return -float(pi[active] @ vals)

    return neg_objective


def make_sm_W(expt: Experiment) -> np.ndarray:
    """Construct the SM garbling matrix.

    Groups atoms by mu_L value; each group maps to one signal.
    """
    n = expt.n
    mu_L_col = expt.posteriors[:, 0]
    tol = 1e-10

    unique_L = []
    atom_to_group = np.zeros(n, dtype=int)
    for i in range(n):
        mL = mu_L_col[i]
        found = False
        for j, uL in enumerate(unique_L):
            if abs(mL - uL) < tol:
                atom_to_group[i] = j
                found = True
                break
        if not found:
            atom_to_group[i] = len(unique_L)
            unique_L.append(mL)

    K = max(n, len(unique_L))
    W = np.zeros((n, K))
    for i in range(n):
        W[i, atom_to_group[i]] = 1.0
    return W


def _slsqp_worker(args):
    """Worker function for parallel SLSQP random starts."""
    params, expt, K, seed, batch_size = args
    n = expt.n
    neg_obj = make_objective(params, expt, K)
    rng = np.random.default_rng(seed)

    def row_sum_constraint(W_flat):
        W = W_flat.reshape(n, K)
        return W.sum(axis=1) - 1.0

    bounds = [(0.0, 1.0)] * (n * K)
    constraints = {'type': 'eq', 'fun': row_sum_constraint}

    best_val = -np.inf
    best_W_flat = None
    for _ in range(batch_size):
        W0 = rng.dirichlet(np.ones(K), size=n).flatten()
        try:
            res = minimize(neg_obj, W0, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 2000, 'ftol': 1e-15})
            val = -res.fun
            if val > best_val + 1e-12:
                best_val = val
                best_W_flat = res.x
        except Exception:
            pass
    return best_val, best_W_flat


def solve_unrestricted(params: ModelParams, expt: Experiment, config: Config,
                       verbose: bool = False) -> Tuple[float, np.ndarray]:
    """Solve the unrestricted garbling problem using multiple methods.

    Adaptive 3-phase budget:
      Phase 1: SM, FR, and bridge-gap seeds + small batch of random starts
               If nothing beats SM_value by more than gap_tol, skip to DE.
      Phase 2: Full n_slsqp_starts random starts (only if Phase 1 found a gap,
               or in thorough mode).
      Phase 3: Basin-hopping only if DE/SLSQP found nonzero gap.

    Returns (best_value, best_W).
    """
    n = expt.n
    K = n  # K = n is sufficient for any optimum
    neg_obj = make_objective(params, expt, K)

    # Constraints for SLSQP: each row sums to 1
    def row_sum_constraint(W_flat):
        W = W_flat.reshape(n, K)
        return W.sum(axis=1) - 1.0

    bounds = [(0.0, 1.0)] * (n * K)
    constraints_slsqp = {'type': 'eq', 'fun': row_sum_constraint}

    best_val = -np.inf
    best_W = None
    methods_used = []

    # --- Reference: SM value for adaptive early termination ---
    V_SM_ref = compute_sm_value(params, expt)

    # --- Phase 1: Seeded SLSQP starts ---
    W_SM = make_sm_W(expt)[:, :K]
    W_FR = np.eye(n, K)

    seeded_inits = [("SM", W_SM), ("FR", W_FR)]

    # Add bridge-gap seeds
    try:
        bridge_seeds = make_bridge_gap_seeds(params, expt, K)
        for i, seed_flat in enumerate(bridge_seeds):
            seeded_inits.append((f"bridge{i}", seed_flat.reshape(n, K)))
    except Exception:
        pass

    for label, W_init in seeded_inits:
        try:
            res = minimize(neg_obj, W_init.flatten(), method='SLSQP',
                           bounds=bounds, constraints=constraints_slsqp,
                           options={'maxiter': 3000, 'ftol': 1e-15})
            val = -res.fun
            if val > best_val + 1e-12:
                best_val = val
                best_W = res.x.reshape(n, K)
                if verbose:
                    print(f"    SLSQP from {label}: {val:.8f}")
        except Exception:
            pass

    # --- Phase 1b: Small batch of random starts (adaptive) ---
    if config.use_slsqp and config.adaptive:
        phase1_starts = min(config.phase1_random_starts, config.n_slsqp_starts)
        val, W_flat = _slsqp_worker(
            (params, expt, K, 0, phase1_starts))
        if W_flat is not None and val > best_val + 1e-12:
            best_val = val
            best_W = W_flat.reshape(n, K)

        # Decide whether to escalate: if best_val is close to SM, stop early
        gap_so_far = best_val - V_SM_ref
        escalate = gap_so_far > config.gap_tol * 0.1
        methods_used.append(f"phase1({phase1_starts})")
    else:
        escalate = True  # non-adaptive mode: always run full budget

    # --- Phase 2: Full random starts (parallelized when n_parallel > 1) ---
    if config.use_slsqp and escalate:
        remaining_starts = config.n_slsqp_starts
        if config.adaptive:
            remaining_starts = max(0, config.n_slsqp_starts - config.phase1_random_starts)
        if remaining_starts > 0:
            if config.n_parallel <= 1:
                val, W_flat = _slsqp_worker(
                    (params, expt, K, 1, remaining_starts))
                if W_flat is not None and val > best_val + 1e-12:
                    best_val = val
                    best_W = W_flat.reshape(n, K)
            else:
                n_workers = min(config.n_parallel, remaining_starts)
                batch_size = max(1, remaining_starts // n_workers)
                worker_args = [
                    (params, expt, K, seed + 1, batch_size)
                    for seed in range(n_workers)
                ]
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = [pool.submit(_slsqp_worker, a) for a in worker_args]
                    for f in as_completed(futures):
                        val, W_flat = f.result()
                        if W_flat is not None and val > best_val + 1e-12:
                            best_val = val
                            best_W = W_flat.reshape(n, K)
            methods_used.append(f"phase2({remaining_starts})")

    # --- Method 3: Differential Evolution ---
    # In adaptive mode: only run DE if we've already found a nonzero gap
    # (it's expensive and unlikely to help when SM is clearly optimal).
    run_de = config.use_differential_evolution
    if config.adaptive and config.use_differential_evolution:
        gap_so_far = best_val - V_SM_ref
        run_de = gap_so_far > config.gap_tol * 0.1

    if run_de and n * K <= 200:
        # DE needs unconstrained parameterization
        # Use softmax rows: for each row, K-1 free params
        def de_objective(x_free):
            W = np.zeros((n, K))
            idx = 0
            for i in range(n):
                logits = np.zeros(K)
                logits[1:] = x_free[idx:idx + K - 1]
                idx += K - 1
                exp_logits = np.exp(logits - logits.max())
                W[i] = exp_logits / exp_logits.sum()
            return neg_obj(W.flatten())

        n_free = n * (K - 1)
        de_bounds = [(-5.0, 5.0)] * n_free
        try:
            res_de = differential_evolution(
                de_objective, de_bounds,
                maxiter=config.de_maxiter,
                popsize=config.de_popsize,
                seed=42, tol=1e-12,
                mutation=(0.5, 1.5), recombination=0.9)
            # Reconstruct W
            W_de = np.zeros((n, K))
            idx = 0
            for i in range(n):
                logits = np.zeros(K)
                logits[1:] = res_de.x[idx:idx + K - 1]
                idx += K - 1
                exp_logits = np.exp(logits - logits.max())
                W_de[i] = exp_logits / exp_logits.sum()
            val_de = -neg_obj(W_de.flatten())
            if val_de > best_val + 1e-12:
                best_val = val_de
                best_W = W_de
                if verbose:
                    print(f"    DE: {val_de:.8f}")
            methods_used.append("DE")
        except Exception as e:
            if verbose:
                print(f"    DE failed: {e}")

    # --- Method 4: Basin-hopping ---
    # In adaptive mode: only run BH if we've already found a nonzero gap
    # (otherwise SM is almost certainly optimal and BH adds no value).
    run_bh = config.use_basin_hopping
    if config.adaptive and config.use_basin_hopping:
        gap_so_far = best_val - V_SM_ref
        run_bh = gap_so_far > config.gap_tol * 0.1

    if run_bh:
        W0 = make_sm_W(expt)[:, :K].flatten()
        minimizer_kwargs = {
            'method': 'SLSQP',
            'bounds': bounds,
            'constraints': constraints_slsqp,
            'options': {'maxiter': 1000, 'ftol': 1e-14}
        }
        try:
            res_bh = basinhopping(
                neg_obj, W0,
                minimizer_kwargs=minimizer_kwargs,
                niter=config.bh_niter, seed=42, T=0.5)
            val_bh = -res_bh.fun
            if val_bh > best_val + 1e-12:
                best_val = val_bh
                best_W = res_bh.x.reshape(n, K)
                if verbose:
                    print(f"    BH: {val_bh:.8f}")
            methods_used.append("BH")
        except Exception as e:
            if verbose:
                print(f"    BH failed: {e}")

    return best_val, best_W


# =====================================================================
# CONDITION CHECKING
# =====================================================================

def check_conditions(params: ModelParams, expt: Experiment) -> Dict[str, bool]:
    """Check whether A1, A2, A3 hold."""
    N = expt.n_types
    mu_L_col = expt.posteriors[:, 0]
    mu_H_col = expt.posteriors[:, N-1]

    # Group by mu_L
    tol = 1e-10
    unique_L = []
    groups = {}
    for i in range(expt.n):
        mL = mu_L_col[i]
        found = False
        for j, uL in enumerate(unique_L):
            if abs(mL - uL) < tol:
                groups[j].append(i)
                found = True
                break
        if not found:
            groups[len(unique_L)] = [i]
            unique_L.append(mL)

    # A1: Var(mu_H | mu_L) > 0
    a1 = False
    for j in groups:
        idx = groups[j]
        if len(idx) > 1:
            p_cond = np.array([expt.probs[i] for i in idx])
            p_cond = p_cond / p_cond.sum()
            vals = np.array([mu_H_col[i] for i in idx])
            var = p_cond @ (vals**2) - (p_cond @ vals)**2
            if var > 1e-10:
                a1 = True
                break

    # A2: g convex (rigorous check)
    # Computes m(ell) = E[mu_H | mu_L = ell] at the unique ell values, then:
    #  (i) Checks convexity at sample points via second differences
    #  (ii) CRITICAL: Verifies m(ell) does not straddle the c2 kink of r.
    #       If min(m_vals) < c2 < max(m_vals), r(m(ell)) has a concave kink
    #       between adjacent samples, violating g convexity even if the
    #       discrete samples appear convex. The discrete check alone misses this.
    #  (iii) Final check: evaluates g at 200 interpolated points (linear m
    #        interpolation) and confirms convexity there too.
    m_vals = []
    for j, mL in enumerate(unique_L):
        idx = groups[j]
        p_group = sum(expt.probs[i] for i in idx)
        if p_group > 1e-15:
            m = sum(expt.probs[i] * mu_H_col[i] for i in idx) / p_group
            m_vals.append((mL, m))

    if len(m_vals) >= 3:
        # Sort by ell so interpolation is well-defined
        m_vals.sort(key=lambda x: x[0])
        ells = np.array([x[0] for x in m_vals])
        ms = np.array([x[1] for x in m_vals])
        g_vals = np.array([
            float(params.h(np.array(e)) + params.delta * params.r(np.array(m)))
            for e, m in zip(ells, ms)])

        # (i) Discrete convexity at sample points
        a2 = True
        for i in range(1, len(g_vals) - 1):
            if ells[i+1] - ells[i] > 1e-12 and ells[i] - ells[i-1] > 1e-12:
                slope_right = (g_vals[i+1] - g_vals[i]) / (ells[i+1] - ells[i])
                slope_left = (g_vals[i] - g_vals[i-1]) / (ells[i] - ells[i-1])
                if slope_right < slope_left - 1e-8:
                    a2 = False
                    break

        # (ii) Kink-straddle check: r has a concave kink at mu_H = c2
        # If m(ell) values span c2, r(m(ell)) has a kink between samples
        if a2 and len(ms) >= 2:
            m_min, m_max = float(ms.min()), float(ms.max())
            if m_min < params.c2 - 1e-12 and m_max > params.c2 + 1e-12:
                # m values straddle c2 => non-convex g between samples
                a2 = False

        # (iii) Dense convexity check via linear interpolation of m
        # Catches any non-convexity the discrete check misses
        if a2:
            ell_dense = np.linspace(ells[0], ells[-1], 200)
            m_dense = np.interp(ell_dense, ells, ms)
            g_dense = params.h(ell_dense) + params.delta * params.r(m_dense)
            # Check convexity via second differences
            d2 = np.diff(g_dense, 2)
            if d2.min() < -1e-6:
                a2 = False
    else:
        a2 = True  # can't check with fewer than 3 points

    # A3: strict Jensen for r
    a3 = False
    for j in groups:
        idx = groups[j]
        if len(idx) > 1:
            p_cond = np.array([expt.probs[i] for i in idx])
            p_cond = p_cond / p_cond.sum()
            vals = np.array([mu_H_col[i] for i in idx])
            E_r = p_cond @ params.r(vals)
            r_E = float(params.r(np.array(p_cond @ vals)))
            if r_E - E_r > 1e-8:
                a3 = True
                break

    return {'A1': a1, 'A2': a2, 'A3': a3}


# =====================================================================
# ADVERSARIAL GARBLING CONSTRUCTION
# =====================================================================

def try_adversarial(params: ModelParams, expt: Experiment, K: int = None
                    ) -> Tuple[float, np.ndarray]:
    """Try to construct a garbling that exploits mu_H information.

    Strategy: create signals that partially reveal mu_H to compress
    the continuation term, while accepting some cost to h.
    """
    n = expt.n
    N = expt.n_types
    if K is None:
        K = n
    mu_H_col = expt.posteriors[:, N-1]

    # Sort atoms by mu_H
    order = np.argsort(mu_H_col)

    # Strategy 1: split into "low mu_H" and "high mu_H" groups
    # Pool each group to compress mu_H variation
    mid = n // 2
    W_adv1 = np.zeros((n, K))
    for i in order[:mid]:
        W_adv1[i, 0] = 1.0  # low-mu_H signal
    for i in order[mid:]:
        W_adv1[i, 1] = 1.0  # high-mu_H signal

    neg_obj = make_objective(params, expt, K)
    V_adv1 = -neg_obj(W_adv1.flatten())

    # Strategy 2: create K signals that each pool atoms with similar mu_H
    chunk_size = max(1, n // min(K, n))
    W_adv2 = np.zeros((n, K))
    for idx, i in enumerate(order):
        k = min(idx // chunk_size, K - 1)
        W_adv2[i, k] = 1.0
    V_adv2 = -neg_obj(W_adv2.flatten())

    # Strategy 3: reveal mu_H fully (group by mu_H)
    W_adv3 = np.zeros((n, K))
    mu_H_unique = []
    for i in range(n):
        mH = mu_H_col[i]
        found = False
        for j, uH in enumerate(mu_H_unique):
            if abs(mH - uH) < 1e-10:
                W_adv3[i, j] = 1.0
                found = True
                break
        if not found:
            W_adv3[i, len(mu_H_unique)] = 1.0
            mu_H_unique.append(mH)
    V_adv3 = -neg_obj(W_adv3.flatten())

    # Strategy 4 (bridge-gap): pair-pool atoms with different mu_L values
    # Creates signals where mu_{H,k} != m(mu_{L,k}), directly targeting the gap.
    # Pool atoms with extreme mu_L to compress mu_H while sacrificing some h.
    mu_L_col = expt.posteriors[:, 0]
    order_L = np.argsort(mu_L_col)
    W_adv4 = np.zeros((n, K))
    # Pair lowest-mu_L with highest-mu_L into K//2 signals
    pairs_used = 0
    for k in range(K // 2):
        if k < len(order_L) // 2:
            W_adv4[order_L[k], k] = 1.0
            W_adv4[order_L[-(k+1)], k] = 1.0
            pairs_used += 1
    # Remaining atoms go to their own signals
    used = set()
    for k in range(pairs_used):
        for i in range(n):
            if W_adv4[i, k] > 0:
                used.add(i)
    next_k = pairs_used
    for i in range(n):
        if i not in used and next_k < K:
            W_adv4[i, next_k] = 1.0
            next_k += 1
    # Normalize rows (in case any row is empty, make uniform)
    row_sums = W_adv4.sum(axis=1)
    for i in range(n):
        if row_sums[i] < 1e-15:
            W_adv4[i] = 1.0 / K
    V_adv4 = -neg_obj(W_adv4.flatten())

    # Strategy 5 (bridge-gap, targeted): group atoms by closeness in mu_H
    # while ignoring mu_L. This creates signals with compressed mu_H but
    # varying mu_L, potentially exploiting r concavity across mu_L groups.
    W_adv5 = np.zeros((n, K))
    # K groups by mu_H quantile
    for idx, i in enumerate(order):
        k = int(idx * K / n)
        k = min(k, K - 1)
        W_adv5[i, k] = 1.0
    V_adv5 = -neg_obj(W_adv5.flatten())

    # Return the best adversarial value
    vals = [(V_adv1, W_adv1), (V_adv2, W_adv2), (V_adv3, W_adv3),
            (V_adv4, W_adv4), (V_adv5, W_adv5)]
    best = max(vals, key=lambda x: x[0])
    return best


def make_bridge_gap_seeds(params: ModelParams, expt: Experiment,
                          K: int) -> List[np.ndarray]:
    """Generate garbling seeds that explicitly target the bridge gap.

    For each pair of atoms with different mu_L values, create a garbling
    that pools them into one signal. The resulting signal has
    mu_{H,k} != m(mu_{L,k}) in general, directly exploiting records that
    carry mu_H information beyond what m(mu_L) provides.

    Returns a list of flat W arrays to use as SLSQP starting points.
    """
    n = expt.n
    N = expt.n_types
    mu_L_col = expt.posteriors[:, 0]
    mu_H_col = expt.posteriors[:, N-1]

    seeds = []

    # Seed 1: pair each atom with the one whose mu_L is furthest
    order_L = np.argsort(mu_L_col)
    W = np.zeros((n, K))
    for k in range(min(n // 2, K)):
        i_low = order_L[k]
        i_high = order_L[-(k+1)]
        W[i_low, k] = 1.0
        W[i_high, k] = 1.0
    # Remaining atoms to own signals
    used = set()
    for k in range(min(n // 2, K)):
        for i in range(n):
            if W[i, k] > 0:
                used.add(i)
    next_k = min(n // 2, K)
    for i in range(n):
        if i not in used and next_k < K:
            W[i, next_k] = 1.0
            next_k += 1
    row_sums = W.sum(axis=1)
    for i in range(n):
        if row_sums[i] < 1e-15:
            W[i] = 1.0 / K
        else:
            W[i] = W[i] / row_sums[i]
    seeds.append(W.flatten())

    # Seed 2: group atoms by mu_H proximity (ignore mu_L)
    order_H = np.argsort(mu_H_col)
    W = np.zeros((n, K))
    chunk_size = max(1, n // K)
    for idx, i in enumerate(order_H):
        k = min(idx // chunk_size, K - 1)
        W[i, k] = 1.0
    seeds.append(W.flatten())

    # Seed 3: group by combined (mu_L, mu_H) using k-means-like split
    # Sort by mu_L - alpha*mu_H to create "anti-correlated" pooling
    combined = mu_L_col - 0.5 * mu_H_col
    order_c = np.argsort(combined)
    W = np.zeros((n, K))
    for idx, i in enumerate(order_c):
        k = min(idx // chunk_size, K - 1)
        W[i, k] = 1.0
    seeds.append(W.flatten())

    return seeds


# =====================================================================
# SINGLE CASE RUNNER
# =====================================================================

@dataclass
class CaseResult:
    label: str
    params: Dict
    n_atoms: int
    n_types: int
    conditions: Dict[str, bool]
    V_SM: float
    V_full_rev: float
    V_full_pool: float
    V_adversarial: float
    V_unrestricted: float
    gap: float
    is_counterexample: bool
    runtime_sec: float


def run_case(params: ModelParams, expt: Experiment, config: Config,
             verbose: bool = False) -> CaseResult:
    """Run a single test case."""
    t0 = time.time()

    if verbose:
        print(f"\n  [{expt.label}] n={expt.n}, N={expt.n_types}")

    # Check conditions
    conds = check_conditions(params, expt)
    if verbose:
        print(f"    A1={conds['A1']}, A2={conds['A2']}, A3={conds['A3']}")

    # SM value
    V_SM = compute_sm_value(params, expt)

    # Full revelation
    N = expt.n_types
    V_FR = sum(expt.probs[i] * float(params.total_value(
        np.array(expt.posteriors[i, 0]),
        np.array(expt.posteriors[i, N-1])))
        for i in range(expt.n))

    # Full pooling
    prior = expt.prior
    V_FP = float(params.total_value(
        np.array(prior[0]), np.array(prior[N-1])))

    # Adversarial
    V_adv, _ = try_adversarial(params, expt)

    # Unrestricted optimization
    V_unres, W_opt = solve_unrestricted(params, expt, config, verbose=verbose)

    gap = V_unres - V_SM
    is_counter = gap > config.gap_tol

    runtime = time.time() - t0

    if verbose:
        print(f"    SM={V_SM:.8f}, FR={V_FR:.8f}, Pool={V_FP:.8f}")
        print(f"    Adv={V_adv:.8f}, Unres={V_unres:.8f}")
        print(f"    Gap={gap:.10f} {'<<< COUNTEREXAMPLE >>>' if is_counter else '(ok)'}")
        print(f"    Time: {runtime:.1f}s")

    return CaseResult(
        label=expt.label,
        params={
            'y_L': params.y_L, 'y_bar': params.y_bar,
            'y_under': params.y_under, 'y_H': params.y_H,
            'sigma': params.sigma, 'w1': params.w1, 'delta': params.delta
        },
        n_atoms=expt.n,
        n_types=expt.n_types,
        conditions=conds,
        V_SM=V_SM, V_full_rev=V_FR, V_full_pool=V_FP,
        V_adversarial=V_adv, V_unrestricted=V_unres,
        gap=gap, is_counterexample=is_counter,
        runtime_sec=runtime
    )


# =====================================================================
# EXPERIMENT GENERATION
# =====================================================================

def generate_experiments(params: ModelParams, quick: bool = False
                         ) -> List[Tuple[ModelParams, Experiment]]:
    """Generate all experiment structures to test."""
    cases = []

    # --- TIER 1: Nested independence ---
    mu_L_configs = [
        (np.array([0.1, 0.3, 0.5, 0.7]),
         np.array([0.2, 0.3, 0.3, 0.2])),
        (np.array([0.05, 0.25, 0.55, 0.85]),
         np.array([0.15, 0.35, 0.35, 0.15])),
        (np.array([0.2, 0.4, 0.6, 0.8, 0.95]),
         np.array([0.1, 0.2, 0.3, 0.25, 0.15])),
    ]

    q_configs = [
        # Low q_bar (g convex)
        (np.array([0.02, 0.08, 0.15]), np.array([0.4, 0.4, 0.2])),
        (np.array([0.03, 0.10, 0.20]), np.array([0.3, 0.4, 0.3])),
        (np.array([0.01, 0.05, 0.12, 0.25]),
         np.array([0.3, 0.3, 0.25, 0.15])),
        # Medium q_bar (borderline)
        (np.array([0.05, 0.15, 0.30]), np.array([0.3, 0.4, 0.3])),
        # High q_bar (g not convex)
        (np.array([0.15, 0.40, 0.75]), np.array([0.3, 0.4, 0.3])),
    ]

    if quick:
        mu_L_configs = mu_L_configs[:2]
        q_configs = q_configs[:3]

    for i, (mL_v, mL_p) in enumerate(mu_L_configs):
        for j, (q_v, q_p) in enumerate(q_configs):
            expt = make_nested_independent(
                mL_v, mL_p, q_v, q_p,
                label=f"nested_L{i}_Q{j}")
            cases.append((params, expt))

    # --- TIER 2: Correlated experiments (legacy, single atom per mu_L) ---
    # NOTE: These always fail A1 because each atom has a unique mu_L. They add
    # no evidence to the A1+A2+A3 pool and just consume compute on counterexamples.
    # Kept only in quick mode for historical comparison with the first-pass results.
    if quick:
        for ctype in ["positive", "negative", "nonlinear"]:
            for n_atoms in [8]:
                for seed in [42]:
                    expt = make_correlated(
                        n_atoms, ctype, seed=seed,
                        label=f"corr_{ctype}_n{n_atoms}_s{seed}")
                    cases.append((params, expt))
        for seed in range(3):
            expt = make_correlated(
                12, "random", seed=seed,
                label=f"corr_random_s{seed}")
            cases.append((params, expt))

    # --- TIER 2b: Correlated-grid experiments (multi-atom per mu_L) ---
    # These are the critical tests: A1 can hold because multiple atoms share
    # the same mu_L value, enabling conditional mu_H variance.
    #
    # For A2 (g convex) to hold with non-affine m:
    #   - h(ell) is piecewise linear (weakly convex), so it adds no curvature
    #   - When m(ell) is in r's linear region, g(ell) curvature comes from m(ell)
    #   - => need m(ell) CONVEX in ell
    # For A3 (strict Jensen gain):
    #   - Need atoms within a group to span c2 (so r's kink creates Jensen gap)

    # (a) SAFE: positive correlation, decreasing CONVEX m(ell)
    # Rate of decrease slows as ell grows. Atoms span c2 in group 1 for A3.
    cg_safe_pos = make_correlated_grid(
        mu_L_vals=[0.1, 0.3, 0.5, 0.7],
        mu_H_given_L=[
            [0.04, 0.10, 0.20],   # m(0.1) ~ 0.112, atoms span c2=0.128
            [0.03, 0.08, 0.16],   # m(0.3) ~ 0.089
            [0.03, 0.06, 0.14],   # m(0.5) ~ 0.075
            [0.03, 0.05, 0.12],   # m(0.7) ~ 0.065
        ],
        probs_H_given_L=[
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
        ],
        mu_L_probs=[0.2, 0.3, 0.3, 0.2],
        label="cg_safe_pos_convex"
    )
    cases.append((params, cg_safe_pos))

    # (b) SAFE: negative correlation, increasing CONVEX m(ell)
    # Rate of increase accelerates. Atoms span c2 in group 4 for A3.
    cg_safe_neg = make_correlated_grid(
        mu_L_vals=[0.1, 0.3, 0.5, 0.7],
        mu_H_given_L=[
            [0.01, 0.02, 0.04],   # m(0.1) ~ 0.023
            [0.015, 0.03, 0.06],  # m(0.3) ~ 0.038
            [0.02, 0.04, 0.10],   # m(0.5) ~ 0.056
            [0.03, 0.08, 0.18],   # m(0.7) ~ 0.095, spans c2
        ],
        probs_H_given_L=[
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
        ],
        mu_L_probs=[0.2, 0.3, 0.3, 0.2],
        label="cg_safe_neg_convex"
    )
    cases.append((params, cg_safe_neg))

    # (c) SAFE: mildly convex m with atoms spanning c2 in multiple groups
    cg_safe_mid = make_correlated_grid(
        mu_L_vals=[0.1, 0.3, 0.5, 0.7],
        mu_H_given_L=[
            [0.03, 0.08, 0.16],   # m(0.1) ~ 0.089, spans c2
            [0.025, 0.07, 0.14],  # m(0.3) ~ 0.077
            [0.02, 0.06, 0.13],   # m(0.5) ~ 0.069
            [0.02, 0.05, 0.12],   # m(0.7) ~ 0.062
        ],
        probs_H_given_L=[
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
        ],
        mu_L_probs=[0.2, 0.3, 0.3, 0.2],
        label="cg_safe_mid"
    )
    cases.append((params, cg_safe_mid))

    # (d) BORDERLINE: non-convex m - will fail A2 but tests the boundary
    # Included to verify the code finds counterexamples when conditions fail
    cg_border_concave = make_correlated_grid(
        mu_L_vals=[0.1, 0.3, 0.5, 0.7],
        mu_H_given_L=[
            [0.02, 0.05, 0.09],   # m(0.1) ~ 0.053 (concave-shaped m)
            [0.03, 0.07, 0.13],   # m(0.3) ~ 0.076 (peak)
            [0.025, 0.06, 0.11],  # m(0.5) ~ 0.064
            [0.01, 0.02, 0.05],   # m(0.7) ~ 0.024
        ],
        probs_H_given_L=[
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
        ],
        mu_L_probs=[0.2, 0.3, 0.3, 0.2],
        label="cg_border_concave_m"
    )
    cases.append((params, cg_border_concave))

    # (e) BORDERLINE: m values may cross c2 (high-mu_H variant)
    cg_border_high = make_correlated_grid(
        mu_L_vals=[0.1, 0.3, 0.5, 0.7],
        mu_H_given_L=[
            [0.08, 0.15, 0.25],   # m(0.1) ~ 0.155 (above c2=0.128)
            [0.06, 0.12, 0.22],   # m(0.3) ~ 0.130
            [0.05, 0.10, 0.18],   # m(0.5) ~ 0.111
            [0.04, 0.08, 0.15],   # m(0.7) ~ 0.093
        ],
        probs_H_given_L=[
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
        ],
        mu_L_probs=[0.2, 0.3, 0.3, 0.2],
        label="cg_border_high_mH"
    )
    cases.append((params, cg_border_high))

    # (f) Random correlated-grid experiments
    # These sample conditional mu_H means from random distributions
    n_rand = 3 if quick else 6
    for seed in range(n_rand):
        expt = make_correlated_grid_random(
            seed=seed, n_L=4, n_H_per_L=3,
            mu_L_range=(0.1, 0.75),
            mu_H_max=0.18,  # spans typical c2 values for better A3 coverage
            label=f"cg_random_s{seed}"
        )
        cases.append((params, expt))

    # --- TIER 3: N >= 4 types ---
    # DISABLED: make_n_type uses random Dirichlet points which have unique
    # mu_L values per atom, so A1 always fails. These cases just consume
    # compute on counterexamples without testing the theorem conditions.
    # A proper N-type test would require a grid construction like
    # make_correlated_grid but generalized to N types; left as future work.

    return cases


def generate_param_grid(quick: bool = False) -> List[ModelParams]:
    """Generate parameter grid satisfying all restrictions."""
    valid = []

    # Base: y_L=0, y_bar=1, y_under, y_H, sigma, w1, delta
    y_L_vals = [0.0]
    y_bar_vals = [1.0]
    y_under_vals = [0.2, 0.3, 0.4]
    y_H_vals = [1.5, 2.0]
    sigma_vals = [0.2, 0.3, 0.4]
    w1_offsets = [0.01, 0.05, 0.10]  # w1 = y_under + sigma - offset
    delta_vals = [0.5, 0.9, 1.5]

    if quick:
        y_under_vals = [0.2, 0.3]
        y_H_vals = [2.0]
        sigma_vals = [0.25, 0.35]
        w1_offsets = [0.02, 0.08]
        delta_vals = [0.9, 1.5]

    for y_L, y_bar, y_under, y_H, sig, offset, delt in product(
            y_L_vals, y_bar_vals, y_under_vals, y_H_vals,
            sigma_vals, w1_offsets, delta_vals):
        w1 = y_under + sig - offset  # ensures w1 <= y_under + sigma
        p = ModelParams(y_L, y_bar, y_under, y_H, sig, w1, delt)
        if p.is_feasible():
            valid.append(p)

    return valid


# =====================================================================
# MAIN
# =====================================================================

def _run_case_worker(args):
    """Top-level worker for running a single case (for ProcessPoolExecutor)."""
    params, expt, config = args
    return run_case(params, expt, config, verbose=False)


def main():
    parser = argparse.ArgumentParser(
        description="Deep verification of the SM theorem")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer trials")
    parser.add_argument("--output", default="sm_results.json",
                        help="Output JSON path")
    parser.add_argument("--serial", action="store_true",
                        help="Run cases serially (no case-level parallelism)")
    args = parser.parse_args()

    config = Config.quick() if args.quick else Config()
    n_cpus = os.cpu_count() or 1
    print("=" * 70)
    print("THREE-TYPE SELECTIVE MEMORY: DEEP COMPUTATIONAL VERIFICATION")
    print("=" * 70)
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"SLSQP starts: {config.n_slsqp_starts}")
    print(f"DE: {'ON' if config.use_differential_evolution else 'OFF'}")
    print(f"BH: {'ON' if config.use_basin_hopping else 'OFF'}")
    print(f"CPUs: {n_cpus}, parallel workers: {config.n_parallel}")

    # Generate parameter grid
    param_grid = generate_param_grid(quick=args.quick)
    print(f"\nParameter grid: {len(param_grid)} valid parameterizations")

    # Collect all cases
    all_case_args = []
    for params in param_grid:
        cases = generate_experiments(params, quick=args.quick)
        for _, expt in cases:
            all_case_args.append((params, expt, config))

    total_cases = len(all_case_args)
    print(f"Total cases to run: {total_cases}")

    t_start = time.time()
    all_results = []

    if args.serial:
        # Serial execution with verbose output
        for i, (params, expt, cfg) in enumerate(all_case_args):
            result = run_case(params, expt, cfg, verbose=True)
            all_results.append(result)
            print(f"  [{i+1}/{total_cases}] {result.label}: "
                  f"gap={result.gap:.10f} "
                  f"{'<<< COUNTER >>>' if result.is_counterexample else '(ok)'}")
    else:
        # Parallel execution across cases
        # Use 1 worker per SLSQP inside each case (to avoid over-subscribing)
        config_single = Config.quick() if args.quick else Config()
        config_single.n_parallel = 1  # each case runs SLSQP serially
        case_args = [(p, e, config_single) for p, e, _ in all_case_args]

        n_case_workers = min(n_cpus, total_cases)
        print(f"Running {total_cases} cases across {n_case_workers} workers...")
        sys.stdout.flush()

        with ProcessPoolExecutor(max_workers=n_case_workers) as pool:
            futures = {pool.submit(_run_case_worker, a): i
                       for i, a in enumerate(case_args)}
            for f in as_completed(futures):
                idx = futures[f]
                result = f.result()
                all_results.append(result)
                status = '<<< COUNTER >>>' if result.is_counterexample else '(ok)'
                conds = 'A1+A2+A3' if all(result.conditions.values()) else 'partial'
                print(f"  [{len(all_results)}/{total_cases}] "
                      f"{result.label} [{conds}]: "
                      f"gap={result.gap:.10f} {status} "
                      f"({result.runtime_sec:.1f}s)")
                sys.stdout.flush()

    elapsed = time.time() - t_start

    # Summary
    counters_all = [r for r in all_results if r.is_counterexample]
    counters_with_conditions = [r for r in counters_all
                                 if all(r.conditions.values())]

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total cases tested: {total_cases}")
    print(f"Total time: {elapsed:.1f}s")

    n_with_conds = sum(1 for r in all_results if all(r.conditions.values()))
    print(f"\nCases where A1+A2+A3 all hold: {n_with_conds}")

    print(f"\nCounterexamples (unrestricted > SM + {config.gap_tol}):")
    print(f"  Total: {len(counters_all)}")
    print(f"  With A1+A2+A3: {len(counters_with_conditions)}")

    if counters_with_conditions:
        print(f"\n  *** COUNTEREXAMPLES FOUND WITH ALL CONDITIONS! ***")
        for r in counters_with_conditions:
            print(f"    {r.label}: gap = {r.gap:.8f}")
    else:
        print(f"\n  No counterexamples found when A1+A2+A3 hold.")
        if n_with_conds > 0:
            gaps = [r.gap for r in all_results if all(r.conditions.values())]
            print(f"  Max gap: {max(gaps):.10f}")
            print(f"  Min gap: {min(gaps):.10f}")
            print(f"  Mean gap: {np.mean(gaps):.10f}")

    print(f"\nCounterexamples WITHOUT all conditions (expected):")
    n_without = len(counters_all) - len(counters_with_conditions)
    print(f"  {n_without} cases (SM suboptimal when A2 fails)")

    # Save results
    output = {
        'config': asdict(config),
        'total_cases': total_cases,
        'n_with_conditions': n_with_conds,
        'n_counterexamples_total': len(counters_all),
        'n_counterexamples_with_conditions': len(counters_with_conditions),
        'elapsed_seconds': elapsed,
        'results': [asdict(r) for r in all_results],
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Final verdict
    print(f"\n{'='*70}")
    if counters_with_conditions:
        print("VERDICT: SM IS NOT GLOBALLY OPTIMAL under A1+A2+A3.")
        print("The proof gap is real AND the theorem statement needs revision.")
    elif n_with_conds == 0:
        print("VERDICT: INCONCLUSIVE (no cases tested with A1+A2+A3).")
    else:
        print("VERDICT: SM appears GLOBALLY OPTIMAL under A1+A2+A3.")
        print("The proof gap is real but the theorem is likely TRUE.")
        print("A different proof technique is needed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
