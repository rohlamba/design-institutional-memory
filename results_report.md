# Selective Memory Theorem: Computational Verification Report

## 1. Executive Summary

**Date**: April 10, 2026
**Mode**: Quick (200 SLSQP starts, DE on, basin-hopping off)
**Runtime**: 312.5 seconds on 16 CPUs

| Metric | Value |
|--------|-------|
| Total cases tested | 192 |
| Parameterizations | 16 |
| Cases with A1+A2+A3 | 66 |
| **Counterexamples with A1+A2+A3** | **0** |
| Counterexamples without all conditions | 96 |

**Verdict**: SM appears **globally optimal** under the theorem's conditions (A1+A2+A3). The proof gap is real as a logical gap, but the theorem statement is likely correct. A different proof technique is needed.

---

## 2. Code Modifications

The original `selective_memory_verify.py` was optimized for performance. Three changes were made:

### a) Vectorized objective function

The objective function `make_objective` evaluates V(W) for a garbling matrix W. The original code looped over K signals in Python:

```python
# ORIGINAL (slow)
total = 0.0
for k in range(K):
    if pi[k] < 1e-15:
        continue
    mu_k = Z[:, k] @ posts / pi[k]
    mu_L_k = mu_k[0]
    mu_H_k = mu_k[N-1]
    total += pi[k] * float(params.total_value(np.array(mu_L_k), np.array(mu_H_k)))
```

The new version computes all institutional posteriors in a single matrix multiply:

```python
# OPTIMIZED (vectorized)
active = pi > 1e-15
mu_all = (Z[:, active].T @ posts) / pi[active, None]  # all posteriors at once
mu_L = mu_all[:, 0]
mu_H = mu_all[:, N-1]
vals = params.total_value(mu_L, mu_H)  # vectorized call
return -float(pi[active] @ vals)
```

**Impact**: ~3x speedup per objective evaluation. This compounds across thousands of optimizer iterations.

### b) Case-level parallelism

All 192 test cases are independent. The main loop was replaced with a `ProcessPoolExecutor` distributing cases across 16 CPU workers:

```python
with ProcessPoolExecutor(max_workers=n_case_workers) as pool:
    futures = {pool.submit(_run_case_worker, a): i for i, a in enumerate(case_args)}
    for f in as_completed(futures):
        result = f.result()
        ...
```

Each worker runs SLSQP starts serially (no nested subprocess pools) to avoid oversubscription.

### c) BLAS thread limiting

```python
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
```

Prevents numpy/scipy from spawning internal threads that would compete with the 16 case-level workers.

### Overall speedup

**~12x faster**: 312 seconds vs estimated 60+ minutes serial.

---

## 3. Parameter Grid

16 valid parameterizations were tested, all with y_L = 0.0, y_bar = 1.0, y_H = 2.0.

The wage w1 is constructed as w1 = y_under + sigma - offset to ensure the concavity constraint w1 <= y_under + sigma.

| y_under | sigma | w1    | delta | c1_h   | c1_r    | c2     |
|---------|-------|-------|-------|--------|---------|--------|
| 0.2     | 0.25  | 0.370 | 0.9   | 0.6300 | -0.0167 | 0.0944 |
| 0.2     | 0.25  | 0.370 | 1.5   | 0.6300 | -0.0167 | 0.0944 |
| 0.2     | 0.25  | 0.430 | 0.9   | 0.8200 | -0.0111 | 0.1278 |
| 0.2     | 0.25  | 0.430 | 1.5   | 0.8200 | -0.0111 | 0.1278 |
| 0.2     | 0.35  | 0.470 | 0.9   | 0.8800 | -0.0444 | 0.1500 |
| 0.2     | 0.35  | 0.470 | 1.5   | 0.8800 | -0.0444 | 0.1500 |
| 0.2     | 0.35  | 0.530 | 0.9   | 0.8200 | -0.0111 | 0.1833 |
| 0.2     | 0.35  | 0.530 | 1.5   | 0.8200 | -0.0111 | 0.1833 |
| 0.3     | 0.25  | 0.470 | 0.9   | 0.7800 | -0.0471 | 0.1000 |
| 0.3     | 0.25  | 0.470 | 1.5   | 0.7800 | -0.0471 | 0.1000 |
| 0.3     | 0.25  | 0.530 | 0.9   | 0.7200 | -0.0118 | 0.1353 |
| 0.3     | 0.25  | 0.530 | 1.5   | 0.7200 | -0.0118 | 0.1353 |
| 0.3     | 0.35  | 0.570 | 0.9   | 0.7800 | -0.0471 | 0.1588 |
| 0.3     | 0.35  | 0.570 | 1.5   | 0.7800 | -0.0471 | 0.1588 |
| 0.3     | 0.35  | 0.630 | 0.9   | 0.7200 | -0.0118 | 0.1941 |
| 0.3     | 0.35  | 0.630 | 1.5   | 0.7200 | -0.0118 | 0.1941 |

All parameterizations satisfy the feasibility constraints:
- y_L + sigma < w1 (h has interior threshold, 0 < c1_h < 1)
- w1 <= y_under + sigma (r is concave, c1_r <= 0)
- y_under < w1 < y_H (ratchet threshold interior, 0 < c2 < 1)

---

## 4. Experiment Structures

Each parameterization was tested with 12 experiments (192 total):

### Tier 1: Nested-Independence (6 per param = 96 total)

Two mu_L configurations crossed with three q configurations:

**mu_L configs:**
- L0: {0.1, 0.3, 0.5, 0.7} with probs {0.2, 0.3, 0.3, 0.2}
- L1: {0.05, 0.25, 0.55, 0.85} with probs {0.15, 0.35, 0.35, 0.15}

**q configs (q = Pr(H|not-L)):**
- Q0: {0.02, 0.08, 0.15} with probs {0.4, 0.4, 0.2} (low q_bar)
- Q1: {0.03, 0.10, 0.20} with probs {0.3, 0.4, 0.3} (low-medium q_bar)
- Q2: {0.01, 0.05, 0.12, 0.25} with probs {0.3, 0.3, 0.25, 0.15} (low q_bar, 4 atoms)

Each nested experiment has n = J x Q atoms (12 or 16). Under this structure, m(ell) = E[mu_H | mu_L = ell] is affine in ell.

### Tier 2: Correlated Experiments (3 per param = 48 total)

Non-nested experiments where mu_L and mu_H can have arbitrary joint distribution (n=8 atoms each):

- **Positive correlation**: mu_L low => mu_H high (quality signal)
- **Negative correlation**: mu_L high => mu_H high (both extremes together)
- **Nonlinear**: U-shaped relationship between mu_L and mu_H

### Tier 3: Random Correlated (3 per param = 48 total)

Random Dirichlet-distributed points on the 3-simplex (n=12 atoms, seeds 0-2).

---

## 5. Optimization Methods

For each case, the script searches for the global maximum of V(W) over all feasible garblings:

1. **SLSQP from SM starting point**: The Selective Memory garbling as initial guess
2. **SLSQP from Full Revelation (FR)**: The identity garbling as initial guess
3. **200 SLSQP random starts**: Dirichlet-distributed random garbling matrices
4. **Differential Evolution**: Population-based global optimizer (maxiter=300, popsize=15)
5. **3 adversarial strategies**: Garblings that sort/split atoms by mu_H value

Basin-hopping was OFF in quick mode.

---

## 6. Results by Experiment Type

| Experiment Type       | Cases | Counterexamples | With A1+A2+A3 | Max Gap (with conds) |
|-----------------------|-------|-----------------|---------------|----------------------|
| Nested-independence   | 96    | 6               | 66            | 3.33e-16             |
| Correlated-positive   | 16    | 16              | 0             | n/a                  |
| Correlated-negative   | 16    | 16              | 0             | n/a                  |
| Correlated-nonlinear  | 16    | 16              | 0             | n/a                  |
| Correlated-random     | 48    | 42              | 0             | n/a                  |
| **Total**             | **192** | **96**        | **66**        | **~0 (machine eps)** |

### Counterexample gap statistics (when conditions fail)

| Type | Count | Min Gap | Max Gap | Mean Gap |
|------|-------|---------|---------|----------|
| Correlated-positive | 16 | 0.0043 | 0.0610 | 0.0221 |
| Correlated-negative | 16 | 0.0197 | 0.1236 | 0.0589 |
| Correlated-nonlinear | 16 | 0.0180 | 0.0919 | 0.0488 |
| Correlated-random | 42 | 0.0009 | 0.1690 | 0.0590 |
| Nested (A2 fails) | 6 | 0.0008 | 0.0037 | 0.0019 |

---

## 7. Condition Analysis

### Condition definitions

- **A1 (Nondegeneracy)**: Var(mu_H | mu_L) > 0 for a positive-probability set of mu_L values
- **A2 (No-blurring / g convex)**: The composite function g(ell) = h(ell) + delta * r(m(ell)) is convex
- **A3 (Strict Jensen gain)**: E[r(mu_H) | mu_L = ell] < r(m(ell)) for a positive-probability set

### Condition prevalence

| Condition | True | False |
|-----------|------|-------|
| A1 | 96 | 96 |
| A2 | 132 | 60 |
| A3 | 72 | 120 |
| All three | 66 | 126 |

### Why conditions fail in counterexamples

All 96 counterexamples have at least one condition failing:

| Missing Conditions | Count | Explanation |
|--------------------|-------|-------------|
| A1 + A2 + A3 | 54 | Correlated experiments: all atoms have unique mu_L, so conditional on mu_L there's no mu_H variation |
| A1 + A3 | 36 | Same issue with A1; A2 holds but A3 also fails |
| A2 only | 6 | Nested-independence with q_bar too high, violating g-convexity |

The 6 nested-independence counterexamples are instructive:
- All occur with q config Q1 ({0.03, 0.10, 0.20}), which has q_bar = 0.11
- The g-convexity condition (A2) requires c2 >= q_bar
- When c2 = 0.0944 < q_bar = 0.11, A2 fails and SM is no longer optimal
- Gaps are small (0.0008-0.0037) but consistently positive

---

## 8. Key Finding

**When all three conditions A1+A2+A3 hold (66 cases), the maximum gap between the unrestricted global optimum and the SM value is 3.33e-16 — indistinguishable from zero (machine epsilon).**

SM is the exact global optimum in every tested case where the theorem's conditions are satisfied.

---

## 9. Interpretation

1. **The proof gap is real but the theorem is likely correct.** The paper's proof optimizes over mu_L-based records in Step 2, but Step 1 can produce records carrying mu_H information. Computationally, no such record beats SM.

2. **The conditions are tight.** When A2 fails (6 cases), SM is genuinely suboptimal — the conditions are not superfluous. The no-blurring condition A2 is necessary.

3. **A different proof technique is needed.** The current two-step proof (conditional Jensen + no-blurring) has a logical gap. The computational evidence suggests a direct proof of global optimality should exist, perhaps via a single-step concavification argument.

4. **Correlated experiments never satisfy A1+A2+A3 in this setup.** Because each atom has a unique mu_L value, Var(mu_H | mu_L) = 0 (A1 fails). This is a limitation of the experiment construction, not of the theorem.

---

## 10. Next Steps: Full Mode Run

### Goal

Expand verification with heavier optimization and broader parameter/experiment coverage to strengthen the computational evidence.

### What full mode adds

| Parameter | Quick | Full | Factor |
|-----------|-------|------|--------|
| SLSQP starts | 200 | 2,000 | 10x |
| DE maxiter | 300 | 1,000 | 3.3x |
| DE popsize | 15 | 30 | 2x |
| Basin-hopping | OFF | 200 iters | new |
| y_under values | {0.2, 0.3} | {0.2, 0.3, 0.4} | 1.5x |
| y_H values | {2.0} | {1.5, 2.0} | 2x |
| sigma values | {0.25, 0.35} | {0.2, 0.3, 0.4} | 1.5x |
| w1 offsets | {0.02, 0.08} | {0.01, 0.05, 0.10} | 1.5x |
| delta values | {0.9, 1.5} | {0.5, 0.9, 1.5} | 1.5x |
| Correlated n_atoms | {8} | {8, 15} | 2x |
| Correlated seeds | {42} | {42, 123, 456} | 3x |
| Random seeds | 3 | 10 | 3.3x |
| N-type extensions | none | N=4,5 with 3 seeds | new |
| **Est. total cases** | **192** | **~2,600+** | **~13x** |

### Planned optimizations for full mode

1. **Early termination**: Skip remaining SLSQP starts if first ~20 starts + seeded starts all agree with SM. Most cases where A1+A2+A3 hold converge immediately from the SM seed.

2. **Adaptive SLSQP budget**: Start with 50 random starts; escalate to 2,000 only if any random start beats SM. Could cut time by ~90% for well-behaved cases.

3. **Conditional basin-hopping**: Only run BH on cases where DE or SLSQP found a nonzero gap. BH is expensive and sequential — skip it when the answer is already clear.

4. **Case-level parallelism**: Already implemented (16 workers across 16 CPUs).

**Estimated runtime with optimizations**: 15-30 minutes (vs hours without).

---

## Appendix: Files

| File | Description |
|------|-------------|
| `selective_memory_spec.md` | Mathematical specification |
| `selective_memory_verify.py` | Verification script (optimized) |
| `sm_results_quick.json` | Full JSON results (192 cases) |
| `run_log.txt` | Console output log |
| `results_report.md` | This report |
