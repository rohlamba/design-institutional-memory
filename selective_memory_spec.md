# Three-Type Selective Memory: Computational Verification

## Overview

This document specifies the mathematical model and computational task for verifying whether the **Selective Memory (SM) theorem** holds globally over all feasible garblings, or only within a restricted class.

### Background

The paper "The Design of Institutional Memory" proves a selective memory theorem for a three-type model where:
- The **current** decision (retain/fire) depends on μ_L = Pr(θ = L | record)
- The **future** exploitation (ratchet) depends on μ_H = Pr(θ = H | record)
- The middle type M decouples the two margins

The current proof establishes SM optimality via two steps:
1. **Conditional Jensen**: for any record, projecting to its μ_L-component weakly improves value (global step)
2. **No-blurring**: among records built from μ_L, full revelation of μ̃_L is optimal when a composite function g is convex (restricted-class step)

A gap was identified: Step 2 only optimizes within records that are garblings of μ̃_L, but Step 1 produces records that may carry information about μ̃_H. The question is whether any record exploiting μ̃_H information can beat SM.

**This script tests computationally whether SM is globally optimal under the theorem's conditions (A1, A2, A3), or whether counterexamples exist.**

---

## 1. Mathematical Model

### 1.1 Type Space and Posteriors

Three types: Θ = {L, M, H}.

An **operational posterior** is μ̃ = (μ̃_L, μ̃_M, μ̃_H) ∈ Δ₃ (the 2-simplex).

An **operational experiment** produces a finite collection of operational posteriors {μ̃₁, ..., μ̃ₙ} with probabilities {p₁, ..., pₙ}, satisfying Bayes plausibility: Σᵢ pᵢ μ̃ᵢ = π (the prior).

### 1.2 Garbling / Recording Rule

A **garbling** is a stochastic matrix W of dimension n × K, where:
- W[i,k] = Pr(signal = k | operational posterior = μ̃ᵢ)
- Each row sums to 1: Σₖ W[i,k] = 1 for all i
- All entries non-negative: W[i,k] ≥ 0

The **institutional posterior** for signal k is:
```
μₖ = Σᵢ pᵢ W[i,k] μ̃ᵢ / Σᵢ pᵢ W[i,k]
```

This automatically satisfies the martingale/garbling constraint E[μ̃ | μ] = μ.

### 1.3 Payoff Functions

**Current value** (retention decision, depends on μ_L):
```
h(μ_L) = max(0, (ȳ + σ - w₁) - (ȳ - y_L) · μ_L)
```
where:
- y_L = low-type current output
- ȳ = output of types M and H (current-period equivalence)
- σ = match surplus
- w₁ = contractual wage

This is convex in μ_L (max of zero and decreasing affine), with retention threshold:
```
c₁ʰ = (ȳ + σ - w₁) / (ȳ - y_L)
```

**Continuation value** (ratchet, depends on μ_H):
```
r(μ_H) = max(0, x(μ_H) + σ - w₂(μ_H))
```
where:
- x(μ_H) = μ_H · y_H + (1 - μ_H) · y̲ (expected continuation output)
- y̲ = continuation output of types L and M (continuation-equivalence)
- y_H = high-type continuation output
- w₂(μ_H) = max(w₁, x(μ_H)) (ratcheted wage)

This gives the piecewise form:
```
r(μ_H) = 0                                    if μ_H ≤ c₁ʳ
        = μ_H(y_H - y̲) + y̲ + σ - w₁         if c₁ʳ < μ_H < c₂
        = σ                                    if μ_H ≥ c₂
```
where c₁ʳ = (w₁ - σ - y̲)/(y_H - y̲) and c₂ = (w₁ - y̲)/(y_H - y̲).

**For r to be concave**, we need c₁ʳ ≤ 0, i.e., w₁ ≤ y̲ + σ.

**Total objective**:
```
V(garbling) = Σₖ πₖ · [h(μₖ_L) + δ · r(μₖ_H)]
```
where πₖ = Σᵢ pᵢ W[i,k] is the probability of signal k.

### 1.4 Selective Memory (SM) Record

SM reveals μ̃_L fully and replaces μ̃_H by its conditional mean m(μ̃_L) = E[μ̃_H | μ̃_L].

Under the nested-independence structure (see Section 1.5), SM groups all operational posteriors with the same μ̃_L value into a single signal.

SM value:
```
V_SM = Σⱼ Pr(μ̃_L = ℓⱼ) · [h(ℓⱼ) + δ · r(m(ℓⱼ))]
```

### 1.5 Operational Experiment Structures

#### Structure A: Nested Independence

Stage 1 determines μ̃_L ∈ {ℓ₁, ..., ℓ_J} with probabilities {p₁ᴸ, ..., p_Jᴸ}.
Stage 2 determines q = Pr(H | not-L) ∈ {q₁, ..., q_Q} with probabilities {p₁ᵠ, ..., p_Qᵠ}, independent of Stage 1.

Then: μ̃_H = (1 - μ̃_L) · q, μ̃_M = (1 - μ̃_L)(1 - q).

This gives n = J × Q operational posteriors.

Under this structure:
```
m(ℓ) = E[μ̃_H | μ̃_L = ℓ] = (1 - ℓ) · q̄    where q̄ = Σ pᵠⱼ qⱼ
```
which is affine in ℓ.

#### Structure B: Correlated Experiment (Non-nested)

The operational experiment produces posteriors (μ̃_L, μ̃_H) with arbitrary joint distribution on the simplex, subject only to Bayes plausibility.

This is the harder test case: μ̃_L and μ̃_H can be correlated in arbitrary ways, and m(ℓ) = E[μ̃_H | μ̃_L = ℓ] need not be affine.

#### Structure C: General N-type Extension

For N ≥ 4 types Θ = {L, M₁, M₂, ..., M_{N-2}, H}:
- Types M₁, ..., M_{N-2}, H are current-period equivalent (h depends on μ_L)
- Types L, M₁, ..., M_{N-2} are continuation-equivalent (r depends on μ_H)
- The posterior μ̃ ∈ Δ_N

The garbling problem and SM definition generalize directly.

### 1.6 Theorem Conditions

The SM theorem requires:

**(A1) Nondegeneracy**: Var(μ̃_H | μ̃_L) > 0 for a positive-probability set of μ̃_L values.

**(A2) No-blurring**: The composite function g(ℓ) = h(ℓ) + δ · r(m(ℓ)) is convex. Under the affine-m structure, this holds when the concave kink of r (mapped to ℓ-space) lies outside [0,1], i.e., c₂ ≥ q̄ (equivalently, ℓ* = 1 - c₂/q̄ ≤ 0).

**(A3) Strict gain**: E[r(μ̃_H) | μ̃_L = ℓ] < r(m(ℓ)) for a positive-probability set of ℓ values. Under piecewise-linear r, this requires the conditional support of μ̃_H | μ̃_L to cross the kink c₂.

### 1.7 Key Question

**Is V_SM ≥ V(W) for ALL feasible garblings W, or only for garblings W that are functions of μ̃_L?**

The computational task is to find the global maximum of V(W) over all W and compare to V_SM.

---

## 2. Parameter Restrictions

For the model to be well-posed:

| Condition | Formula | Ensures |
|-----------|---------|---------|
| h has interior threshold | y_L + σ < w₁ < ȳ + σ | 0 < c₁ʰ < 1 |
| r is concave | w₁ ≤ y̲ + σ | c₁ʳ ≤ 0 |
| Ratchet is interior | y̲ < w₁ < y_H | 0 < c₂ < 1 |
| g is convex (A2) | c₂ ≥ q̄ | No r-kink in ℓ-space |

Combined feasible region:
```
max(y_L + σ, y̲) < w₁ ≤ y̲ + σ,  w₁ < y_H
```

---

## 3. Computational Strategy

### 3.1 What to Test

For each parameterization:
1. Verify A1, A2, A3 hold
2. Compute V_SM
3. Compute V_unrestricted = max over ALL garblings
4. Compare: gap = V_unrestricted - V_SM
5. If gap > ε: counterexample found (SM is not globally optimal)
6. If gap ≤ ε for all parameterizations: evidence that SM is globally optimal

### 3.2 Experiment Structures to Test

**Tier 1: Nested independence** (affine m). This is the structure the paper focuses on. Test with varying J, Q, parameter values.

**Tier 2: Correlated experiments** (non-affine m). Construct operational posteriors where μ̃_L and μ̃_H are positively correlated, negatively correlated, or have nonlinear dependence. These are the hardest tests for the bridge gap.

**Tier 3: Adversarial construction**. Explicitly construct garblings that try to exploit the bridge gap by transmitting μ̃_H information.

**Tier 4: N ≥ 4 types**. Test with 4 and 5 types to check robustness.

### 3.3 Optimization Methods

Use multiple global optimization methods:
- **Differential evolution** (scipy): population-based global optimizer
- **Basin-hopping** (scipy): perturbed local search
- **SLSQP from many starts**: local optimizer with 2000+ random initializations
- **SM and full-revelation as explicit seeds**: ensure the optimizer knows about SM

### 3.4 Validation

For each reported optimum:
- Verify feasibility: all rows sum to 1, all entries in [0,1]
- Verify martingale property: E[μ̃ | signal] = institutional posterior
- Re-evaluate objective at the reported W
- Compare to SM starting point to catch optimizer failures

---

## 4. Output Specification

The script should produce:

1. **Per-case results**: For each parameterization, report V_SM, V_unrestricted, gap, and whether A1/A2/A3 hold.

2. **Counterexample details**: If gap > ε, report the full garbling matrix, the optimal institutional posteriors, and which operational atoms are mixed.

3. **Summary statistics**: Number of cases tested, number of counterexamples, max/min/mean gap.

4. **JSON output**: Machine-readable results for further analysis.
