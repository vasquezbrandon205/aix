# AIX v3 — Phase 1 Report: Formal Property Sieve

**Status:** COMPLETE — test battery 20/20 (10 accept + 10 reject)
**Date:** 2026-10-11
**Location:** `~/workspace/bmv1919-review/aix-v3/aix/`

## What was built

A `verify()` function that takes an equation spec (sympy AST + variable
dimensions + expected properties) and returns per-check verdicts with
method grades and proof traces.

### Modules

| Module | Check | Method grade |
|---|---|---|
| `dimensions.py` | Dimensional consistency via exact symbolic propagation | **FORMAL** |
| `limits.py` | Asymptotic behavior via sympy.limit + numerical fallback | **SEMI-FORMAL** |
| `invariants.py` | Fixed points, invariant preservation, boundedness via RK4 | **HEURISTIC** |
| `sieve.py` | `verify()` orchestrator, verdict assembly, trace logging | — |

### Key design decisions

1. **Dimensions as dicts** (`{"L": 1, "T": -1}`), propagated exactly through
   Add/Mul/Pow/Derivative. Transcendental functions require dimensionless
   args (hard fail otherwise). `Abs` preserves dimensions (added after
   test battery caught the gap).

2. **Limits try symbolic first** (`sympy.limit`), fall back to numerical
   probing at extremes. Every result labeled with its method. Parameter
   values substituted before probing (bug caught by test battery).

3. **Invariants are honestly labeled HEURISTIC.** RK4 integration is
   evidence, not proof. The verdict structure makes this unmissable.

4. **Blowup scan is informational only** — reports divergence without
   failing, since "should this be bounded?" requires domain metadata
   the sieve doesn't have.

## Test battery results

**10 PASS** (real physics, all verified):
P1 exponential decay, P2 harmonic oscillator, P3 damped oscillator,
P4 pendulum, P5 logistic growth, P6 Stokes drag, P7 tanh saturation,
P8 free fall, P9 driven decay, P10 quadratic drag.

**10 FAIL** (broken variants, all correctly rejected):
F1 dim mismatch (L + L/T), F2 sin([L]), F3 exp([T]), F4 LHS/RHS mismatch,
F5 tanh([L/T]), F6 cubic blowup vs boundedness claim, F7 wrong fixed point,
F8 log([L]), F9 [L]+[M], F10 false →0 limit claim.

## Bugs caught by the battery (3)

1. `Abs()` not handled in dimension propagator → added (preserves dims).
2. Limit checker didn't substitute param values → numerical probe crashed
   on unbound symbols → fixed.
3. Two test specs were dimensionally sloppy (bare numbers subtracted from
   dimensioned quantities) → fixed specs, not the sieve. The sieve was
   right to reject them.

## What's PROVEN vs TESTED vs HYPOTHESIZED

- **PROVEN:** Dimensional analysis is exact symbolic checking. If it says
  dimensions balance, they balance. If it rejects, the rejection is correct.
- **TESTED:** Limit checker works on 20/20 battery cases. Symbolic limits
  are exact when sympy resolves; numerical fallback is labeled.
- **TESTED:** Invariant/fixed-point/boundedness checks work numerically
  on battery cases. Labeled heuristic — not proof.
- **HYPOTHESIZED:** That this sieve, combined with Phases 2-4, will
  produce verifiable scientific insight beyond what prediction-only
  systems achieve (the EurekaBench gap). Not yet tested.

## What's NOT in Phase 1 (by design)

- Z3/Lean4 integration — evaluated, deferred. sympy's symbolic engine
  covers the dimensional + limit cases; full SMT is Phase 1b if needed.
- Multi-variable PDE support — single-state ODE only. PDE extension is
  future work.
- Conservation law *derivation* — the sieve *checks* user-specified
  invariants; it doesn't discover them. Discovery is Phase 2-4.

## Ready for Phase 2

The sieve is a standalone module with a clean interface:
`verify(eq_spec) -> verdict`. Phase 2 (multi-hypothesis population) can
call it as a filter on candidate equations.
