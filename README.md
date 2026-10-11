# AIX — Adversarial Identifiability Engine

**Don't find the best equation. Maintain every surviving hypothesis, synthesize the experiment that kills all but one, and prove what remains.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-20%2F20-brightgreen.svg)](#test-battery)

---

## The problem nobody is solving

Symbolic regression in 2026 can fit curves. It cannot do science.

- **EurekaBench** (Sep 2026): AI agents match humans at *prediction* (47.4% vs 48.8%) but score **under half** at verifiable mechanistic *insight* (42.4% vs 69.7%).
- **The identifiability gap** (LLM-ACES, 2026): structurally distinct equations, observationally indistinguishable on limited data. Every fitter converges to *an* answer. None tells you about the others.
- **Zero mainstream papers** on formal verification of discovered equations. An "AI-discovered law" cannot be published without human re-derivation, because nothing *proves* its properties.

The field outputs one equation. Real science maintains competing hypotheses.

## What AIX does differently

```
┌─────────────────────────────────────────────────────────┐
│  PHASE 1: FORMAL PROPERTY SIEVE                         │
│  Dimensional analysis (FORMAL) + limits (SEMI-FORMAL)   │
│  + invariants (HEURISTIC, labeled). Rejects impossible  │
│  equations before they waste compute.                   │
├─────────────────────────────────────────────────────────┤
│  PHASE 2: MULTI-HYPOTHESIS POPULATION                   │
│  K structurally distinct families evolve in parallel.   │
│  Tree-edit-distance diversity enforcement — families    │
│  cannot collapse into each other.                       │
├─────────────────────────────────────────────────────────┤
│  PHASE 3: ADVERSARIAL DISCRIMINATOR                     │
│  Actively SYNTHESIZE (via optimization) the input that  │
│  maximizes divergence between top hypotheses. Not       │
│  sampling — solving: x* = argmax |pred_i(x) - pred_j(x)|│
├─────────────────────────────────────────────────────────┤
│  PHASE 4: DISCOVERY DECISION TREE                       │
│  Output is NOT an equation. It's: ranked hypotheses +   │
│  discriminating experiments + verification certificates │
│  + honesty ledger (PROVEN / TESTED / HYPOTHESIZED).      │
└─────────────────────────────────────────────────────────┘
```

## Key results

**Formal Property Sieve** — 20/20 test battery:
- 10/10 real physics equations verified (harmonic oscillator, pendulum, logistic growth, Stokes drag, tanh saturation, ...)
- 10/10 broken variants correctly rejected (dimension mismatches, `sin`/`exp`/`log` of dimensioned quantities, false limit claims, wrong fixed points)

**Multi-hypothesis engine** — 7/8 structural families maintained on ambiguous data (damped oscillation + linear trend, no single template correct), diversity held at 0.50 (threshold 0.30), zero family collapse over 50 generations.

**Adversarial discriminator** — top pair diverges **4500x** training MSE at synthesized optimal input. Every recommendation verified by direct evaluation.

## How this differs from existing work

| | InsightSR (2026) | LLM-ACES (2026) | **AIX** |
|---|---|---|---|
| LLM role | Search-space designer | Operator priors + active sampling | Pluggable guide (heuristic now, API-ready) |
| Output | Single best equation | Single best + acquisition policy | **Ranked hypothesis set + discriminating experiments** |
| Identifiability | Not addressed | Disagreement sampling | **Adversarial synthesis** (optimization, not sampling) |
| Verification | None | None | **Formal sieve** (dimensional/limit/invariant) |
| Honesty | Standard | Standard | **Method ledger** (every claim graded) |

## Quick start

```bash
git clone https://github.com/vasquezbrandon205/aix.git
cd aix
pip install -r requirements.txt

# Run the sieve test battery (20 tests, ~2 seconds)
python -m aix.test_battery

# See it in action: end-to-end on a synthetic unknown system
python examples/end_to_end.py
```

## Verify a single equation

```python
from aix.sieve import verify
from sympy import Symbol

x, k, m = Symbol('x'), Symbol('k'), Symbol('m')
verdict = verify({
    "lhs": Symbol('dvdt'),
    "rhs": -(k/m) * x,                    # harmonic oscillator
    "var_dims": {
        Symbol('dvdt'): {"L": 1, "T": -2},
        x: {"L": 1}, k: {"M": 1, "T": -2}, m: {"M": 1},
    },
})
print(verdict["overall"])  # PASS
for name, check in verdict["checks"].items():
    print(f"  [{check['status']}] {name} ({check['method']})")
```

## Project status

- [x] **Phase 1**: Formal Property Sieve — complete, 20/20 battery
- [x] **Phase 2**: Multi-hypothesis population — complete, diversity validated
- [x] **Phase 3**: Adversarial discriminator — complete, 4500x divergence demonstrated
- [x] **Phase 4**: Discovery decision tree reports — complete, end-to-end tested
- [ ] **Phase 5**: GPU scale-up (131k population, builds on [AXIOM-X v2](docs/AXIOM-X-V2.md))
- [ ] **Phase 6**: LLM guide integration (interface ready, needs API key)
- [ ] **Phase 7**: PDE/SDE extension

## What's proven vs hypothesized

We grade every claim. From the latest end-to-end run:

| Claim | Status |
|---|---|
| Dimensional analysis is exact | **PROVEN** (symbolic) |
| Sieve rejects broken equations | **TESTED** (20/20 battery) |
| Diversity enforcement holds | **TESTED** (50 generations, no collapse) |
| Discriminator finds max-divergence inputs | **TESTED** (verified by direct eval) |
| True law is in the hypothesis set | **HYPOTHESIZED** (not guaranteed) |
| Recommended experiments are physically realizable | **HYPOTHESIZED** (needs domain review) |

## Citation

```bibtex
@software{vasquez2026aix,
  author = {Vasquez, Brandon Manuel},
  title = {AIX: Adversarial Identifiability Engine},
  year = {2026},
  url = {https://github.com/vasquezbrandon205/aix}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*Built on the [AXIOM-X](docs/AXIOM-X-V2.md) GPU evolution engine. The field fits curves. We're trying to do science.*
