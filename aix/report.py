"""
AIX v3 — Phase 4: Discovery Decision Tree Report Generator

The output of the system is NOT an equation. It is:
  1. Ranked hypothesis set (with fitness, structural IDs, sieve verdicts)
  2. Discriminating experiments (ranked by divergence)
  3. Formal verification certificates (per-hypothesis sieve traces)
  4. Method honesty ledger (what's PROVEN / TESTED / HYPOTHESIZED)

This is the "last mile" artifact: what a scientist needs to decide
what to do next, not just what the model thinks.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .sieve import verify
from .discriminator import DiscriminationResult


@dataclass
class HypothesisReport:
    rank: int
    structure_id: str
    description: str
    equation_str: str
    params: Dict[str, float]
    train_fitness: float
    sieve_verdict: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class ExperimentRecommendation:
    rank: int
    hyp_a: str
    hyp_b: str
    x_star: float
    expected_divergence: float
    divergence_ratio: float
    interpretation: str


class DiscoveryReport:
    def __init__(self, title: str = "AIX Discovery Report"):
        self.title = title
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.hypotheses: List[HypothesisReport] = []
        self.experiments: List[ExperimentRecommendation] = []
        self.method_ledger: Dict[str, str] = {}
        self.warnings: List[str] = []

    def add_hypothesis(self, h: HypothesisReport):
        self.hypotheses.append(h)

    def add_experiment(self, e: ExperimentRecommendation):
        self.experiments.append(e)

    def set_method_ledger(self, ledger: Dict[str, str]):
        """Maps claim -> PROVEN / TESTED / HYPOTHESIZED."""
        self.method_ledger = ledger

    def add_warning(self, w: str):
        self.warnings.append(w)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            f"Generated: {self.timestamp}",
            "",
            "## Ranked Hypotheses",
            "",
            "| Rank | Structure | Equation | Train Fitness | Sieve |",
            "|------|-----------|----------|---------------|-------|",
        ]
        for h in sorted(self.hypotheses, key=lambda x: x.rank):
            sieve = (h.sieve_verdict["overall"]
                     if h.sieve_verdict else "not run")
            lines.append(
                f"| {h.rank} | {h.structure_id} | `{h.equation_str}` | "
                f"{h.train_fitness:.4e} | {sieve} |")
        lines += ["", "## Discriminating Experiments", "",
                  "Ranked by expected divergence. Run the top experiment;",
                  "at most one hypothesis survives.", "",
                  "| Rank | A vs B | x* | Expected Δ | Δ / train MSE |",
                  "|------|--------|----|-----------|----------------|"]
        for e in sorted(self.experiments, key=lambda x: x.rank):
            lines.append(
                f"| {e.rank} | {e.hyp_a} vs {e.hyp_b} | {e.x_star:+.3f} | "
                f"{e.expected_divergence:.3f} | {e.divergence_ratio:.1f}x |")
        for e in sorted(self.experiments, key=lambda x: x.rank):
            lines += ["", f"### Experiment {e.rank}: {e.hyp_a} vs {e.hyp_b}",
                      f"- **Run at:** x = {e.x_star:+.4f}",
                      f"- **Expect:** |pred_A - pred_B| = {e.expected_divergence:.4f}",
                      f"- **Interpretation:** {e.interpretation}"]

        lines += ["", "## Verification Certificates", ""]
        for h in sorted(self.hypotheses, key=lambda x: x.rank):
            lines.append(f"### {h.structure_id} (rank {h.rank})")
            if h.sieve_verdict:
                v = h.sieve_verdict
                lines.append(f"- Overall: **{v['overall']}**")
                for name, c in v["checks"].items():
                    lines.append(f"  - [{c['status']}] {name} ({c['method']})")
            else:
                lines.append("- Sieve not run")
            for note in h.notes:
                lines.append(f"- Note: {note}")
            lines.append("")

        lines += ["## Method Honesty Ledger", "",
                  "| Claim | Status |",
                  "|-------|--------|"]
        for claim, status in self.method_ledger.items():
            lines.append(f"| {claim} | {status} |")

        if self.warnings:
            lines += ["", "## Warnings", ""]
            for w in self.warnings:
                lines.append(f"- ⚠ {w}")

        return "\n".join(lines)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_markdown())
        return path


def build_report(champions: List, templates: Dict,
                 disc_results: List[DiscriminationResult],
                 var_dims: Optional[Dict] = None,
                 state_sym=None,
                 title: str = "AIX Discovery Report") -> DiscoveryReport:
    """
    Assemble a full discovery report from Phase 2 champions,
    Phase 3 discriminations, and Phase 1 sieve verdicts.

    champions: list of Hypothesis (from population.get_champions())
    """
    from sympy import Symbol
    report = DiscoveryReport(title=title)

    # Hypotheses with sieve certificates
    for rank, h in enumerate(
            sorted(champions, key=lambda c: c.fitness), start=1):
        template_expr, desc = templates[h.structure_id]
        subs = {Symbol(k): v for k, v in h.params.items()}
        eq_str = str(template_expr.subs(subs))

        # Run sieve if dimensions available
        sieve_verdict = None
        notes = []
        if var_dims is not None and state_sym is not None:
            try:
                sieve_verdict = verify({
                    "lhs": Symbol("dydx"),
                    "rhs": template_expr.subs(subs),
                    "var_dims": {**var_dims, Symbol("dydx"): var_dims.get(
                        state_sym, {})},
                })
            except Exception as e:
                notes.append(f"sieve error: {e}")
        else:
            notes.append("sieve skipped: no dimension metadata provided")

        report.add_hypothesis(HypothesisReport(
            rank=rank, structure_id=h.structure_id,
            description=desc, equation_str=eq_str,
            params=dict(h.params), train_fitness=h.fitness,
            sieve_verdict=sieve_verdict, notes=notes))

    # Experiments
    for rank, r in enumerate(disc_results, start=1):
        interp = (
            f"If measured value at x={r.x_star:+.3f} is closer to "
            f"{r.hyp_i}'s prediction, {r.hyp_j} is falsified (and vice versa). "
            f"Expected separation is {r.divergence_ratio:.1f}x the training "
            f"noise floor — well above measurement uncertainty.")
        report.add_experiment(ExperimentRecommendation(
            rank=rank, hyp_a=r.hyp_i, hyp_b=r.hyp_j,
            x_star=r.x_star, expected_divergence=r.divergence,
            divergence_ratio=r.divergence_ratio,
            interpretation=interp))

    # Honesty ledger
    report.set_method_ledger({
        "Hypothesis fitness rankings": "TESTED (on training data)",
        "Structural diversity of hypothesis set": "TESTED (TED metric)",
        "Discriminating experiment predictions": "TESTED (numerical optimization)",
        "Dimensional consistency of hypotheses": "PROVEN (formal, where sieve ran)",
        "That the true law is in the hypothesis set": "HYPOTHESIZED (not guaranteed)",
        "That discriminating experiments are physically realizable":
            "HYPOTHESIZED (requires domain expert review)",
    })

    report.add_warning(
        "No hypothesis has been confirmed — only ranked and discriminated. "
        "Run the recommended experiments before claiming discovery.")
    report.add_warning(
        "Sieve dimensional checks require correct variable metadata. "
        "Garbage metadata → garbage certificates.")

    return report
