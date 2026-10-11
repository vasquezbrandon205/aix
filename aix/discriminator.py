"""
AIX v3 — Phase 3: Adversarial Discriminator

Given top hypotheses, ACTIVELY SYNTHESIZE (via global optimization)
the input conditions that maximize predictive divergence between each
pair. This goes beyond disagreement *sampling* (LLM-ACES): we solve
the optimization problem directly.

    x* = argmax_x | pred_i(x) - pred_j(x)|

subject to x within a search domain (default: 2x the training range,
to find extrapolation disagreements where models diverge most).

Output: ranked list of (pair, optimal_x, divergence) — "run the
experiment at x* and at most one hypothesis survives."
"""
import numpy as np
from sympy import Symbol, lambdify
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DiscriminationResult:
    hyp_i: str          # structure_id of hypothesis i
    hyp_j: str          # structure_id of hypothesis j
    x_star: float       # maximally discriminating input
    divergence: float   # |pred_i(x*) - pred_j(x*)|
    divergence_ratio: float  # divergence / training MSE scale
    method: str = "differential_evolution"


class AdversarialDiscriminator:
    def __init__(self, templates: Dict, x_symbol: Symbol = None,
                 seed: int = 42):
        """
        Args:
            templates: {structure_id: (sympy_expr, description)}
            x_symbol: the input variable symbol
        """
        self.templates = templates
        self.x = x_symbol or Symbol('x')
        self.rng = np.random.default_rng(seed)
        # Pre-compile evaluators
        self._compiled = {}
        for tid, (expr, _) in templates.items():
            param_syms = sorted(
                {str(s) for s in expr.free_symbols if str(s).isupper()})
            fn = lambdify([self.x] + [Symbol(p) for p in param_syms],
                          expr, modules=["numpy"])
            self._compiled[tid] = (fn, param_syms)

    def predict(self, structure_id: str, params: Dict[str, float],
                x_vals: np.ndarray) -> np.ndarray:
        """Evaluate a hypothesis at given x values."""
        fn, param_syms = self._compiled[structure_id]
        pvals = [params.get(p, 0.0) for p in param_syms]
        try:
            result = np.asarray(fn(x_vals, *pvals), dtype=float)
            # Broadcast scalar to array if needed
            if result.ndim == 0:
                result = np.full_like(x_vals, result, dtype=float)
            return result  # keep NaN/inf as-is; validity checked by caller
        except Exception:
            return np.full_like(x_vals, np.nan, dtype=float)

    def _both_valid(self, hyp_i: Tuple[str, Dict],
                    hyp_j: Tuple[str, Dict], x_scalar: float) -> bool:
        """Check that both models produce finite predictions at x."""
        x_arr = np.array([x_scalar])
        pi = self.predict(hyp_i[0], hyp_i[1], x_arr)[0]
        pj = self.predict(hyp_j[0], hyp_j[1], x_arr)[0]
        return np.isfinite(pi) and np.isfinite(pj)

    def _divergence_objective(self, x_scalar: float,
                              hyp_i: Tuple[str, Dict],
                              hyp_j: Tuple[str, Dict]) -> float:
        """Negative divergence for minimization. Returns 0 (no reward)
        where either model is invalid — we want genuine disagreement,
        not numerical crashes."""
        if not self._both_valid(hyp_i, hyp_j, x_scalar):
            return 0.0
        x_arr = np.array([x_scalar])
        pi = self.predict(hyp_i[0], hyp_i[1], x_arr)[0]
        pj = self.predict(hyp_j[0], hyp_j[1], x_arr)[0]
        return -abs(pi - pj)

    def discriminate_pair(self, hyp_i: Tuple[str, Dict],
                           hyp_j: Tuple[str, Dict],
                           search_bounds: Tuple[float, float],
                           n_starts: int = 20,
                           train_mse_scale: float = 1.0
                           ) -> DiscriminationResult:
        """
        Find x* that maximizes |pred_i(x) - pred_j(x)| via multi-start
        local optimization (L-BFGS-B from diverse starting points) +
        dense grid seeding.

        This is *synthesis*, not sampling: we solve the optimization.
        """
        from scipy.optimize import minimize_scalar, minimize

        lo, hi = search_bounds
        best_x, best_div = None, -1.0

        # Phase A: dense grid to find promising regions
        grid = np.linspace(lo, hi, 200)
        divs = np.array([
            -self._divergence_objective(g, hyp_i, hyp_j) for g in grid
        ])
        # Top candidates as starting points
        top_idx = np.argsort(divs)[-n_starts:]

        # Phase B: local refinement from each promising start
        for idx in top_idx:
            x0 = grid[idx]
            try:
                res = minimize_scalar(
                    lambda x: self._divergence_objective(x, hyp_i, hyp_j),
                    bounds=(lo, hi), method='bounded',
                    options={'xatol': 1e-8},
                )
                div = -res.fun
                if div > best_div:
                    best_div, best_x = div, float(res.x)
            except Exception:
                continue

        # Fallback: grid best
        if best_x is None:
            best_idx = int(np.argmax(divs))
            best_x, best_div = float(grid[best_idx]), float(divs[best_idx])

        return DiscriminationResult(
            hyp_i=hyp_i[0], hyp_j=hyp_j[0],
            x_star=best_x, divergence=best_div,
            divergence_ratio=(best_div / train_mse_scale
                              if train_mse_scale > 0 else float('inf')),
        )

    def discriminate_all(self, champions: List[Tuple[str, Dict]],
                          search_bounds: Tuple[float, float],
                          train_mse_scale: float = 1.0,
                          top_k: Optional[int] = None
                          ) -> List[DiscriminationResult]:
        """
        Discriminate all pairs, ranked by divergence (most discriminating
        first). Returns: "run experiment at x* for the top pair."
        """
        results = []
        n = len(champions)
        for i in range(n):
            for j in range(i + 1, n):
                r = self.discriminate_pair(
                    champions[i], champions[j],
                    search_bounds, train_mse_scale=train_mse_scale)
                results.append(r)
        results.sort(key=lambda r: r.divergence, reverse=True)
        return results[:top_k] if top_k else results

    def recommend_experiment(self, champions: List[Tuple[str, Dict]],
                              search_bounds: Tuple[float, float],
                              train_mse_scale: float = 1.0
                              ) -> DiscriminationResult:
        """Single best discriminating experiment across all pairs."""
        results = self.discriminate_all(champions, search_bounds,
                                         train_mse_scale, top_k=1)
        return results[0] if results else None
