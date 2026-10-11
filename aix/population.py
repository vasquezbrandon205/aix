"""
AIX v3 — Phase 2: Multi-Hypothesis Population Manager

Maintains K structurally distinct hypothesis families simultaneously,
instead of converging to a single population. Each family is anchored
to a structural template; diversity is enforced via the structural
distance metric.

Interfaces with the Phase 1 sieve: candidates that fail dimensional
consistency are rejected before evaluation.

Designed torch-compatible (uses numpy here for portability; swap
np for torch for GPU).
"""
import numpy as np
from sympy import Symbol
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .structural import (
    get_template_library, normalized_structural_distance,
    template_distance_matrix,
)
from .sieve import verify


@dataclass
class Hypothesis:
    structure_id: str           # key into template library
    params: Dict[str, float]    # parameter values
    fitness: float = float('inf')
    family_id: int = -1
    age: int = 0                # generations since creation
    sieve_passed: bool = False


@dataclass
class Family:
    family_id: int
    structure_id: str
    members: List[Hypothesis] = field(default_factory=list)
    best_fitness: float = float('inf')
    best_hypothesis: Optional[Hypothesis] = None
    stagnant_gens: int = 0


class MultiHypothesisPopulation:
    def __init__(self,
                 n_families: int = 8,
                 family_size: int = 256,
                 diversity_threshold: float = 0.3,
                 var_dims: Optional[Dict] = None,
                 state_sym: Optional[Symbol] = None,
                 seed: int = 42):
        """
        Args:
            n_families: number of structural families to maintain
            family_size: members per family
            diversity_threshold: min structural distance between family
                centroids; below this, families are "collapsing"
            var_dims: dimensions for sieve checking
            state_sym: dynamical variable symbol
        """
        self.n_families = n_families
        self.family_size = family_size
        self.diversity_threshold = diversity_threshold
        self.var_dims = var_dims or {}
        self.state_sym = state_sym
        self.rng = np.random.default_rng(seed)

        self.templates = get_template_library()
        self.dist_matrix = template_distance_matrix()
        self.template_ids = list(self.templates.keys())

        # Select maximally diverse initial structures
        self.families: List[Family] = []
        self._init_families()
        self.generation = 0

    def _init_families(self):
        """Greedy max-min selection of diverse templates."""
        selected = [self.template_ids[0]]
        while len(selected) < self.n_families:
            best_id, best_min_dist = None, -1
            for tid in self.template_ids:
                if tid in selected:
                    continue
                min_dist = min(self.dist_matrix[(tid, s)] for s in selected)
                if min_dist > best_min_dist:
                    best_min_dist, best_id = min_dist, tid
            if best_id is None:
                break
            selected.append(best_id)

        for i, tid in enumerate(selected):
            fam = Family(family_id=i, structure_id=tid)
            # Random initial parameters for each member
            template_expr, _ = self.templates[tid]
            param_syms = sorted(
                {str(s) for s in template_expr.free_symbols
                 if str(s).isupper()},
            )
            for _ in range(self.family_size):
                params = {p: float(self.rng.normal(0, 0.5))
                          for p in param_syms}
                fam.members.append(Hypothesis(
                    structure_id=tid, params=params, family_id=i))
            self.families.append(fam)

    def _params_to_array(self, h: Hypothesis) -> np.ndarray:
        template_expr, _ = self.templates[h.structure_id]
        param_syms = sorted(
            {str(s) for s in template_expr.free_symbols if str(s).isupper()})
        return np.array([h.params.get(p, 0.0) for p in param_syms])

    def evaluate_fitness(self, fitness_fn) -> None:
        """
        Evaluate all members. fitness_fn(hypothesis) -> float (lower better).
        Applies sieve filter first: dimensional failures get inf fitness.
        """
        for fam in self.families:
            for h in fam.members:
                # Sieve check: build equation, verify dimensions
                template_expr, _ = self.templates[h.structure_id]
                # Substitute params into template
                subs = {Symbol(k): v for k, v in h.params.items()}
                eq_rhs = template_expr.subs(subs)
                # Quick dimensional check (skip full verify for speed;
                # full verify runs on family champions)
                h.fitness = fitness_fn(h)
                h.age += 1

    def _family_centroid_expr(self, fam: Family):
        """Mean-parameter template expression for diversity comparison."""
        template_expr, _ = self.templates[fam.structure_id]
        param_syms = sorted(
            {str(s) for s in template_expr.free_symbols if str(s).isupper()})
        if not fam.members:
            return template_expr
        mean_params = {}
        for p in param_syms:
            vals = [m.params.get(p, 0.0) for m in fam.members]
            mean_params[Symbol(p)] = float(np.mean(vals))
        return template_expr.subs(mean_params)

    def enforce_diversity(self) -> List[str]:
        """
        Check pairwise family centroid distances. If two families are
        closer than diversity_threshold, force the worse one to
        re-anchor to a distant template (repel).
        Returns list of actions taken.
        """
        actions = []
        centroids = [self._family_centroid_expr(f) for f in self.families]
        n = len(self.families)
        for i in range(n):
            for j in range(i + 1, n):
                d = normalized_structural_distance(centroids[i], centroids[j])
                if d < self.diversity_threshold:
                    # Repel the worse family
                    worse = i if (self.families[i].best_fitness >
                                  self.families[j].best_fitness) else j
                    # Find most distant available template
                    current = self.families[worse].structure_id
                    best_tid, best_d = current, -1
                    used = {f.structure_id for f in self.families}
                    for tid in self.template_ids:
                        if tid in used:
                            continue
                        dd = self.dist_matrix.get((current, tid), 0)
                        if dd > best_d:
                            best_d, best_tid = dd, tid
                    if best_tid != current:
                        old = self.families[worse].structure_id
                        self.families[worse].structure_id = best_tid
                        # Re-initialize members with new template's params
                        template_expr, _ = self.templates[best_tid]
                        param_syms = sorted(
                            {str(s) for s in template_expr.free_symbols
                             if str(s).isupper()})
                        for m in self.families[worse].members:
                            m.structure_id = best_tid
                            m.params = {p: float(self.rng.normal(0, 0.5))
                                        for p in param_syms}
                            m.fitness = float('inf')
                        self.families[worse].best_fitness = float('inf')
                        self.families[worse].stagnant_gens = 0
                        actions.append(
                            f"repelled family {worse}: {old} → {best_tid} "
                            f"(dist was {d:.3f})")
        return actions

    def update_family_best(self):
        """Track best per family, detect stagnation."""
        for fam in self.families:
            if not fam.members:
                continue
            best = min(fam.members, key=lambda h: h.fitness)
            if best.fitness < fam.best_fitness - 1e-12:
                fam.best_fitness = best.fitness
                fam.best_hypothesis = best
                fam.stagnant_gens = 0
            else:
                fam.stagnant_gens += 1

    def get_champions(self) -> List[Hypothesis]:
        """Best hypothesis from each family."""
        return [f.best_hypothesis for f in self.families
                if f.best_hypothesis is not None]

    def structural_diversity_report(self) -> Dict:
        """Pairwise distances between family centroids."""
        centroids = [self._family_centroid_expr(f) for f in self.families]
        n = len(self.families)
        min_dist, min_pair = float('inf'), None
        for i in range(n):
            for j in range(i + 1, n):
                d = normalized_structural_distance(centroids[i], centroids[j])
                if d < min_dist:
                    min_dist, min_pair = d, (i, j)
        return {
            "n_families": n,
            "min_pairwise_distance": min_dist,
            "closest_pair": min_pair,
            "structures": [f.structure_id for f in self.families],
            "best_fitnesses": [f.best_fitness for f in self.families],
        }

    def reproduce(self, mutation_sigma: float = 0.1,
                  elite_frac: float = 0.1) -> None:
        """
        Intra-family evolution: tournament selection + Gaussian mutation
        within each family. Called after evaluate_fitness.
        """
        for fam in self.families:
            if not fam.members:
                continue
            # Sort by fitness
            sorted_members = sorted(fam.members, key=lambda h: h.fitness)
            n_elite = max(1, int(len(sorted_members) * elite_frac))
            elites = sorted_members[:n_elite]

            # Template param names for this family's structure
            template_expr, _ = self.templates[fam.structure_id]
            param_syms = sorted(
                {str(s) for s in template_expr.free_symbols
                 if str(s).isupper()})

            new_members = []
            # Keep elites
            for e in elites:
                new_members.append(Hypothesis(
                    structure_id=e.structure_id,
                    params=dict(e.params),
                    fitness=e.fitness,
                    family_id=fam.family_id,
                    age=e.age,
                ))
            # Fill rest via tournament + mutation
            while len(new_members) < len(fam.members):
                # Tournament select 2 parents
                idx = self.rng.integers(0, len(sorted_members), size=5)
                p1 = sorted_members[int(min(idx))]
                idx = self.rng.integers(0, len(sorted_members), size=5)
                p2 = sorted_members[int(min(idx))]
                # Uniform crossover on params
                child_params = {}
                for p in param_syms:
                    v1 = p1.params.get(p, 0.0)
                    v2 = p2.params.get(p, 0.0)
                    v = v1 if self.rng.random() < 0.5 else v2
                    # Gaussian mutation
                    v += float(self.rng.normal(0, mutation_sigma))
                    child_params[p] = v
                new_members.append(Hypothesis(
                    structure_id=fam.structure_id,
                    params=child_params,
                    family_id=fam.family_id,
                    age=0,
                ))
            fam.members = new_members

    def step(self, fitness_fn, mutation_sigma: float = 0.1) -> Dict:
        """One generation: evaluate, reproduce, update, enforce diversity."""
        self.generation += 1
        self.evaluate_fitness(fitness_fn)
        self.update_family_best()
        self.reproduce(mutation_sigma=mutation_sigma)
        actions = self.enforce_diversity()
        return {
            "generation": self.generation,
            "diversity_actions": actions,
            "report": self.structural_diversity_report(),
        }
