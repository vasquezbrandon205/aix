"""
AIX v3 — Phase 2: Structural Distance Metric
Tree edit distance on sympy expression ASTs.

Used to enforce hypothesis diversity: two equations are "structurally
distinct" if their AST edit distance exceeds a threshold, regardless of
how close their parameters are.
"""
from sympy import Symbol, Function, Number
from typing import Dict, Tuple


def _node_label(expr) -> str:
    """Canonical label for an AST node."""
    if isinstance(expr, Symbol):
        # Distinguish variables from parameters by naming convention:
        # single lowercase letter = variable, else parameter
        name = str(expr)
        return f"VAR:{name}" if len(name) == 1 and name.islower() else f"PARAM:{name}"
    if isinstance(expr, Number) or expr.is_number:
        return "NUM"
    if isinstance(expr, Function):
        return f"FUNC:{type(expr).__name__}"
    # Operator nodes: Add, Mul, Pow
    return f"OP:{type(expr).__name__}"


def _children(expr) -> list:
    """Ordered children of an AST node."""
    if isinstance(expr, Symbol) or isinstance(expr, Number) or expr.is_number:
        return []
    return list(expr.args)


def tree_edit_distance(t1, t2, _memo: dict = None) -> int:
    """
    Simplified tree edit distance (Zhang-Shasha style, recursive with
    memoization on (id1, id2) — works for small expression trees).

    Operations: relabel (cost 1 if labels differ), delete/insert subtree
    (cost = subtree size).
    """
    if _memo is None:
        _memo = {}
    key = (id(t1), id(t2))
    if key in _memo:
        return _memo[key]

    l1, l2 = _node_label(t1), _node_label(t2)
    c1, c2 = _children(t1), _children(t2)

    # Both leaves
    if not c1 and not c2:
        result = 0 if l1 == l2 else 1
        _memo[key] = result
        return result

    # One is leaf: cost = relabel + insert/delete all children of other
    def subtree_size(t):
        return 1 + sum(subtree_size(c) for c in _children(t))

    if not c1:
        result = (0 if l1 == l2 else 1) + sum(subtree_size(c) for c in c2)
        _memo[key] = result
        return result
    if not c2:
        result = (0 if l1 == l2 else 1) + sum(subtree_size(c) for c in c1)
        _memo[key] = result
        return result

    # Both internal: relabel cost + optimal child alignment
    # For ordered children, use simple DP alignment (like string edit distance
    # but with subtree distances as substitution costs)
    relabel = 0 if l1 == l2 else 1
    m, n = len(c1), len(c2)
    # dp[i][j] = min cost to align c1[:i] with c2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + subtree_size(c1[i-1])
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + subtree_size(c2[j-1])
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = tree_edit_distance(c1[i-1], c2[j-1], _memo)
            dp[i][j] = min(
                dp[i-1][j] + subtree_size(c1[i-1]),   # delete
                dp[i][j-1] + subtree_size(c2[j-1]),   # insert
                dp[i-1][j-1] + sub_cost,              # align
            )
    result = relabel + dp[m][n]
    _memo[key] = result
    return result


def normalized_structural_distance(t1, t2) -> float:
    """
    TED normalized via Dice-style: 2 * TED / (size1 + size2).
    0 = identical structure. Larger = more different.
    Note: can exceed 1.0 for very different trees (not a strict [0,1]
    normalization, but monotonic — larger always means more different).
    For diversity enforcement, threshold ~0.3 separates "same family"
    from "distinct".
    """
    def size(t):
        return 1 + sum(size(c) for c in _children(t))
    raw = tree_edit_distance(t1, t2)
    total = size(t1) + size(t2)
    return (2.0 * raw / total) if total > 0 else 0.0


# ============================================================
# Structural template library
# Maps structure_id -> (sympy template expression, description)
# ============================================================
def get_template_library():
    from sympy import sin, cos, exp, tanh, log, sqrt, Abs
    x = Symbol('x')
    # Parameters use uppercase to distinguish from variables
    A, B, C, D = Symbol('A'), Symbol('B'), Symbol('C'), Symbol('D')
    return {
        "linear":       (A * x, "a*x"),
        "affine":       (A * x + B, "a*x + b"),
        "quadratic":    (A * x**2 + B * x + C, "a*x^2 + b*x + c"),
        "exponential":  (A * exp(B * x), "a*exp(b*x)"),
        "decay":        (A * exp(-B * x), "a*exp(-b*x)"),
        "sinusoidal":   (A * sin(B * x + C), "a*sin(b*x + c)"),
        "saturating":   (A * tanh(B * x), "a*tanh(b*x)"),
        "michaelis":    (A * x / (B + x), "a*x/(b+x)"),
        "power":        (A * x**B, "a*x^b"),
        "logarithmic":  (A * log(B * x + 1), "a*log(b*x+1)"),
        "poly_exp":     (A * x * exp(-B * x), "a*x*exp(-b*x)"),
        "damped_osc":   (A * exp(-B * x) * sin(C * x), "a*exp(-b*x)*sin(c*x)"),
    }


def template_distance_matrix() -> Dict[Tuple[str, str], float]:
    """Precompute pairwise normalized structural distances."""
    lib = get_template_library()
    ids = list(lib.keys())
    matrix = {}
    for i, id1 in enumerate(ids):
        for id2 in ids[i:]:
            d = normalized_structural_distance(lib[id1][0], lib[id2][0])
            matrix[(id1, id2)] = d
            matrix[(id2, id1)] = d
    return matrix
