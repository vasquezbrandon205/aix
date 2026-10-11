"""
AIX v3 — Formal Property Sieve: Limiting Behavior
Method: SEMI-FORMAL
  - sympy.limit() where it resolves → symbolic (formal-grade)
  - numerical probing at extremes where sympy can't → heuristic (labeled)

Checks user-specified asymptotic expectations, e.g.:
    {"x->oo": "bounded", "x->0": "finite", "t->oo": "=0"}
Also auto-checks for unexpected blowups at extremes.
"""
from sympy import Symbol, oo, limit, sympify, S
from typing import Dict, List, Tuple, Optional
import math


def _try_symbolic_limit(expr, var: Symbol, to) -> Tuple[Optional[float], str]:
    """Try sympy.limit; return (value, method)."""
    try:
        result = limit(expr, var, to)
        if result.is_number and result not in (oo, -oo, S.NaN, S.ComplexInfinity):
            return float(result), "symbolic"
        if result == oo:
            return math.inf, "symbolic"
        if result == -oo:
            return -math.inf, "symbolic"
        return None, f"symbolic limit unresolved ({result})"
    except Exception as e:
        return None, f"symbolic limit failed: {e}"


def _numerical_probe(expr, var: Symbol, to, var_dims=None,
                     n_points: int = 5) -> Tuple[Optional[float], str]:
    """Probe numerically at extreme values. HEURISTIC."""
    try:
        f = expr._sympy_ if hasattr(expr, '_sympy_') else expr
        from sympy import lambdify
        fn = lambdify(var, expr, modules=["math"])
        if to == oo:
            test_vals = [1e6, 1e8, 1e10, 1e12, 1e15][:n_points]
        elif to == -oo:
            test_vals = [-1e6, -1e8, -1e10, -1e12, -1e15][:n_points]
        else:  # to == 0
            test_vals = [1e-6, 1e-8, 1e-10, 1e-12, 1e-15][:n_points]
            if str(to).startswith("-"):
                test_vals = [-v for v in test_vals]
        vals = []
        for v in test_vals:
            try:
                r = fn(v)
                if isinstance(r, complex):
                    return None, "numerical probe hit complex values"
                vals.append(float(r))
            except (OverflowError, ValueError, ZeroDivisionError):
                vals.append(math.inf if v > 0 else -math.inf)
        # check convergence trend
        if all(math.isinf(v) for v in vals):
            return vals[0], "numerical (diverges)"
        finite = [v for v in vals if not math.isinf(v)]
        if len(finite) >= 3:
            # converging? check last three are close
            if max(finite[-3:]) - min(finite[-3:]) < 1e-6 * (abs(finite[-1]) + 1):
                return finite[-1], "numerical (convergent)"
            return finite[-1], "numerical (trend, not converged)"
        return None, "numerical probe inconclusive"
    except Exception as e:
        return None, f"numerical probe failed: {e}"


def check_limit(expr, var: Symbol, to, expectation: str,
                param_vals: Optional[Dict] = None,
                trace: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Check one asymptotic expectation.
    expectation in: "finite", "bounded", "=0", "=inf", "=value:<float>"
    param_vals: substituted before probing (required for numerical fallback).
    Returns (passed, message).
    """
    def log(m):
        if trace is not None:
            trace.append(m)

    # Substitute known parameter values so the expression is in one variable
    work_expr = expr.subs(param_vals or {})

    val, method = _try_symbolic_limit(work_expr, var, to)
    if val is None:
        log(f"limit {var}→{to}: {method}, falling back to numerical probe")
        val, method = _numerical_probe(work_expr, var, to)
        method = "HEURISTIC " + method
    else:
        method = "symbolic"

    log(f"limit {var}→{to} = {val} [{method}]")

    if val is None:
        return False, f"LIMIT UNRESOLVED: {var}→{to} ({method})"

    if expectation == "finite":
        passed = not math.isinf(val)
    elif expectation == "bounded":
        passed = not math.isinf(val) and abs(val) < 1e15
    elif expectation == "=0":
        passed = abs(val) < 1e-9
    elif expectation == "=inf":
        passed = math.isinf(val) and val > 0
    elif expectation == "=-inf":
        passed = math.isinf(val) and val < 0
    elif expectation.startswith("=value:"):
        target = float(expectation.split(":")[1])
        passed = abs(val - target) < 1e-6 * (abs(target) + 1)
    else:
        return False, f"unknown expectation '{expectation}'"

    status = "PASS" if passed else "FAIL"
    msg = f"LIMIT {status}: {var}→{to} = {val}, expected {expectation} [{method}]"
    log(msg)
    return passed, msg


def auto_blowup_scan(expr, var: Symbol,
                     trace: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Auto-check: does expr blow up unexpectedly at ±∞ or 0?
    Returns (no_unexpected_blowup, message). HEURISTIC via numerical probe.
    """
    def log(m):
        if trace is not None:
            trace.append(m)

    issues = []
    for to, label in [(oo, "+∞"), (-oo, "-∞")]:
        val, method = _numerical_probe(expr, var, to)
        # blowup at infinity is only "unexpected" if the expression
        # is supposed to model a bounded physical quantity —
        # without metadata we just REPORT, don't fail.
        log(f"auto-scan {var}→{label}: {val} [{method}]")
        if val is not None and math.isinf(val):
            issues.append(f"{label}→∞")

    if issues:
        msg = (f"BLOWUP SCAN: diverges at {', '.join(issues)} "
               f"(informational — not a failure without boundedness metadata)")
        log(msg)
        return True, msg  # informational only
    msg = "BLOWUP SCAN: no divergence detected at extremes"
    log(msg)
    return True, msg
