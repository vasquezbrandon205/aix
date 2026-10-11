"""
AIX v3 — Formal Property Sieve: main verify() entry point.

Each check is labeled with its method grade:
  FORMAL      — exact symbolic proof (dimensional analysis)
  SEMI-FORMAL — symbolic where possible, numerical fallback (limits)
  HEURISTIC   — numerical evidence, not proof (invariants, boundedness)

verify() takes:
  eq_spec: dict with:
    "lhs": sympy expression (typically Derivative or Symbol)
    "rhs": sympy expression
    "var_dims": {Symbol: Dim}  — dimensions for every symbol
    "state_sym": Symbol        — the dynamical variable (for trajectory checks)
    "param_vals": {Symbol: float} — parameter values (for numerical checks)
    "limits": {var_name: {to: expectation}} — e.g. {"x": {"oo": "bounded"}}
    "fixed_points": [float]    — expected equilibria (optional)
    "invariant": sympy expr    — conserved quantity (optional)
    "check_boundedness": bool  — whether to require bounded trajectories
    "ics": [float]             — initial conditions for trajectory checks

Returns a structured verdict with per-check proof traces.
"""
from sympy import Symbol, oo
from typing import Dict, List, Any

from .dimensions import check_equation_dimensions
from .limits import check_limit, auto_blowup_scan
from .invariants import check_fixed_points, check_invariant, check_boundedness

METHOD_GRADES = {
    "dimensional_consistency": "FORMAL",
    "limiting_behavior": "SEMI-FORMAL",
    "blowup_scan": "HEURISTIC (informational)",
    "fixed_points": "HEURISTIC",
    "invariant": "HEURISTIC",
    "boundedness": "HEURISTIC",
}


def verify(eq_spec: Dict[str, Any]) -> Dict[str, Any]:
    trace: List[str] = []
    checks: Dict[str, Dict[str, Any]] = {}
    overall = True

    lhs = eq_spec["lhs"]
    rhs = eq_spec["rhs"]
    var_dims = eq_spec["var_dims"]

    trace.append(f"=== Verifying: {lhs} = {rhs} ===")

    # ---- 1. Dimensional consistency (FORMAL) ----
    dim_trace: List[str] = []
    passed, msg = check_equation_dimensions(lhs, rhs, var_dims, dim_trace)
    checks["dimensional_consistency"] = {
        "status": "PASS" if passed else "FAIL",
        "method": METHOD_GRADES["dimensional_consistency"],
        "message": msg,
        "trace": dim_trace,
    }
    trace.extend(dim_trace)
    overall = overall and passed

    # ---- 2. Limiting behavior (SEMI-FORMAL) ----
    for var_name, expectations in eq_spec.get("limits", {}).items():
        var = Symbol(var_name)
        for to_str, expectation in expectations.items():
            to = {"oo": oo, "-oo": -oo, "0": 0}.get(to_str, to_str)
            lim_trace: List[str] = []
            passed, msg = check_limit(rhs, var, to, expectation,
                                      param_vals=eq_spec.get("param_vals"),
                                      trace=lim_trace)
            key = f"limit_{var_name}->{to_str}"
            checks[key] = {
                "status": "PASS" if passed else "FAIL",
                "method": METHOD_GRADES["limiting_behavior"],
                "message": msg,
                "trace": lim_trace,
            }
            trace.extend(lim_trace)
            overall = overall and passed

    # ---- 3. Auto blowup scan (informational) ----
    state_sym = eq_spec.get("state_sym")
    if state_sym is not None:
        scan_trace: List[str] = []
        _, msg = auto_blowup_scan(rhs, state_sym, scan_trace)
        checks["blowup_scan"] = {
            "status": "INFO",
            "method": METHOD_GRADES["blowup_scan"],
            "message": msg,
            "trace": scan_trace,
        }
        trace.extend(scan_trace)

    # ---- 4. Fixed points (HEURISTIC) ----
    if eq_spec.get("fixed_points") is not None and state_sym is not None:
        fp_trace: List[str] = []
        passed, msg = check_fixed_points(
            rhs, state_sym, eq_spec["fixed_points"],
            eq_spec.get("param_vals", {}), trace=fp_trace)
        checks["fixed_points"] = {
            "status": "PASS" if passed else "FAIL",
            "method": METHOD_GRADES["fixed_points"],
            "message": msg,
            "trace": fp_trace,
        }
        trace.extend(fp_trace)
        overall = overall and passed

    # ---- 5. Invariant (HEURISTIC) ----
    if eq_spec.get("invariant") is not None and state_sym is not None:
        inv_trace: List[str] = []
        passed, msg = check_invariant(
            rhs, state_sym, eq_spec["invariant"],
            eq_spec.get("param_vals", {}),
            eq_spec.get("ics", [1.0]), trace=inv_trace)
        checks["invariant"] = {
            "status": "PASS" if passed else "FAIL",
            "method": METHOD_GRADES["invariant"],
            "message": msg,
            "trace": inv_trace,
        }
        trace.extend(inv_trace)
        overall = overall and passed

    # ---- 6. Boundedness (HEURISTIC, opt-in) ----
    if eq_spec.get("check_boundedness") and state_sym is not None:
        b_trace: List[str] = []
        passed, msg = check_boundedness(
            rhs, state_sym, eq_spec.get("param_vals", {}),
            eq_spec.get("ics", [1.0]), trace=b_trace)
        checks["boundedness"] = {
            "status": "PASS" if passed else "FAIL",
            "method": METHOD_GRADES["boundedness"],
            "message": msg,
            "trace": b_trace,
        }
        trace.extend(b_trace)
        overall = overall and passed

    return {
        "equation": f"{lhs} = {rhs}",
        "overall": "PASS" if overall else "FAIL",
        "checks": checks,
        "method_grades": METHOD_GRADES,
        "trace": trace,
    }


def print_verdict(verdict: Dict[str, Any]):
    print(f"\n{'='*60}")
    print(f"  {verdict['equation']}")
    print(f"  OVERALL: {verdict['overall']}")
    print(f"{'='*60}")
    for name, c in verdict["checks"].items():
        print(f"  [{c['status']:4s}] {name:28s} ({c['method']})")
        print(f"         {c['message']}")
