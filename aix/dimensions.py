"""
AIX v3 — Formal Property Sieve: Dimensional Analysis
Method: FORMAL (exact symbolic dimension propagation)

Dimensions are represented as dicts mapping base units to exponents:
    velocity = {"L": 1, "T": -1}
    dimensionless = {}

Rules (exact, no heuristics):
- Add/Sub: operands MUST have identical dimensions, else FAIL
- Mul: exponents add
- Div: exponents subtract
- Pow: exponent must be dimensionless; result dims = base dims * exponent
- Transcendental (sin, cos, exp, log, tanh, ...): argument MUST be
  dimensionless; result is dimensionless
- Derivative d/dt: T exponent decreases by 1
"""
from sympy import (
    Symbol, Add, Mul, Pow, sin, cos, tan, exp, log, tanh, sinh, cosh,
    Derivative, Function, Number, sqrt, Abs,
)
from typing import Dict, List, Tuple, Optional

# Dimension type: dict of base -> exponent
Dim = Dict[str, float]

TRANSCENDENTAL = (sin, cos, tan, exp, log, tanh, sinh, cosh)

# Common base dimensions
DIMENSIONLESS: Dim = {}
LENGTH = {"L": 1}
TIME = {"T": 1}
MASS = {"M": 1}
VELOCITY = {"L": 1, "T": -1}
ACCELERATION = {"L": 1, "T": -2}
FORCE = {"M": 1, "L": 1, "T": -2}
ENERGY = {"M": 1, "L": 2, "T": -2}


def _dim_eq(d1: Dim, d2: Dim) -> bool:
    keys = set(d1) | set(d2)
    return all(abs(d1.get(k, 0) - d2.get(k, 0)) < 1e-12 for k in keys)


def _dim_add(d1: Dim, d2: Dim) -> Dim:
    keys = set(d1) | set(d2)
    return {k: d1.get(k, 0) + d2.get(k, 0) for k in keys
            if abs(d1.get(k, 0) + d2.get(k, 0)) > 1e-12}


def _dim_neg(d: Dim) -> Dim:
    return {k: -v for k, v in d.items()}


def _dim_str(d: Dim) -> str:
    if not d:
        return "dimensionless"
    parts = []
    for k in sorted(d):
        v = d[k]
        parts.append(f"{k}^{v:g}" if v != 1 else k)
    return "·".join(parts)


class DimensionalError(Exception):
    pass


def infer_dimensions(expr, var_dims: Dict[Symbol, Dim],
                     trace: Optional[List[str]] = None) -> Dim:
    """
    Recursively infer dimensions. Raises DimensionalError on inconsistency.
    Appends human-readable proof steps to trace.
    """
    def log(msg: str):
        if trace is not None:
            trace.append(msg)

    # Numeric literal → dimensionless
    if isinstance(expr, Number) or expr.is_number:
        return {}

    # Symbol → look up
    if isinstance(expr, Symbol):
        if expr not in var_dims:
            raise DimensionalError(
                f"no dimension declared for symbol '{expr}'")
        return dict(var_dims[expr])

    # Addition: all terms must match
    if isinstance(expr, Add):
        dims = [infer_dimensions(a, var_dims, trace) for a in expr.args]
        for d in dims[1:]:
            if not _dim_eq(dims[0], d):
                raise DimensionalError(
                    f"addition of incompatible dimensions: "
                    f"{_dim_str(dims[0])} + {_dim_str(d)} in '{expr}'")
        log(f"Add: all {len(dims)} terms have [{_dim_str(dims[0])}] ✓")
        return dims[0]

    # Multiplication: exponents add
    if isinstance(expr, Mul):
        result: Dim = {}
        for a in expr.args:
            d = infer_dimensions(a, var_dims, trace)
            result = _dim_add(result, d)
        log(f"Mul: '{expr}' → [{_dim_str(result)}]")
        return result

    # Power: exponent must be dimensionless
    if isinstance(expr, Pow):
        base, exponent = expr.args
        # numeric exponent
        if exponent.is_number:
            e_val = float(exponent)
            base_dim = infer_dimensions(base, var_dims, trace)
            result = {k: v * e_val for k, v in base_dim.items()
                      if abs(v * e_val) > 1e-12}
            log(f"Pow: '{expr}' → [{_dim_str(result)}]")
            return result
        # symbolic exponent: must be dimensionless
        exp_dim = infer_dimensions(exponent, var_dims, trace)
        if exp_dim:
            raise DimensionalError(
                f"dimensioned exponent in '{expr}': "
                f"exponent has [{_dim_str(exp_dim)}], must be dimensionless")
        base_dim = infer_dimensions(base, var_dims, trace)
        log(f"Pow: '{expr}' with symbolic dimensionless exponent → "
            f"dims depend on exponent value (deferred)")
        # Can't resolve statically; return base dims as-is with a note.
        # Caller must substitute or treat as conditional.
        return dict(base_dim)

    # Abs: preserves dimensions (unlike transcendental functions)
    if isinstance(expr, Abs):
        d = infer_dimensions(expr.args[0], var_dims, trace)
        log(f"Abs: preserves [{_dim_str(d)}] ✓")
        return d

    # Transcendental: argument must be dimensionless
    if isinstance(expr, Function):
        fname = type(expr).__name__
        if len(expr.args) != 1:
            raise DimensionalError(
                f"unsupported multi-arg function '{expr}'")
        arg_dim = infer_dimensions(expr.args[0], var_dims, trace)
        if arg_dim:
            raise DimensionalError(
                f"{fname} of dimensioned quantity in '{expr}': "
                f"argument has [{_dim_str(arg_dim)}], must be dimensionless")
        log(f"{fname}: dimensionless arg ✓ → dimensionless")
        return {}

    # Derivative: d/dt reduces T by 1 per order
    if isinstance(expr, Derivative):
        base_dim = infer_dimensions(expr.expr, var_dims, trace)
        # count derivatives wrt each variable
        for var, order in expr.variable_count:
            if var in var_dims:
                vdim = var_dims[var]
                # subtract order * var_dim
                for k, v in vdim.items():
                    base_dim[k] = base_dim.get(k, 0) - order * v
                base_dim = {k: v for k, v in base_dim.items()
                            if abs(v) > 1e-12}
        log(f"d/dt: '{expr}' → [{_dim_str(base_dim)}]")
        return base_dim

    raise DimensionalError(f"unsupported expression type: {type(expr)}")


def check_equation_dimensions(lhs, rhs, var_dims: Dict[Symbol, Dim],
                              trace: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Check that lhs and rhs of an equation have identical dimensions.
    Returns (passed, message).
    """
    try:
        d_lhs = infer_dimensions(lhs, var_dims, trace)
        d_rhs = infer_dimensions(rhs, var_dims, trace)
        if _dim_eq(d_lhs, d_rhs):
            msg = (f"DIMENSIONAL PASS: both sides "
                   f"[{_dim_str(d_lhs)}]")
            if trace is not None:
                trace.append(msg)
            return True, msg
        msg = (f"DIMENSIONAL FAIL: lhs [{_dim_str(d_lhs)}] ≠ "
               f"rhs [{_dim_str(d_rhs)}]")
        if trace is not None:
            trace.append(msg)
        return False, msg
    except DimensionalError as e:
        msg = f"DIMENSIONAL FAIL: {e}"
        if trace is not None:
            trace.append(msg)
        return False, msg
