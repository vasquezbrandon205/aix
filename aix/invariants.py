"""
AIX v3 — Formal Property Sieve: Conservation & Invariants
Method: HEURISTIC (numerical trajectory integration)
  Labeled as such. These are evidence, not proof.

Checks:
- Fixed points: does dx/dt = 0 where the user says equilibria should be?
- Invariant preservation: given I(x) that should be constant along
  trajectories, integrate numerically and measure drift.
- Boundedness: integrate from several ICs, check for unexpected blowup
  within finite time.
"""
from sympy import Symbol, lambdify
from typing import Dict, List, Tuple, Optional, Callable
import math


def _make_rhs_fn(rhs_expr, state_sym: Symbol, param_vals: Dict):
    """Lambdify RHS for numerical integration. Single-state ODE."""
    subs_expr = rhs_expr.subs(param_vals)
    fn = lambdify(state_sym, subs_expr, modules=["math"])
    def rhs(x: float) -> float:
        try:
            return float(fn(x))
        except (OverflowError, ValueError, ZeroDivisionError):
            return math.inf
    return rhs


def rk4_step(rhs: Callable[[float], float], x: float, dt: float) -> float:
    k1 = rhs(x)
    k2 = rhs(x + 0.5 * dt * k1)
    k3 = rhs(x + 0.5 * dt * k2)
    k4 = rhs(x + dt * k3)
    return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def integrate(rhs: Callable[[float], float], x0: float,
              dt: float, n_steps: int) -> Tuple[List[float], bool]:
    """Integrate; return (trajectory, blew_up)."""
    xs = [x0]
    x = x0
    for _ in range(n_steps):
        x = rk4_step(rhs, x, dt)
        if math.isinf(x) or math.isnan(x) or abs(x) > 1e12:
            return xs, True
        xs.append(x)
    return xs, False


def check_fixed_points(rhs_expr, state_sym: Symbol,
                       expected_fp: List[float],
                       param_vals: Dict,
                       tol: float = 1e-6,
                       trace: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    HEURISTIC: verify that rhs ≈ 0 at each expected fixed point.
    """
    def log(m):
        if trace is not None:
            trace.append(m)

    rhs = _make_rhs_fn(rhs_expr, state_sym, param_vals)
    all_ok = True
    for fp in expected_fp:
        val = rhs(fp)
        ok = abs(val) < tol
        all_ok = all_ok and ok
        log(f"fixed point x={fp}: f(x)={val:.2e} "
            f"{'✓' if ok else '✗'} [HEURISTIC numerical]")
    msg = (f"FIXED POINTS {'PASS' if all_ok else 'FAIL'}: "
           f"{len(expected_fp)} checked [HEURISTIC]")
    log(msg)
    return all_ok, msg


def check_invariant(rhs_expr, state_sym: Symbol,
                    invariant_expr, param_vals: Dict,
                    x0_values: List[float],
                    dt: float = 0.01, n_steps: int = 1000,
                    tol: float = 1e-4,
                    trace: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    HEURISTIC: integrate trajectories and check that I(x(t)) stays constant.
    invariant_expr is a sympy expression in state_sym that should be conserved.
    """
    def log(m):
        if trace is not None:
            trace.append(m)

    rhs = _make_rhs_fn(rhs_expr, state_sym, param_vals)
    inv_fn = lambdify(state_sym, invariant_expr.subs(param_vals), modules=["math"])

    all_ok = True
    for x0 in x0_values:
        traj, blew_up = integrate(rhs, x0, dt, n_steps)
        if blew_up:
            log(f"invariant check x0={x0}: trajectory blew up [HEURISTIC]")
            all_ok = False
            continue
        inv_vals = [float(inv_fn(x)) for x in traj]
        drift = max(inv_vals) - min(inv_vals)
        scale = abs(inv_vals[0]) + 1e-12
        rel_drift = drift / scale
        ok = rel_drift < tol
        all_ok = all_ok and ok
        log(f"invariant x0={x0}: rel drift {rel_drift:.2e} "
            f"{'✓' if ok else '✗'} [HEURISTIC numerical, {n_steps} steps]")
    msg = (f"INVARIANT {'PASS' if all_ok else 'FAIL'}: "
           f"{len(x0_values)} trajectories [HEURISTIC]")
    log(msg)
    return all_ok, msg


def check_boundedness(rhs_expr, state_sym: Symbol,
                      param_vals: Dict,
                      x0_values: List[float],
                      dt: float = 0.01, n_steps: int = 2000,
                      trace: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    HEURISTIC: integrate from several ICs; fail if any blow up in finite time.
    (Only meaningful when the system is expected to be bounded —
    caller decides whether to include this check.)
    """
    def log(m):
        if trace is not None:
            trace.append(m)

    rhs = _make_rhs_fn(rhs_expr, state_sym, param_vals)
    all_ok = True
    for x0 in x0_values:
        _, blew_up = integrate(rhs, x0, dt, n_steps)
        ok = not blew_up
        all_ok = all_ok and ok
        log(f"boundedness x0={x0}: {'bounded ✓' if ok else 'BLEW UP ✗'} "
            f"[HEURISTIC numerical]")
    msg = (f"BOUNDEDNESS {'PASS' if all_ok else 'FAIL'} [HEURISTIC]")
    log(msg)
    return all_ok, msg
