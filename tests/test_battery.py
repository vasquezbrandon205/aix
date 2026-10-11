"""
AIX v3 — Phase 1 Test Battery
10 known physics equations (must PASS) + 10 broken variants (must FAIL).

Each test specifies the equation, dimensions, and expected properties.
Run: python3 -m aix.test_battery  (from aix-v3 directory)
"""
import sys
sys.path.insert(0, '.')
from sympy import Symbol, exp, sin, cos, tanh, sqrt, pi
from aix.sieve import verify

# Shared symbols
x = Symbol('x')       # position [L]
v = Symbol('v')       # velocity [L/T]
t = Symbol('t')       # time [T]
m = Symbol('m')       # mass [M]
k = Symbol('k')       # spring constant [M/T^2]
c = Symbol('c')       # damping [M/T]
g = Symbol('g')       # gravity [L/T^2]
L = Symbol('L')       # pendulum length [L]
omega = Symbol('omega')  # angular freq [1/T]
gamma = Symbol('gamma')  # decay rate [1/T]
T_var = Symbol('T')   # temperature [K] — avoid clash with time dim key
alpha = Symbol('alpha')  # generic rate [1/T]

DIMS = {
    x: {"L": 1}, v: {"L": 1, "T": -1}, t: {"T": 1}, m: {"M": 1},
    k: {"M": 1, "T": -2}, c: {"M": 1, "T": -1}, g: {"L": 1, "T": -2},
    L: {"L": 1}, omega: {"T": -1}, gamma: {"T": -1},
    T_var: {"K": 1}, alpha: {"T": -1},
}

# LHS symbols for each equation type
dxdt = Symbol('dxdt')   # [L/T]
dvdt = Symbol('dvdt')   # [L/T^2]
dTdt = Symbol('dTdt')   # [K/T]


def make_dims(extra: dict) -> dict:
    d = dict(DIMS)
    d.update(extra)
    return d


# ============================================================
# 10 PASS CASES — real physics, dimensionally consistent
# ============================================================
PASS_CASES = [
    {
        "name": "P1: Exponential decay (Newton cooling / RC discharge)",
        "spec": {
            "lhs": dTdt, "rhs": -alpha * T_var,
            "var_dims": make_dims({dTdt: {"K": 1, "T": -1}}),
            "state_sym": T_var, "param_vals": {alpha: 0.5},
            "fixed_points": [0.0],
            "check_boundedness": True, "ics": [10.0, 100.0],
            # Note: no limit check on RHS — dT/dt → -inf as T→∞ is CORRECT
            # (restoring force grows without bound; that's stability, not a bug)
        },
    },
    {
        "name": "P2: Harmonic oscillator (undamped)",
        "spec": {
            "lhs": dvdt, "rhs": -(k / m) * x,
            "var_dims": make_dims({dvdt: {"L": 1, "T": -2}}),
        },
    },
    {
        "name": "P3: Damped harmonic oscillator",
        "spec": {
            "lhs": dvdt, "rhs": -(k / m) * x - (c / m) * v,
            "var_dims": make_dims({dvdt: {"L": 1, "T": -2}}),
        },
    },
    {
        "name": "P4: Pendulum (small angle)",
        "spec": {
            "lhs": dvdt, "rhs": -(g / L) * x,
            "var_dims": make_dims({dvdt: {"L": 1, "T": -2}}),
        },
    },
    {
        "name": "P5: Logistic growth",
        "spec": {
            "lhs": dxdt, "rhs": alpha * x * (1 - x / Symbol('K_cap')),
            "var_dims": make_dims({
                dxdt: {"L": 1, "T": -1},
                Symbol('K_cap'): {"L": 1},
            }),
            "state_sym": x, "param_vals": {alpha: 0.5, Symbol('K_cap'): 10.0},
            "fixed_points": [0.0, 10.0],
            "check_boundedness": True, "ics": [1.0, 5.0],
        },
    },
    {
        "name": "P6: Stokes drag (linear)",
        "spec": {
            "lhs": dvdt, "rhs": -(c / m) * v,
            "var_dims": make_dims({dvdt: {"L": 1, "T": -2}}),
            "state_sym": v, "param_vals": {c: 0.3, m: 1.0},
            "fixed_points": [0.0],
            "check_boundedness": True, "ics": [5.0],
        },
    },
    {
        "name": "P7: Saturating dynamics (tanh)",
        "spec": {
            "lhs": dxdt, "rhs": Symbol('v_sat') * tanh(x / Symbol('x0')),
            "var_dims": make_dims({
                dxdt: {"L": 1, "T": -1},
                Symbol('v_sat'): {"L": 1, "T": -1},
                Symbol('x0'): {"L": 1},
            }),
            "state_sym": x,
            "param_vals": {Symbol('v_sat'): 1.0, Symbol('x0'): 2.0},
            "fixed_points": [0.0],
            "limits": {"x": {"oo": "bounded", "-oo": "bounded"}},
        },
    },
    {
        "name": "P8: Free fall with gravity",
        "spec": {
            "lhs": dvdt, "rhs": -g,
            "var_dims": make_dims({dvdt: {"L": 1, "T": -2}}),
        },
    },
    {
        "name": "P9: Driven decay (exponential + constant drive)",
        "spec": {
            "lhs": dTdt, "rhs": -alpha * T_var + alpha * Symbol('T_env'),
            "var_dims": make_dims({
                dTdt: {"K": 1, "T": -1},
                Symbol('T_env'): {"K": 1},
            }),
            "state_sym": T_var,
            "param_vals": {alpha: 0.5, Symbol('T_env'): 20.0},
            "fixed_points": [20.0],
            "check_boundedness": True, "ics": [0.0, 100.0],
        },
    },
    {
        "name": "P10: Quadratic drag",
        "spec": {
            "lhs": dvdt, "rhs": -Symbol('Cd') * v * abs(v) / m,
            "var_dims": make_dims({
                dvdt: {"L": 1, "T": -2},
                Symbol('Cd'): {"M": 1, "L": -1},
            }),
        },
    },
]

# ============================================================
# 10 FAIL CASES — deliberately broken (dimensional or physical)
# ============================================================
FAIL_CASES = [
    {
        "name": "F1: Adding length to velocity (dim mismatch)",
        "spec": {
            "lhs": dvdt, "rhs": -(k / m) * x + v * t,  # v*t = [L], not [L/T^2]
            "var_dims": make_dims({dvdt: {"L": 1, "T": -2}}),
        },
        "expect_fail_on": "dimensional_consistency",
    },
    {
        "name": "F2: sin of dimensioned quantity",
        "spec": {
            "lhs": dxdt, "rhs": alpha * sin(x),  # sin([L]) illegal
            "var_dims": make_dims({dxdt: {"L": 1, "T": -1}}),
        },
        "expect_fail_on": "dimensional_consistency",
    },
    {
        "name": "F3: exp of dimensioned quantity",
        "spec": {
            "lhs": dxdt, "rhs": alpha * exp(-t),  # exp([T]) illegal
            "var_dims": make_dims({dxdt: {"L": 1, "T": -1}}),
        },
        "expect_fail_on": "dimensional_consistency",
    },
    {
        "name": "F4: Wrong LHS/RHS dimensions",
        "spec": {
            "lhs": dxdt, "rhs": -g,  # [L/T] vs [L/T^2]
            "var_dims": make_dims({dxdt: {"L": 1, "T": -1}}),
        },
        "expect_fail_on": "dimensional_consistency",
    },
    {
        "name": "F5: tanh of dimensioned quantity",
        "spec": {
            "lhs": dxdt, "rhs": alpha * tanh(v),  # tanh([L/T]) illegal
            "var_dims": make_dims({dxdt: {"L": 1, "T": -1}}),
        },
        "expect_fail_on": "dimensional_consistency",
    },
    {
        "name": "F6: Unphysical blowup (cubic growth, expected bounded)",
        "spec": {
            "lhs": dxdt, "rhs": alpha * x**3,
            "var_dims": make_dims({
                dxdt: {"L": 1, "T": -1},
                alpha: {"L": -2, "T": -1},
            }),
            "state_sym": x, "param_vals": {alpha: 0.1},
            "check_boundedness": True, "ics": [1.0],
        },
        "expect_fail_on": "boundedness",
    },
    {
        "name": "F7: Wrong fixed point (claims equilibrium at wrong location)",
        "spec": {
            "lhs": dTdt, "rhs": -alpha * (T_var - Symbol('T_ref')),
            "var_dims": make_dims({
                dTdt: {"K": 1, "T": -1},
                Symbol('T_ref'): {"K": 1},
            }),
            "state_sym": T_var,
            "param_vals": {alpha: 0.5, Symbol('T_ref'): 20.0},
            "fixed_points": [0.0],  # actual FP is at T_ref=20, not 0
        },
        "expect_fail_on": "fixed_points",
    },
    {
        "name": "F8: log of dimensioned quantity",
        "spec": {
            "lhs": dxdt, "rhs": alpha * Symbol('x0') * __import__('sympy').log(x),
            "var_dims": make_dims({
                dxdt: {"L": 1, "T": -1},
                Symbol('x0'): {"L": 1},
            }),
        },
        "expect_fail_on": "dimensional_consistency",
    },
    {
        "name": "F9: Adding mass to length",
        "spec": {
            "lhs": dxdt, "rhs": alpha * (x + m),  # [L] + [M] illegal
            "var_dims": make_dims({dxdt: {"L": 1, "T": -1}}),
        },
        "expect_fail_on": "dimensional_consistency",
    },
    {
        "name": "F10: Limit violation (claimed →0 but diverges)",
        "spec": {
            "lhs": dxdt, "rhs": Symbol('v0') * exp(x / Symbol('x0')),
            "var_dims": make_dims({
                dxdt: {"L": 1, "T": -1},
                Symbol('v0'): {"L": 1, "T": -1},
                Symbol('x0'): {"L": 1},
            }),
            "param_vals": {Symbol('v0'): 1.0, Symbol('x0'): 2.0},
            "limits": {"x": {"oo": "=0"}},  # actually → +inf
        },
        "expect_fail_on": "limit",
    },
]


def run_battery():
    print("\n" + "="*70)
    print("  AIX v3 PHASE 1 — FORMAL PROPERTY SIEVE TEST BATTERY")
    print("="*70)

    results = {"pass_cases": [], "fail_cases": []}

    print("\n--- PASS CASES (real physics, must verify) ---")
    for case in PASS_CASES:
        v = verify(case["spec"])
        ok = v["overall"] == "PASS"
        results["pass_cases"].append(ok)
        print(f"  [{'✓' if ok else '✗ FAIL'}] {case['name']}")
        if not ok:
            for name, c in v["checks"].items():
                if c["status"] == "FAIL":
                    print(f"      → {name}: {c['message']}")

    print("\n--- FAIL CASES (broken physics, must reject) ---")
    for case in FAIL_CASES:
        v = verify(case["spec"])
        # For fail cases: overall should be FAIL (correctly rejected)
        correctly_rejected = v["overall"] == "FAIL"
        results["fail_cases"].append(correctly_rejected)
        target = case.get("expect_fail_on", "?")
        print(f"  [{'✓' if correctly_rejected else '✗ MISS'}] {case['name']}")
        if not correctly_rejected:
            print(f"      → expected rejection on '{target}' but PASSED")
        else:
            failed_checks = [n for n, c in v["checks"].items()
                             if c["status"] == "FAIL"]
            print(f"      → correctly rejected via: {failed_checks}")

    np_pass = sum(results["pass_cases"])
    np_fail = sum(results["fail_cases"])
    print(f"\n{'='*70}")
    print(f"  PASS cases: {np_pass}/{len(PASS_CASES)} verified")
    print(f"  FAIL cases: {np_fail}/{len(FAIL_CASES)} correctly rejected")
    print(f"{'='*70}")

    all_good = np_pass == len(PASS_CASES) and np_fail == len(FAIL_CASES)
    print(f"\n  BATTERY: {'✓ ALL PASS' if all_good else '✗ FAILURES PRESENT'}\n")
    return all_good


if __name__ == "__main__":
    success = run_battery()
    sys.exit(0 if success else 1)
