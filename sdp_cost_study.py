"""Compare VQLS cost functions on linear systems from a Max-Cut SDP solver.

This is the study script: it drives the solvers in `vqls.py` over the operators
in `data/` and reports how each cost function performs. `sdp_example.py` is a
single-system smoke test; this one answers "global vs local, and how does that
change along the optimiser's path".

    python sdp_cost_study.py                              # global vs local, one system
    python sdp_cost_study.py --solvers all                # + the two Sampler variants
    python sdp_cost_study.py --sweep ipm_schur_n4.npz     # every IPM iteration
    python sdp_cost_study.py --sweep ipm_schur_n8.npz --csv results.csv

Reading the output
------------------
`final cost` is NOT comparable across rows. C_G and C_L are different objective
functions:

    C_G = 1 - |<b|psi>|^2 / <psi|psi>
    C_L = 1/2 - (1/2n) sum_j <psi| B (Z_j (x) I) B^dag |psi> / <psi|psi>

At identical parameters they give different numbers (e.g. 0.975 vs 0.646 on a
2-qubit system). Both reach 0 at the solution, but they descend different
landscapes, so a smaller `final cost` does not mean a better answer.

The columns that ARE comparable, because they measure the recovered solution
rather than the objective:

    fidelity  = |<x_exact|x_vqls>|^2   -- quadratic near the optimum, so it
                                          saturates at 1.0000 and flatters VQLS
    residual  = ||A x - b|| / ||b||    -- linear in the state error; use this

Report the residual. See SDP_OPERATORS / README section 8.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from ansatz import HE_ansatz, fixed_ansatz
from sdp_operators import load_system, load_systems
from vqls import (
    EstimatorVQLSGlobal,
    EstimatorVQLSLocal,
    SamplerVQLSGlobal,
    SamplerVQLSLocal,
)

# set AFTER importing vqls -- vqls.py sets its own logger to INFO at import time
logging.getLogger("vqls_logger").setLevel(logging.WARNING)

SOLVERS = {
    "estimator-global": (EstimatorVQLSGlobal, "C_G", "exact"),
    "estimator-local": (EstimatorVQLSLocal, "C_L", "exact"),
    "sampler-global": (SamplerVQLSGlobal, "C_G", "shots"),
    "sampler-local": (SamplerVQLSLocal, "C_L", "shots"),
}
ANSATZE = {"HE_ansatz": (HE_ansatz, 2), "fixed_ansatz": (fixed_ansatz, 1)}


def run_one(system, solver_name, *, d, maxiter, seed, num_shots, ansatz_name):
    """Run a single (system, solver) pair and return a row of metrics."""
    cls, cost_symbol, evaluation = SOLVERS[solver_name]
    ansatz_fn, params_per_layer = ANSATZE[ansatz_name]

    np.random.seed(seed)
    n = system.n_qubits
    params0 = np.random.uniform(0, 2 * np.pi, params_per_layer * n * d)

    kwargs = dict(system.for_vqls(), d=d, params0=params0, ansatz_function=ansatz_fn)
    if evaluation == "shots":
        kwargs["num_shots"] = num_shots

    t0 = time.time()
    try:
        solver = cls(**kwargs)
        solver.run(method="COBYLA", options={"maxiter": maxiter})
        elapsed = time.time() - t0

        x_hat = solver.get_optimal_statevector()
        fidelity = float(abs(np.vdot(system.exact_solution(), x_hat)) ** 2)
        _, x_scaled = system.recover_scale(x_hat)
        residual = system.residual(x_scaled)
        status = "ok"
        final_cost = float(solver.cost_history[-1])
        n_evals = len(solver.cost_history)
    except Exception as exc:  # a solver failing should not kill the sweep
        elapsed = time.time() - t0
        final_cost = fidelity = residual = float("nan")
        n_evals = 0
        status = f"FAILED: {type(exc).__name__}: {exc}"[:60]

    return {
        "solver": solver_name,
        "cost": cost_symbol,
        "evaluation": evaluation,
        "n_qubits": system.n_qubits,
        "pauli_terms": len(system.pauli_labels),
        "cond": system.condition_number,
        "final_cost": final_cost,
        "evals": n_evals,
        "fidelity": fidelity,
        "residual": residual,
        "seconds": elapsed,
        "status": status,
        **{k: v for k, v in system.metadata.items()
           if k in {"source", "n", "ipm_iteration", "admm_iteration", "system"}},
    }


HEADER = (f"{'solver':<17}{'cost':<5}{'cond':>9}{'terms':>7}"
          f"{'final_cost':>12}{'evals':>7}{'fidelity':>11}{'residual':>11}{'sec':>8}")


HEADER_MULTI = (f"{'solver':<17}{'cost':<5}{'cond':>9}{'terms':>7}"
                f"{'residual median':>17}{'[min':>11}{'max]':>11}{'sec':>8}")


def print_row(r):
    if r["status"] != "ok":
        print(f"{r['solver']:<17}{r['cost']:<5}{'':>9}{'':>7}  {r['status']}")
        return
    print(f"{r['solver']:<17}{r['cost']:<5}{r['cond']:>9.3g}{r['pauli_terms']:>7}"
          f"{r['final_cost']:>12.3e}{r['evals']:>7}{r['fidelity']:>11.6f}"
          f"{r['residual']:>11.2e}{r['seconds']:>8.1f}")


def print_row_multi(reps):
    """Aggregate several seeds of the same (system, solver) pair."""
    ok = [r for r in reps if r["status"] == "ok"]
    if not ok:
        print(f"{reps[0]['solver']:<17}{reps[0]['cost']:<5}  all seeds failed")
        return
    res = np.array([r["residual"] for r in ok])
    r0 = ok[0]
    print(f"{r0['solver']:<17}{r0['cost']:<5}{r0['cond']:>9.3g}{r0['pauli_terms']:>7}"
          f"{np.median(res):>17.2e}{res.min():>11.2e}{res.max():>11.2e}"
          f"{sum(r['seconds'] for r in ok):>8.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="ipm_schur_n4.npz")
    ap.add_argument("--index", type=int, default=2,
                    help="system index (default 2 = first non-diagonal Schur matrix)")
    ap.add_argument("--sweep", metavar="FILE", default=None,
                    help="run every predictor system in FILE instead of a single index")
    ap.add_argument("--solvers", nargs="+", default=["estimator-global", "estimator-local"],
                    choices=list(SOLVERS) + ["all"])
    ap.add_argument("--ansatz", default="HE_ansatz", choices=list(ANSATZE))
    ap.add_argument("--d", type=int, default=2, help="ansatz layers")
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--num-shots", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=1, metavar="N",
                    help="repeat each run with N different initial-parameter seeds "
                         "and report the median + spread. A single seed CANNOT "
                         "distinguish C_G from C_L -- the difference at small n is "
                         "COBYLA luck, not structure. Use >= 5 for any real claim.")
    ap.add_argument("--csv", metavar="PATH", default=None)
    args = ap.parse_args()

    solvers = list(SOLVERS) if "all" in args.solvers else args.solvers

    if args.sweep:
        systems = [s for s in load_systems(args.sweep)
                   if s.metadata.get("system", "predictor") == "predictor"]
        title = f"sweep: {args.sweep} (predictor systems)"
    else:
        systems = [load_system(args.file, index=args.index)]
        title = f"{args.file} [index {args.index}]"

    print(f"\n{title}")
    print(f"ansatz={args.ansatz}  d={args.d}  maxiter={args.maxiter}  seed={args.seed}")
    print("\nfinal_cost is NOT comparable across cost functions (C_G vs C_L are")
    print("different objectives). Compare the residual column.\n")

    if args.seeds > 1:
        print(f"averaging over {args.seeds} seeds; showing median [min, max] residual\n")

    rows = []
    for s in systems:
        print(f"--- {s.describe()}")
        print(HEADER if args.seeds == 1 else HEADER_MULTI)
        for name in solvers:
            reps = [
                run_one(s, name, d=args.d, maxiter=args.maxiter,
                        seed=args.seed + k, num_shots=args.num_shots,
                        ansatz_name=args.ansatz)
                for k in range(args.seeds)
            ]
            rows.extend(reps)
            if args.seeds == 1:
                print_row(reps[0])
            else:
                print_row_multi(reps)
        print()

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
