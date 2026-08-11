"""
VQLS scaling benchmark.

Sweeps qubit count n and reports, for each solver flavour:
  - L, the number of Pauli terms in the LCU of A
  - wall-clock seconds per cost evaluation
  - final cost and fidelity vs numpy.linalg.solve

Two families of A are compared:
  dense  : A = M M^T + c*I   (random dense -> L grows like 4^n)
  pauli  : A = I + a*X_0 + b*Z_0 Z_1  (fixed L = 3, independent of n)

Usage:
    python benchmark_scaling.py --nmax 4 --maxiter 100
    python benchmark_scaling.py --family pauli --nmax 6
    python benchmark_scaling.py --solver sampler --nmax 3 --shots 4096
"""
import argparse
import logging
import time

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator

from ansatz import HE_ansatz, fixed_ansatz
from vqls import (
    EstimatorVQLSLocal,
    EstimatorVQLSGlobal,
    SamplerVQLSLocal,
    SamplerVQLSGlobal,
)

# vqls.py sets its logger to INFO at import time, so silence it AFTER importing.
logging.getLogger("vqls_logger").setLevel(logging.WARNING)

SOLVERS = {
    "estimator-local": EstimatorVQLSLocal,
    "estimator-global": EstimatorVQLSGlobal,
    "sampler-local": SamplerVQLSLocal,
    "sampler-global": SamplerVQLSGlobal,
}


def make_A(n, family, seed=1):
    """Return a Hermitian A of size 2^n, plus its condition number."""
    N = 2**n
    rng = np.random.default_rng(seed)
    if family == "dense":
        M = rng.standard_normal((N, N))
        A = M @ M.T + 4 * np.eye(N)
    elif family == "pauli":
        # Sparse LCU: 3 Pauli terms regardless of n. This is the form used in
        # the VQLS paper, and the only form that stays affordable as n grows.
        terms = ["I" * n]
        coeffs = [1.0]
        if n >= 1:
            terms.append("I" * (n - 1) + "X")
            coeffs.append(0.25)
        if n >= 2:
            terms.append("I" * (n - 2) + "ZZ")
            coeffs.append(0.25)
        A = SparsePauliOp(terms, coeffs).to_matrix().real
    else:
        raise ValueError(family)
    return A, np.linalg.cond(A)


def run_one(n, family, solver_name, maxiter, d, shots, seed=1):
    A, cond = make_A(n, family, seed)
    N = 2**n
    b = np.zeros(N)
    b[0] = 1.0

    L = len(SparsePauliOp.from_operator(Operator(A)))

    rng = np.random.default_rng(seed)
    params0 = rng.uniform(0, 2 * np.pi, 2 * n * d)  # HE_ansatz: 2*n*d

    cls = SOLVERS[solver_name]
    kwargs = dict(A=A, b=b, d=d, params0=params0, ansatz_function=HE_ansatz)
    if solver_name.startswith("sampler"):
        kwargs["num_shots"] = shots

    solver = cls(**kwargs)
    t0 = time.perf_counter()
    solver.run(method="COBYLA", options={"maxiter": maxiter})
    elapsed = time.perf_counter() - t0

    x_vqls = solver.get_optimal_statevector()
    x_exact = np.linalg.solve(A, b)
    x_exact /= np.linalg.norm(x_exact)
    fid = abs(np.vdot(x_exact, x_vqls)) ** 2

    n_evals = len(solver.cost_history)
    return dict(
        n=n, dim=N, L=L, cond=cond, evals=n_evals, secs=elapsed,
        per_eval=elapsed / max(n_evals, 1),
        cost=solver.cost_history[-1], fid=fid,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nmax", type=int, default=4)
    p.add_argument("--nmin", type=int, default=2)
    p.add_argument("--family", choices=["dense", "pauli"], default="dense")
    p.add_argument("--solver", choices=list(SOLVERS), default="estimator-local")
    p.add_argument("--maxiter", type=int, default=100)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--shots", type=int, default=4096)
    args = p.parse_args()

    print(f"solver={args.solver}  family={args.family}  "
          f"maxiter={args.maxiter}  d={args.depth}"
          + (f"  shots={args.shots}" if args.solver.startswith("sampler") else ""))
    print(f"{'n':>2} {'dim':>5} {'L':>5} {'cond':>8} {'evals':>6} "
          f"{'total_s':>9} {'s/eval':>8} {'cost':>9} {'fidelity':>9}")

    for n in range(args.nmin, args.nmax + 1):
        try:
            r = run_one(n, args.family, args.solver,
                        args.maxiter, args.depth, args.shots)
        except KeyboardInterrupt:
            print(f"{n:>2}  interrupted -- this is where it stops being affordable")
            break
        print(f"{r['n']:>2} {r['dim']:>5} {r['L']:>5} {r['cond']:>8.1f} "
              f"{r['evals']:>6} {r['secs']:>9.1f} {r['per_eval']:>8.3f} "
              f"{r['cost']:>9.5f} {r['fid']:>9.4f}")


if __name__ == "__main__":
    main()
