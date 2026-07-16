"""Run VQLS on a linear system taken from a Max-Cut SDP solver.

Companion to example.py, but instead of a random Hermitian matrix it uses the
real operators that an interior point method / ADMM produce while solving the
basic SDP relaxation of Max-Cut. See SDP_OPERATORS.md for where they come from.

Run:  python sdp_example.py
      python sdp_example.py --file ipm_schur_n16.npz --index 2
      python sdp_example.py --file admm_kkt_n5.npz --index 0 --maxiter 400
      python sdp_example.py --list
"""

import argparse
import logging

import numpy as np

from ansatz import HE_ansatz
from sdp_operators import available, load_system, load_systems
from vqls import EstimatorVQLSLocal

logging.getLogger("vqls_logger").setLevel(logging.WARNING)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", default="ipm_schur_n4.npz",
                    help="resource file in data/ (default: the 2-qubit IPM system)")
    ap.add_argument("--index", type=int, default=2,
                    help="which system in the archive (default 2: first non-diagonal Schur matrix)")
    ap.add_argument("--d", type=int, default=2, help="ansatz layers")
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--list", action="store_true", help="list available systems and exit")
    args = ap.parse_args()

    if args.list:
        for f in available():
            print(f"\n{f.name}")
            for i, s in enumerate(load_systems(f)):
                print(f"  [{i:>2}] {s.describe()}")
        return

    np.random.seed(args.seed)

    # --- load the SDP operator -------------------------------------------
    s = load_system(args.file, index=args.index)
    print(f"system : {s.describe()}")
    print(f"source : {s.metadata}")
    print()

    n = s.n_qubits
    params0 = np.random.uniform(0, 2 * np.pi, 2 * n * args.d)  # HE_ansatz: 2*n*d

    # for_vqls() hands over A, the NORMALIZED b, and the precomputed Pauli LCU
    solver = EstimatorVQLSLocal(
        **s.for_vqls(),
        d=args.d,
        params0=params0,
        ansatz_function=HE_ansatz,
    )
    solver.run(method="COBYLA", options={"maxiter": args.maxiter})

    # --- compare against the exact classical answer ----------------------
    x_vqls = solver.get_optimal_statevector()
    x_exact = s.exact_solution(normalized=True)
    fidelity = abs(np.vdot(x_exact, x_vqls)) ** 2

    # VQLS returns only the DIRECTION of x; re-fit the scale classically.
    alpha, x_scaled = s.recover_scale(x_vqls)

    print(f"final cost      : {solver.cost_history[-1]:.3e}")
    print(f"cost evaluations: {len(solver.cost_history)}")
    print(f"fidelity        : {fidelity:.8f}   (1.0 = perfect match)")
    print(f"cond(A)         : {s.condition_number:.4g}")
    print()
    print(f"exact  x (norm) : {np.round(np.abs(x_exact), 3)}  (magnitudes)")
    print(f"VQLS   x (norm) : {np.round(np.abs(x_vqls), 3)}  (magnitudes)")
    print()
    print(f"recovered scale : alpha = {alpha.real:+.4f}")
    print(f"relative residual ||A x - b|| / ||b|| : {s.residual(x_scaled):.4e}")
    print(f"  (classical reference               : {s.residual(np.linalg.solve(s.A, s.b_raw)):.4e})")


if __name__ == "__main__":
    main()
