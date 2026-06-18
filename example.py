"""
Minimal end-to-end VQLS example.

Solves a 2-qubit (4x4) Hermitian linear system A x = b with the local-cost
Estimator solver, then compares the recovered solution against the exact
classical answer from numpy.

Run:  python example.py
"""
import logging
import numpy as np
from qiskit.quantum_info import Statevector

from ansatz import HE_ansatz
from vqls import EstimatorVQLSLocal

# Silence the per-iteration cost print; raise to logging.INFO to watch convergence.
logging.getLogger("vqls_logger").setLevel(logging.WARNING)


def main():
    np.random.seed(1)

    # --- define a well-conditioned 2-qubit Hermitian system ---
    n = 2
    M = np.random.randn(4, 4)
    A = M @ M.T + 4 * np.eye(4)          # symmetric positive-definite (Hermitian)
    b = np.array([1.0, 0.0, 0.0, 0.0])   # |00>, already normalized

    # --- ansatz / parameter setup ---
    d = 2                                # number of ansatz layers
    num_params = 2 * n * d               # HE_ansatz needs 2 * n * d parameters
    params0 = np.random.uniform(0, 2 * np.pi, num_params)

    # --- build and run the solver ---
    solver = EstimatorVQLSLocal(
        A=A, b=b, d=d,
        params0=params0,
        ansatz_function=HE_ansatz,
    )
    solver.run(method="COBYLA", options={"maxiter": 200})

    # --- recover the solution and compare to the exact answer ---
    x_vqls = solver.get_optimal_statevector()
    x_exact = np.linalg.solve(A, b)
    x_exact = x_exact / np.linalg.norm(x_exact)

    fidelity = abs(np.vdot(x_exact, x_vqls)) ** 2

    print(f"final cost      : {solver.cost_history[-1]:.5f}")
    print(f"cost evaluations: {len(solver.cost_history)}")
    print(f"fidelity        : {fidelity:.4f}   (1.0 = perfect match)")
    print(f"exact   x       : {np.round(x_exact, 3)}")
    print(f"VQLS    x       : {np.round(np.abs(x_vqls), 3)}  (magnitudes)")


if __name__ == "__main__":
    main()
