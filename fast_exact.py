"""Exact (statevector) VQLS costs computed with dense linear algebra.

Mathematically identical to EstimatorVQLSGlobal / EstimatorVQLSLocal, but
without the SparsePauliOp operator products, which blow up whenever `b` is a
dense vector (generic state preparation -> B_LCU has 4^n terms).

The Estimator path is already an exact statevector simulation, so nothing
quantum is lost -- only the exponential bookkeeping. Verify with:

    python fast_exact.py --verify        # agree with vqls.py on the 2-qubit demo
    python fast_exact.py --kkt admm_kkt_n5.npz --index 0 --seeds 5
    python fast_exact.py --kkt admm_kkt_n5.npz --index 0 --seeds 10 --maxiter 1500 --d 8
    python fast_exact.py --kkt admm_kkt_n10.npz --index 0 --seeds 5 --maxiter 2000 --d 10
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import Operator
from qiskit.circuit.library import StatePreparation

from ansatz import HE_ansatz, fixed_ansatz


class FastExactVQLS:
    """Common machinery: |x> = V|0>, |psi> = A|x>, cost -> 0 at the solution."""

    def __init__(self, A, b, d, params0, ansatz_function, B=None, **_ignored):
        self.A = np.asarray(A, dtype=complex)
        self.b = np.asarray(b, dtype=complex)
        self.d = d
        self.params0 = np.asarray(params0, dtype=float)
        self.ansatz_function = ansatz_function
        self.n_qubits = int(np.log2(self.A.shape[0]))
        # B|0> = |b>. Needed by the local cost; built once, not per evaluation.
        self.B = (np.asarray(B, dtype=complex) if B is not None
                  else Operator(StatePreparation(self.b)).to_matrix())
        self.cost_history = []
        self._optimal_params = None
        # Precompute the j-th local operator B Z_j B^dag once.
        self._local_ops = []
        for j in range(self.n_qubits):
            z = np.ones(2**self.n_qubits)
            for k in range(2**self.n_qubits):
                # qiskit qubit ordering: qubit j is bit j of the index
                if (k >> j) & 1:
                    z[k] = -1.0
            self._local_ops.append(self.B @ np.diag(z) @ self.B.conj().T)

    def _psi(self, params):
        """|psi> = A V(params)|0>, unnormalized."""
        V = Operator(self.ansatz_function(self.n_qubits, self.d, params)).to_matrix()
        x = V[:, 0]                      # V|0> is the first column
        return self.A @ x, x

    def run(self, method="COBYLA", options=None):
        self.cost_history = []
        res = minimize(self.calculate_cost, self.params0,
                       method=method, options=options or {})
        self._optimal_params = res.x
        self._optimal_solution = res
        return res

    def get_optimal_statevector(self):
        _, x = self._psi(self._optimal_params)
        return x / np.linalg.norm(x)

    def get_optimal_params(self):
        return self._optimal_params


class FastExactGlobal(FastExactVQLS):
    """C_G = 1 - |<b|psi>|^2 / <psi|psi>."""

    def calculate_cost(self, params):
        psi, _ = self._psi(params)
        denom = np.real(np.vdot(psi, psi))
        num = abs(np.vdot(self.b, psi)) ** 2
        cost = float(np.real(1.0 - num / denom))
        self.cost_history.append(cost)
        return cost


class FastExactLocal(FastExactVQLS):
    """C_L = 1/2 - (1/2n) sum_j <psi| B Z_j B^dag |psi> / <psi|psi>."""

    def calculate_cost(self, params):
        psi, _ = self._psi(params)
        denom = np.real(np.vdot(psi, psi))
        num = sum(np.real(np.vdot(psi, op @ psi)) for op in self._local_ops)
        cost = float(np.real(0.5 - 0.5 * num / (self.n_qubits * denom)))
        self.cost_history.append(cost)
        return cost


# --------------------------------------------------------------------------

def verify():
    """Check the dense costs match vqls.py term for term on the 2-qubit demo."""
    import logging
    from vqls import EstimatorVQLSGlobal, EstimatorVQLSLocal
    logging.getLogger("vqls_logger").setLevel(logging.WARNING)

    np.random.seed(1)
    n, d = 2, 2
    M = np.random.randn(4, 4)
    A = M @ M.T + 4 * np.eye(4)
    b = np.array([1.0, 0.0, 0.0, 0.0])
    params = np.random.uniform(0, 2 * np.pi, 2 * n * d)

    print(f"{'cost':<8}{'vqls.py':>14}{'fast_exact':>14}{'abs diff':>12}")
    for name, slow_cls, fast_cls in (
        ("C_G", EstimatorVQLSGlobal, FastExactGlobal),
        ("C_L", EstimatorVQLSLocal, FastExactLocal),
    ):
        kw = dict(A=A, b=b, d=d, params0=params, ansatz_function=HE_ansatz)
        slow = slow_cls(**kw).calculate_cost(params)
        fast = fast_cls(**kw).calculate_cost(params)
        print(f"{name:<8}{slow:>14.10f}{fast:>14.10f}{abs(slow-fast):>12.2e}")


def kkt(fname, index, seeds, maxiter, d, ansatz_name):
    from sdp_operators import load_system
    ansatz_fn, per_layer = {"HE_ansatz": (HE_ansatz, 2),
                            "fixed_ansatz": (fixed_ansatz, 1)}[ansatz_name]
    s = load_system(fname, index=index)
    print(f"\n--- {s.describe()}")
    print(f"ansatz={ansatz_name} d={d} maxiter={maxiter} seeds={seeds}\n")
    print(f"{'cost':<6}{'residual median':>17}{'[min':>11}{'max]':>11}"
          f"{'fid median':>13}{'sec':>8}")

    kw = s.for_vqls()
    kw.pop("A_LCU", None)          # dense path does not need the decomposition
    x_exact = s.exact_solution()

    for label, cls in (("C_G", FastExactGlobal), ("C_L", FastExactLocal)):
        res_list, fid_list, t0 = [], [], time.time()
        for k in range(seeds):
            rng = np.random.default_rng(1 + k)
            p0 = rng.uniform(0, 2 * np.pi, per_layer * s.n_qubits * d)
            solver = cls(**kw, d=d, params0=p0, ansatz_function=ansatz_fn)
            solver.run(method="COBYLA", options={"maxiter": maxiter})
            x_hat = solver.get_optimal_statevector()
            fid_list.append(float(abs(np.vdot(x_exact, x_hat)) ** 2))
            _, x_scaled = s.recover_scale(x_hat)
            res_list.append(s.residual(x_scaled))
        r = np.array(res_list)
        print(f"{label:<6}{np.median(r):>17.2e}{r.min():>11.2e}{r.max():>11.2e}"
              f"{np.median(fid_list):>13.4f}{time.time()-t0:>8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--kkt", metavar="FILE")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--ansatz", default="HE_ansatz")
    a = ap.parse_args()
    if a.verify:
        verify()
    if a.kkt:
        kkt(a.kkt, a.index, a.seeds, a.maxiter, a.d, a.ansatz)


if __name__ == "__main__":
    main()
