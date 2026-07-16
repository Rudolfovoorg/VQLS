"""Load SDP linear systems (A x = b) and adapt them to this VQLS solver.

The systems in ``data/`` come from the Max-Cut SDP relaxation solvers in
https://github.com/Rudolfovoorg/Quantum-resource-analysis :

  * ``ipm_schur_n*.npz``  -- the Schur matrix M = Z^-1 o X from each Newton step
    of the primal-dual interior point method. SPD, graph-dependent.
  * ``admm_kkt_n*.npz``   -- the KKT matrix from the ADMM x-update. Symmetric but
    INDEFINITE, and identical for every graph (see SDP_OPERATORS.md).

Why an adapter is needed
------------------------
``BaseVQLS`` requires:

  1. ``A`` Hermitian of size 2^n x 2^n  -- the stored matrices are already padded
     to a power of two, and are real symmetric.
  2. ``b`` of length 2^n and **normalized** -- ``BaseVQLS`` feeds ``b`` straight
     into ``qiskit.circuit.library.StatePreparation``, which rejects a vector
     that is not unit norm. The stored RHS vectors are the raw ones the solver
     actually used, so they must be normalized here.
  3. The solver returns only the **normalized** ``|x>``. The scale is not
     recoverable from the quantum state alone, so ``recover_scale`` re-fits it
     classically -- see the method docstring.

This module has no dependency on the analysis repository; the .npz files are
self-contained.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp

_DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(slots=True)
class SDPSystem:
    """One linear system A x = b, ready for a VQLS solver."""

    A: np.ndarray                     # (2^n, 2^n) real symmetric
    b_raw: np.ndarray                 # (2^n,) unnormalized RHS as the solver used it
    n_qubits: int
    original_dimension: int
    pauli_labels: list[str] = field(default_factory=list)
    pauli_coeffs: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def b_norm(self) -> float:
        """||b|| -- needed to undo the normalization after solving."""
        return float(np.linalg.norm(self.b_raw))

    @property
    def b(self) -> np.ndarray:
        """Normalized RHS, as StatePreparation requires."""
        nrm = self.b_norm
        if nrm == 0:
            raise ValueError("RHS is the zero vector; cannot normalize for VQLS.")
        return self.b_raw / nrm

    @property
    def A_LCU(self) -> SparsePauliOp | None:
        """Precomputed Pauli decomposition, or None if it was not stored.

        Passing this to the solver as ``A_LCU=`` skips ``qc_utils.get_LCU(A)``,
        which rebuilds the same decomposition on every construction.
        """
        if self.pauli_coeffs is None or not self.pauli_labels:
            return None
        return SparsePauliOp(self.pauli_labels, self.pauli_coeffs)

    @property
    def condition_number(self) -> float:
        return float(np.linalg.cond(self.A))

    def is_spd(self, tol: float = 0.0) -> bool:
        return bool(np.linalg.eigvalsh(self.A)[0] > tol)

    def exact_solution(self, normalized: bool = True) -> np.ndarray:
        """Classical reference solution, for validating a VQLS run."""
        x = np.linalg.solve(self.A, self.b_raw)
        if normalized:
            nrm = np.linalg.norm(x)
            return x / nrm if nrm else x
        return x

    def for_vqls(self) -> dict:
        """Kwargs to splat into a solver constructor.

        >>> s = load_system("data/ipm_schur_n16.npz", index=2)
        >>> solver = EstimatorVQLSLocal(**s.for_vqls(), d=2, params0=p0,
        ...                             ansatz_function=HE_ansatz)
        """
        kwargs = {"A": self.A, "b": self.b}
        if self.A_LCU is not None:
            kwargs["A_LCU"] = self.A_LCU
        return kwargs

    def recover_scale(self, x_hat: np.ndarray) -> tuple[complex, np.ndarray]:
        """Turn the normalized VQLS state back into a solution of A x = b.

        VQLS returns a unit-norm ``|x>`` defined only up to a global phase, so
        the magnitude of the true solution is lost. Least-squares re-fit of the
        single scalar alpha in ``A (alpha x_hat) ~ b``:

            alpha = <A x_hat, b> / ||A x_hat||^2

        Returns ``(alpha, alpha * x_hat)``. This is a classical O(N^2)
        matrix-vector product, not part of the quantum routine -- it is the
        standard caveat that VQLS solves for the *direction* of x.
        """
        x_hat = np.asarray(x_hat).reshape(-1)
        Ax = self.A @ x_hat
        denom = float(np.vdot(Ax, Ax).real)
        if denom == 0:
            raise ValueError("A x_hat is zero; cannot fit a scale.")
        alpha = complex(np.vdot(Ax, self.b_raw) / denom)
        return alpha, alpha * x_hat

    def residual(self, x: np.ndarray) -> float:
        """Relative residual ||A x - b|| / ||b|| for a (scaled) solution."""
        return float(
            np.linalg.norm(self.A @ x - self.b_raw) / np.linalg.norm(self.b_raw)
        )

    def describe(self) -> str:
        src = self.metadata.get("source", "?")
        bits = [f"{src}", f"{self.n_qubits}q", f"dim {self.original_dimension}->{self.A.shape[0]}"]
        if "ipm_iteration" in self.metadata:
            bits.append(f"iter {self.metadata['ipm_iteration']} {self.metadata.get('system','')}")
        if "admm_iteration" in self.metadata:
            bits.append(f"iter {self.metadata['admm_iteration']}")
        bits.append(f"cond {self.condition_number:.3g}")
        bits.append(f"{len(self.pauli_labels)} Pauli terms")
        bits.append("SPD" if self.is_spd() else "indefinite")
        return "  ".join(bits)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    candidate = _DATA_DIR / p.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No such system file: {path}")


def load_systems(path: str | Path) -> list[SDPSystem]:
    """Load every system stored in a .npz archive."""
    data = np.load(_resolve(path), allow_pickle=True)
    meta = json.loads(str(data["metadata"]))
    out = []
    for m in meta:
        i = m["index"]
        coeff_key = f"sys{i}_pauli_coeffs"
        out.append(
            SDPSystem(
                A=np.asarray(data[f"sys{i}_matrix"]),
                b_raw=np.asarray(data[f"sys{i}_rhs"]),
                n_qubits=m["num_qubits"],
                original_dimension=m["original_dimension"],
                pauli_labels=[str(x) for x in data[f"sys{i}_pauli_labels"]],
                pauli_coeffs=(np.asarray(data[coeff_key]) if coeff_key in data else None),
                metadata={
                    k: v for k, v in m.items()
                    if k not in {"index", "num_qubits", "original_dimension",
                                 "padded_dimension", "num_pauli_terms"}
                },
            )
        )
    return out


def load_system(path: str | Path, index: int = 0) -> SDPSystem:
    """Load a single system by position in the archive."""
    systems = load_systems(path)
    return systems[index]


def available() -> list[Path]:
    """List the shipped .npz resource files."""
    return sorted(_DATA_DIR.glob("*.npz"))


if __name__ == "__main__":
    for f in available():
        systems = load_systems(f)
        print(f"\n{f.name}  ({len(systems)} systems)")
        for i, s in enumerate(systems):
            print(f"  [{i:>2}] {s.describe()}")
