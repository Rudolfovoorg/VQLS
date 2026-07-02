"""
Flexible end-to-end VQLS example.

Features
--------
- Accepts a user-selected matrix size.
- Uses power-of-two dimensions directly.
- Pads non-power-of-two systems to the next power of two.
- Preserves the original example's random-number sequence for:
      python3 example.py --size 4 --seed 1
- Compares the VQLS state with the exact NumPy solution.

Examples
--------
    python3 example.py
    python3 example.py --size 4
    python3 example.py --size 8 --layers 3 --maxiter 500
    python3 example.py --size 10
    python3 example.py --size 4 --show-matrix
"""

from __future__ import annotations

import argparse
import logging
import math

import numpy as np

from ansatz import HE_ansatz
from vqls import EstimatorVQLSLocal


LOGGER_NAME = "vqls_logger"


def next_power_of_two(size: int) -> int:
    """Return the smallest power of two greater than or equal to size."""
    if size < 1:
        raise ValueError("Matrix size must be a positive integer.")
    return 1 << (size - 1).bit_length()


def create_system(
    size: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a real symmetric positive-definite system A x = b.

    The construction

        A = M M^T + size I

    guarantees that A is symmetric, positive definite, and invertible.

    RandomState is intentionally used instead of default_rng so that
    --size 4 --seed 1 reproduces the original example.
    """
    M = rng.randn(size, size)
    A = M @ M.T + size * np.eye(size)

    b = np.zeros(size, dtype=float)
    b[0] = 1.0

    return A, b


def validate_system(A: np.ndarray, b: np.ndarray) -> None:
    """Validate the classical linear system before sending it to VQLS."""
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix.")

    rows, columns = A.shape
    if rows != columns:
        raise ValueError(f"A must be square; received shape {A.shape}.")

    if b.ndim != 1:
        raise ValueError("b must be a one-dimensional vector.")

    if len(b) != rows:
        raise ValueError(
            f"A is {rows} x {columns}, but b has length {len(b)}."
        )

    if not np.allclose(A, A.conj().T):
        raise ValueError("A must be Hermitian.")

    if np.linalg.norm(b) == 0:
        raise ValueError("b must not be the zero vector.")

    if np.linalg.matrix_rank(A) < rows:
        raise ValueError("A is singular and does not have a unique solution.")


def pad_system(
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Pad A and b to the next power-of-two dimension.

    For an m x m system embedded into dimension N:

        A_padded = A direct-sum I_(N-m)
        b_padded = [b, 0, ..., 0]^T

    The identity padding avoids introducing zero eigenvalues.
    """
    original_size = A.shape[0]
    padded_size = next_power_of_two(original_size)

    b_normalized = b / np.linalg.norm(b)

    if padded_size == original_size:
        return A.copy(), b_normalized.copy(), padded_size

    A_padded = np.eye(padded_size, dtype=A.dtype)
    A_padded[:original_size, :original_size] = A

    b_padded = np.zeros(padded_size, dtype=b_normalized.dtype)
    b_padded[:original_size] = b_normalized

    return A_padded, b_padded, padded_size


def align_global_phase(
    reference: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """
    Align state with reference by removing their irrelevant global phase.

    |x> and exp(i phi)|x> represent the same physical quantum state.
    """
    overlap = np.vdot(reference, state)

    if abs(overlap) < 1e-14:
        return state.copy()

    return state * np.exp(-1j * np.angle(overlap))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a flexible-size Hermitian system using VQLS."
    )

    parser.add_argument(
        "--size",
        type=int,
        default=4,
        help="Original matrix dimension. Default: 4",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of HE_ansatz layers. Default: 2",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=200,
        help="Maximum COBYLA cost evaluations. Default: 200",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random-number seed. Default: 1",
    )
    parser.add_argument(
        "--show-matrix",
        action="store_true",
        help="Print the generated original A matrix and b vector.",
    )
    parser.add_argument(
        "--show-cost",
        action="store_true",
        help="Show the VQLS cost during optimization.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.size < 2:
        raise ValueError("--size must be at least 2.")

    if args.layers < 1:
        raise ValueError("--layers must be at least 1.")

    if args.maxiter < 1:
        raise ValueError("--maxiter must be at least 1.")

    logging.getLogger(LOGGER_NAME).setLevel(
        logging.INFO if args.show_cost else logging.WARNING
    )

    # Use the legacy NumPy generator deliberately. Reusing this same object
    # for M and params0 preserves the original program's random sequence.
    rng = np.random.RandomState(args.seed)

    # ------------------------------------------------------------
    # 1. Create and validate the original classical system
    # ------------------------------------------------------------
    A_original, b_original = create_system(args.size, rng)
    validate_system(A_original, b_original)

    if args.show_matrix:
        np.set_printoptions(precision=5, suppress=True)
        print("Original A matrix")
        print("-----------------")
        print(A_original)
        print("\nOriginal b vector")
        print("-----------------")
        print(b_original)
        print()

    # ------------------------------------------------------------
    # 2. Convert to a valid qubit-register dimension
    # ------------------------------------------------------------
    A_quantum, b_quantum, quantum_dimension = pad_system(
        A_original,
        b_original,
    )

    # quantum_dimension = 2**n
    n = int(math.log2(quantum_dimension))

    # ------------------------------------------------------------
    # 3. Build the trainable parameter vector
    # ------------------------------------------------------------
    d = args.layers
    num_params = 2 * n * d

    # Continue using the same RNG after generating M. This is important:
    # for size=4, seed=1, layers=2, this matches the original example.
    params0 = rng.uniform(0.0, 2.0 * np.pi, num_params)

    print("System information")
    print("------------------")
    print(f"Original matrix size : {args.size} x {args.size}")
    print(
        f"Quantum matrix size  : "
        f"{quantum_dimension} x {quantum_dimension}"
    )
    print(f"Number of qubits     : {n}")
    print(f"Ansatz layers        : {d}")
    print(f"Trainable parameters : {num_params}")

    if quantum_dimension == args.size:
        print("Padding              : not required")
    else:
        print(
            f"Padding              : "
            f"{args.size} -> {quantum_dimension}"
        )

    # ------------------------------------------------------------
    # 4. Run VQLS
    # ------------------------------------------------------------
    solver = EstimatorVQLSLocal(
        A=A_quantum,
        b=b_quantum,
        d=d,
        params0=params0,
        ansatz_function=HE_ansatz,
    )

    solver.run(
        method="COBYLA",
        options={"maxiter": args.maxiter},
    )

    if not solver.cost_history:
        raise RuntimeError("The solver completed without recording a cost.")

    x_vqls = np.asarray(
        solver.get_optimal_statevector(),
        dtype=complex,
    )

    # ------------------------------------------------------------
    # 5. Calculate the exact reference solution
    # ------------------------------------------------------------
    b_original_normalized = b_original / np.linalg.norm(b_original)

    x_exact_original = np.linalg.solve(
        A_original,
        b_original_normalized,
    )

    # Embed the original exact solution into the padded dimension.
    x_exact = np.zeros(quantum_dimension, dtype=complex)
    x_exact[:args.size] = x_exact_original
    x_exact /= np.linalg.norm(x_exact)

    # ------------------------------------------------------------
    # 6. Compare exact and VQLS states
    # ------------------------------------------------------------
    fidelity = float(abs(np.vdot(x_exact, x_vqls)) ** 2)

    x_vqls_aligned = align_global_phase(x_exact, x_vqls)

    padding_probability = float(
        np.sum(np.abs(x_vqls[args.size:]) ** 2)
    )

    exact_display = np.real_if_close(
        x_exact[:args.size],
        tol=1000,
    )
    vqls_display = np.real_if_close(
        x_vqls_aligned[:args.size],
        tol=1000,
    )

    print("\nResults")
    print("-------")
    print(f"Final cost       : {solver.cost_history[-1]:.8f}")
    print(f"Cost evaluations : {len(solver.cost_history)}")
    print(f"Fidelity         : {fidelity:.8f}")
    print(f"Padding leakage  : {padding_probability:.8e}")

    print("\nOriginal solution components")
    print("----------------------------")
    print("Exact normalized x :", np.round(exact_display, 5))
    print("VQLS aligned x     :", np.round(vqls_display, 5))
    print(
        "VQLS magnitudes    :",
        np.round(np.abs(x_vqls[:args.size]), 5),
    )

    if quantum_dimension > args.size:
        print(
            f"\nThe final {quantum_dimension - args.size} amplitudes "
            "belong to padding-only basis states."
        )


if __name__ == "__main__":
    main()
