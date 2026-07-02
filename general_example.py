"""
Flexible VQLS example with automatic Hermitian block embedding.

The program solves A x = b using VQLS.

Processing rules
----------------
1. If A is Hermitian, use A directly.
2. If A is non-Hermitian, replace the system by

       H = [ 0   A  ]       b_H = [ b ]
           [ A†  0  ]             [ 0 ]

   and solve

       H [0] = [b].
         [x]   [0]

   The original solution x is therefore stored in the second block.
3. If the resulting dimension is not a power of two, pad it with an
   identity block so it can be represented by qubits.

Examples
--------
Reproduce the original Hermitian 4 x 4 example:
    python3 example.py --size 4 --seed 1

Generate and solve a non-Hermitian 4 x 4 system:
    python3 example.py --size 4 --matrix-type nonhermitian

Use a non-power-of-two non-Hermitian system:
    python3 example.py --size 5 --matrix-type nonhermitian \
        --layers 4 --maxiter 800

Show the generated matrix:
    python3 example.py --size 4 --matrix-type nonhermitian --show-matrix
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
    matrix_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create an invertible test system A x = b.

    hermitian:
        A = M M^T + size I

        This reproduces the construction used by the original example.

    nonhermitian:
        A = M + size I

        This is generally non-Hermitian. Adding size I makes the random
        example more likely to be well-conditioned and invertible.
    """
    M = rng.randn(size, size)

    if matrix_type == "hermitian":
        A = M @ M.T + size * np.eye(size)
    elif matrix_type == "nonhermitian":
        A = M + size * np.eye(size)

        # This should almost always be non-Hermitian, but make the test
        # deterministic even in the extremely unlikely symmetric case.
        if np.allclose(A, A.conj().T):
            A[0, -1] += 0.5
    else:
        raise ValueError(f"Unsupported matrix type: {matrix_type}")

    b = np.zeros(size, dtype=float)
    b[0] = 1.0

    return A, b


def validate_original_system(A: np.ndarray, b: np.ndarray) -> None:
    """
    Validate the original system.

    A is allowed to be non-Hermitian here because it will be embedded
    before being passed to VQLS.
    """
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

    if np.linalg.norm(b) == 0:
        raise ValueError("b must not be the zero vector.")

    if np.linalg.matrix_rank(A) < rows:
        raise ValueError(
            "A is singular and does not have a unique solution."
        )


def hermitian_block_embed(
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a square non-Hermitian system A x = b into

        [ 0   A  ] [0] = [b]
        [ A†  0  ] [x]   [0]

    The enlarged matrix is Hermitian, and the original solution x is
    stored in entries m through 2m - 1 of the enlarged solution.
    """
    m = A.shape[0]

    dtype = np.result_type(A.dtype, b.dtype, np.complex128)
    A_complex = np.asarray(A, dtype=dtype)
    b_complex = np.asarray(b, dtype=dtype)

    zero = np.zeros((m, m), dtype=dtype)

    H = np.block([
        [zero, A_complex],
        [A_complex.conj().T, zero],
    ])

    b_embedded = np.concatenate([
        b_complex,
        np.zeros(m, dtype=dtype),
    ])

    if not np.allclose(H, H.conj().T):
        raise RuntimeError(
            "Hermitian block embedding failed: H != H dagger."
        )

    return H, b_embedded


def prepare_vqls_system(
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool, slice]:
    """
    Return a Hermitian system suitable for VQLS.

    Returns
    -------
    A_work:
        Hermitian coefficient matrix.
    b_work:
        Matching right-hand-side vector.
    was_embedded:
        True when Hermitian block embedding was applied.
    solution_slice:
        Position of the original x inside the unpadded working solution.
    """
    original_size = A.shape[0]

    if np.allclose(A, A.conj().T):
        return A.copy(), b.copy(), False, slice(0, original_size)

    H, b_embedded = hermitian_block_embed(A, b)

    # The solution of H y = [b, 0]^T is y = [0, x]^T.
    return (
        H,
        b_embedded,
        True,
        slice(original_size, 2 * original_size),
    )


def pad_system(
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Pad a Hermitian system to the next power-of-two dimension.

    The padded matrix is

        A_padded = A direct-sum I

    and the padded vector is

        b_padded = [b, 0, ..., 0]^T.

    Identity padding avoids introducing zero eigenvalues.
    """
    working_size = A.shape[0]
    padded_size = next_power_of_two(working_size)

    b_normalized = b / np.linalg.norm(b)

    if padded_size == working_size:
        return A.copy(), b_normalized.copy(), padded_size

    A_padded = np.eye(padded_size, dtype=A.dtype)
    A_padded[:working_size, :working_size] = A

    b_padded = np.zeros(padded_size, dtype=b_normalized.dtype)
    b_padded[:working_size] = b_normalized

    if not np.allclose(A_padded, A_padded.conj().T):
        raise RuntimeError("The padded matrix is not Hermitian.")

    return A_padded, b_padded, padded_size


def align_global_phase(
    reference: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """
    Remove the physically irrelevant global phase between two states.
    """
    overlap = np.vdot(reference, state)

    if abs(overlap) < 1e-14:
        return state.copy()

    return state * np.exp(-1j * np.angle(overlap))


def normalize_nonzero(vector: np.ndarray, name: str) -> np.ndarray:
    """Normalize a vector and fail clearly if its norm is zero."""
    norm = np.linalg.norm(vector)

    if norm < 1e-14:
        raise RuntimeError(f"{name} has zero or near-zero norm.")

    return vector / norm


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a flexible-size linear system using VQLS, with "
            "automatic Hermitian block embedding when required."
        )
    )

    parser.add_argument(
        "--size",
        type=int,
        default=4,
        help="Original matrix dimension. Default: 4",
    )
    parser.add_argument(
        "--matrix-type",
        choices=("hermitian", "nonhermitian"),
        default="hermitian",
        help=(
            "Type of random test matrix to generate. "
            "Default: hermitian"
        ),
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
        help="Print the original and transformed matrices.",
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

    # RandomState intentionally preserves the original example's sequence.
    rng = np.random.RandomState(args.seed)

    # ------------------------------------------------------------
    # 1. Create and validate the original system A x = b
    # ------------------------------------------------------------
    A_original, b_original = create_system(
        size=args.size,
        rng=rng,
        matrix_type=args.matrix_type,
    )

    validate_original_system(A_original, b_original)

    original_size = A_original.shape[0]
    original_is_hermitian = np.allclose(
        A_original,
        A_original.conj().T,
    )

    # ------------------------------------------------------------
    # 2. Apply Hermitian block embedding when necessary
    # ------------------------------------------------------------
    (
        A_work,
        b_work,
        was_embedded,
        solution_slice,
    ) = prepare_vqls_system(A_original, b_original)

    working_size = A_work.shape[0]

    if not np.allclose(A_work, A_work.conj().T):
        raise RuntimeError(
            "The matrix passed toward VQLS is not Hermitian."
        )

    # ------------------------------------------------------------
    # 3. Pad to a power-of-two dimension
    # ------------------------------------------------------------
    A_quantum, b_quantum, quantum_dimension = pad_system(
        A_work,
        b_work,
    )

    n = int(math.log2(quantum_dimension))

    # ------------------------------------------------------------
    # 4. Build the trainable ansatz parameters
    # ------------------------------------------------------------
    d = args.layers
    num_params = 2 * n * d

    # Continue from the same random sequence used to create A.
    params0 = rng.uniform(
        0.0,
        2.0 * np.pi,
        num_params,
    )

    print("System information")
    print("------------------")
    print(
        f"Original matrix size : "
        f"{original_size} x {original_size}"
    )
    print(f"Requested matrix type: {args.matrix_type}")
    print(
        f"Original Hermitian   : "
        f"{'yes' if original_is_hermitian else 'no'}"
    )
    print(
        f"Block embedding      : "
        f"{'applied' if was_embedded else 'not required'}"
    )
    print(
        f"Embedded/work size   : "
        f"{working_size} x {working_size}"
    )
    print(
        f"Quantum matrix size  : "
        f"{quantum_dimension} x {quantum_dimension}"
    )
    print(f"Number of qubits     : {n}")
    print(f"Ansatz layers        : {d}")
    print(f"Trainable parameters : {num_params}")

    if quantum_dimension == working_size:
        print("Power-of-two padding : not required")
    else:
        print(
            f"Power-of-two padding : "
            f"{working_size} -> {quantum_dimension}"
        )

    if args.show_matrix:
        np.set_printoptions(precision=5, suppress=True)

        print("\nOriginal A")
        print("----------")
        print(A_original)

        print("\nOriginal b")
        print("----------")
        print(b_original)

        if was_embedded:
            print("\nHermitian block matrix H")
            print("------------------------")
            print(A_work)

            print("\nEmbedded right-hand side")
            print("------------------------")
            print(b_work)

        if quantum_dimension != working_size:
            print("\nFinal padded matrix passed to VQLS")
            print("----------------------------------")
            print(A_quantum)

            print("\nFinal padded vector passed to VQLS")
            print("----------------------------------")
            print(b_quantum)

    # ------------------------------------------------------------
    # 5. Run VQLS
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
        raise RuntimeError(
            "The solver completed without recording a cost."
        )

    x_vqls_full = np.asarray(
        solver.get_optimal_statevector(),
        dtype=complex,
    )

    # ------------------------------------------------------------
    # 6. Calculate the exact original solution
    # ------------------------------------------------------------
    b_original_normalized = normalize_nonzero(
        np.asarray(b_original, dtype=complex),
        "b_original",
    )

    x_exact_original = np.linalg.solve(
        np.asarray(A_original, dtype=complex),
        b_original_normalized,
    )
    x_exact_original = normalize_nonzero(
        x_exact_original,
        "x_exact_original",
    )

    # Build the exact state in the same embedded/padded coordinate system
    # used by VQLS.
    x_exact_full = np.zeros(
        quantum_dimension,
        dtype=complex,
    )
    x_exact_full[solution_slice] = x_exact_original
    x_exact_full = normalize_nonzero(
        x_exact_full,
        "x_exact_full",
    )

    # ------------------------------------------------------------
    # 7. Compare the full VQLS state with the exact embedded state
    # ------------------------------------------------------------
    full_fidelity = float(
        abs(np.vdot(x_exact_full, x_vqls_full)) ** 2
    )

    x_vqls_full_aligned = align_global_phase(
        x_exact_full,
        x_vqls_full,
    )

    # Extract the block containing the original solution x.
    x_vqls_extracted = x_vqls_full_aligned[solution_slice]
    extracted_probability = float(
        np.sum(np.abs(x_vqls_extracted) ** 2)
    )
    x_vqls_extracted = normalize_nonzero(
        x_vqls_extracted,
        "extracted VQLS solution",
    )

    extracted_fidelity = float(
        abs(np.vdot(x_exact_original, x_vqls_extracted)) ** 2
    )

    # Leakage outside the block where x should appear.
    expected_mask = np.zeros(quantum_dimension, dtype=bool)
    expected_mask[solution_slice] = True

    outside_solution_probability = float(
        np.sum(np.abs(x_vqls_full[~expected_mask]) ** 2)
    )

    padding_probability = float(
        np.sum(np.abs(x_vqls_full[working_size:]) ** 2)
    )

    exact_display = np.real_if_close(
        x_exact_original,
        tol=1000,
    )
    vqls_display = np.real_if_close(
        x_vqls_extracted,
        tol=1000,
    )

    print("\nResults")
    print("-------")
    print(f"Final cost               : {solver.cost_history[-1]:.8f}")
    print(f"Cost evaluations         : {len(solver.cost_history)}")
    print(f"Full embedded fidelity   : {full_fidelity:.8f}")
    print(f"Extracted x fidelity     : {extracted_fidelity:.8f}")
    print(f"Probability in x block   : {extracted_probability:.8f}")
    print(
        f"Outside-x probability    : "
        f"{outside_solution_probability:.8e}"
    )
    print(f"Padding leakage          : {padding_probability:.8e}")

    print("\nOriginal solution components")
    print("----------------------------")
    print("Exact normalized x :", np.round(exact_display, 5))
    print("VQLS extracted x   :", np.round(vqls_display, 5))
    print(
        "VQLS magnitudes    :",
        np.round(np.abs(x_vqls_extracted), 5),
    )

    if was_embedded:
        print(
            "\nThe original solution was extracted from the second "
            f"block: indices {original_size} through "
            f"{2 * original_size - 1}."
        )


if __name__ == "__main__":
    main()
