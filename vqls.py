from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator, Pauli
from qiskit.circuit.library import StatePreparation

from scipy.optimize import minimize
import numpy as np

import qc_utils as qcu

from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as Sampler

from abc import ABC, abstractmethod

import logging
import os
import sys

logger = logging.getLogger("vqls_logger")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

logger.addHandler(ch)


class LazyEvaluationAsString:
    """Class for lazy evaluation of function as string."""
    def __init__(self, func: callable, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return f'{self.func(*self.args, **self.kwargs)}'


class PrintInfo:
    """Base class for info/debug printing."""
    class _LazyStrings:
        def __init__(self, *args):
            self.args = args

        def __str__(self):
            return ''.join((str(s) for s in self.args))

    def who(self, back: int = 0) -> str:
        frame = sys._getframe(back + 1)
        return f'{os.path.basename(frame.f_code.co_filename)}, {frame.f_lineno}, {type(self).__name__}.{frame.f_code.co_name}()'

    def print_info(self, message: str = '', *args):
        logger.info(self._LazyStrings(f'{self.who(1)}: ', message, *args))

    def print_debug(self, message: str = '', *args):
        logger.debug(self._LazyStrings(f'{self.who(1)}: ', message, *args))

    def print_warning(self, message: str = '', *args):
        logger.warning(self._LazyStrings(f'{self.who(1)}: ', message, *args))

    def print_error(self, message: str = '', *args):
        logger.error(self._LazyStrings(f'{self.who(1)}: ', message, *args))


class BaseVQLS(ABC, PrintInfo):
    """Abstract base class for Variational Quantum Linear Solvers."""

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        d: int,
        params0: list | np.ndarray,
        ansatz_function: callable,
        B: np.ndarray = None,
        A_LCU: SparsePauliOp = None,
        B_LCU: SparsePauliOp = None,
        backend=None,
    ):
        # --- basic attributes ---
        self.A = A
        self.b = b
        self.B = B
        self.d = d
        self.ansatz_function = ansatz_function
        self.params0 = params0
        self.backend = backend or AerSimulator()

        self.cost_history = []
        self._optimal_solution = None
        self._optimal_params = None

        # --- validation ---
        n = np.log2(self.A.shape[0])
        self.n_qubits = int(n)

        if 2**self.n_qubits != A.shape[0]:
            raise ValueError("Matrix A must be of size [2**n, 2**n].")
        if not np.allclose(A, A.T.conj(), atol=1e-6):
            raise ValueError("Matrix A must be Hermitian.")
        if b.shape[0] != 2**self.n_qubits:
            raise ValueError("Vector b must have length 2**n.")

        # --- LCU encoding ---
        self.A_LCU = A_LCU if A_LCU is not None else qcu.get_LCU(A)

        if self.B is None:
            prep = StatePreparation(self.b)
            self.B = Operator(prep).to_matrix()

        self.B_LCU = B_LCU if B_LCU is not None else qcu.get_LCU(self.B)

    @abstractmethod
    def calculate_cost(self, params):
        pass

    def get_optimal_solution(self):
        """Return the scipy OptimizeResult from the last run()."""
        return self._optimal_solution

    def get_optimal_params(self):
        """Return the optimal variational parameters."""
        return self._optimal_params

    def get_optimal_circuit(self):
        """Return the ansatz circuit V(theta*) at the optimal parameters."""
        if self._optimal_solution is None:
            return None
        return self.ansatz_function(self.n_qubits, self.d, self._optimal_params)

    def get_optimal_statevector(self):
        """Return the normalized solution statevector x = V(theta*)|0>."""
        from qiskit.quantum_info import Statevector
        qc = self.get_optimal_circuit()
        if qc is None:
            return None
        return Statevector(qc).data

    def run(self, initial_params: np.ndarray = None, method: str = "COBYLA", **kwargs):
        """Optimize variational parameters to minimize the cost."""
        # FIX: `if not initial_params` raises on numpy arrays (ambiguous truth value).
        if initial_params is None:
            initial_params = self.params0

        self._optimal_solution = minimize(
            fun=lambda p: self.calculate_cost(p),
            x0=initial_params,
            method=method,
            **kwargs,
        )

        self._optimal_params = self._optimal_solution.x
        return self._optimal_solution


class EstimatorVQLS(BaseVQLS):

    def _transpile(self, circuit: QuantumCircuit):
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=0)
        return pm.run(circuit)

    def _expectation(self, circuit: QuantumCircuit, operator: SparsePauliOp):
        """Compute expectation value <0| U^{dagger} O U |0>."""
        qc_transpiled = self._transpile(circuit)
        op_layout = operator.apply_layout(qc_transpiled.layout)
        # Cache a single Estimator instance instead of re-creating it each call.
        if not hasattr(self, "_estimator"):
            self._estimator = Estimator()
        job = self._estimator.run([(qc_transpiled, op_layout)])
        result = job.result()[0]
        return result.data.evs


class EstimatorVQLSGlobal(EstimatorVQLS):

    def __init__(self, A, b, d, params0, ansatz_function, B=None, A_LCU=None, B_LCU=None, backend=None):
        super().__init__(A=A, b=b, d=d, params0=params0, ansatz_function=ansatz_function,
                         B=B, A_LCU=A_LCU, B_LCU=B_LCU, backend=backend)
        self._A_dagger_A = SparsePauliOp(self.A_LCU.adjoint() @ self.A_LCU)

    def calculate_psi_norm(self, q_circuit):
        """<psi|psi> = <0|V^d A^d A V|0>."""
        return self._expectation(q_circuit, self._A_dagger_A)

    def calculate_inner_product_b_psi(self, q_circuit):
        """<b|psi> = <0|B^d A V|0>."""
        V_LCU = qcu.get_LCU(q_circuit)
        M = Operator(self.B_LCU.adjoint() @ self.A_LCU @ V_LCU).data
        H_mat = 0.5 * (M + M.conj().T)
        K_mat = (M - M.conj().T) / (2j)

        H = SparsePauliOp.from_operator(Operator(H_mat))
        K = SparsePauliOp.from_operator(Operator(K_mat))

        qc = QuantumCircuit(self.A_LCU.num_qubits)
        real = self._expectation(qc, H)
        imag = self._expectation(qc, K)
        return real + 1j * imag

    def calculate_cost(self, params):
        """C_G = 1 - |<b|psi>|^2 / <psi|psi>."""
        q_circuit = self.ansatz_function(self.n_qubits, self.d, params)
        denom = self.calculate_psi_norm(q_circuit)
        b_psi = self.calculate_inner_product_b_psi(q_circuit)
        num = np.abs(b_psi)**2

        cost = np.real(1 - (num / denom))
        self.print_info(cost)
        self.cost_history.append(cost)
        return cost


class EstimatorVQLSLocal(EstimatorVQLS):

    def __init__(self, A, b, d, params0, ansatz_function, B=None, A_LCU=None, B_LCU=None, backend=None):
        super().__init__(A=A, b=b, d=d, params0=params0, ansatz_function=ansatz_function,
                         B=B, A_LCU=A_LCU, B_LCU=B_LCU, backend=backend)
        self._A_dagger_A = SparsePauliOp(self.A_LCU.adjoint() @ self.A_LCU)

    def calculate_psi_norm(self, q_circuit):
        """<psi|psi> = <0|V^d A^d A V|0>."""
        return self._expectation(q_circuit, self._A_dagger_A)

    def calculate_delta_j(self, j, q_circuit):
        """delta_j = <0|V^d A^d B (Z_j x I) B^d A V|0>."""
        sp = ''.join('Z' if i == j else 'I' for i in range(self.n_qubits))
        j_term = SparsePauliOp(sp)
        op = self.A_LCU.adjoint() @ self.B_LCU @ j_term @ self.B_LCU.adjoint() @ self.A_LCU
        return self._expectation(q_circuit, op)

    def calculate_cost(self, params):
        """C_L = 0.5 - (0.5 * sum_j delta_j) / (n * <psi|psi>)."""
        q_circuit = self.ansatz_function(self.n_qubits, self.d, params)
        denom = self.calculate_psi_norm(q_circuit)

        num = 0
        for j in range(self.n_qubits):
            num += self.calculate_delta_j(j, q_circuit)

        cost = np.real(0.5 - (0.5 * num / (self.n_qubits * denom)))
        self.print_info(cost)
        self.cost_history.append(cost)
        return cost


# ----------------------------- Sampler classes -----------------------------
class SamplerVQLS(BaseVQLS):

    def __init__(self, A, b, d, params0, ansatz_function, B=None, A_LCU=None, B_LCU=None, backend=None, num_shots=1024):
        super().__init__(A=A, b=b, d=d, params0=params0, ansatz_function=ansatz_function,
                         B=B, A_LCU=A_LCU, B_LCU=B_LCU, backend=backend)
        self.num_shots = num_shots

    def _transpile(self, circuit):
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=0)
        return pm.run(circuit)

    def _run_circuit(self, qc_transpiled):
        sampler = Sampler()
        return sampler.run([qc_transpiled], shots=self.num_shots).result()[0]

    def _expectation(self, U0, operators):
        """Compute <0| U0^d (prod operators) U0 |0> via Hadamard tests."""
        composite_op = SparsePauliOp.from_operator(operators[0])
        for op in operators[1:]:
            composite_op = SparsePauliOp.from_operator(op) @ composite_op

        is_hermitian = np.allclose(composite_op.to_matrix(), composite_op.adjoint().to_matrix(), atol=1e-8)

        hadamard_test = qcu.SamplerHadamardTest(U_psi=U0, Us=operators, num_shots=self.num_shots)
        exp_real = hadamard_test.get_expectation_real()
        exp_imag = 0 if is_hermitian else hadamard_test.get_expectation_imag()
        return exp_real + 1j * exp_imag


class SamplerVQLSGlobal(SamplerVQLS):

    def __init__(self, A, b, d, params0, ansatz_function, B=None, A_LCU=None, B_LCU=None, backend=None, num_shots=1024):
        super().__init__(A=A, b=b, d=d, params0=params0, ansatz_function=ansatz_function,
                         B=B, A_LCU=A_LCU, B_LCU=B_LCU, backend=backend, num_shots=num_shots)
        self.B_dagger_gate = qcu.make_gate(self.B_LCU.adjoint())

    def _get_beta_lm(self, qc_ansatz, l, m):
        """beta_lm = <0|V^d A_m^d A_l V |0>."""
        if l == m:
            return 1
        V_operator = Operator(qc_ansatz)
        A_l = self.A_LCU[l].paulis[0]
        A_m = self.A_LCU[m].paulis[0]
        return self._expectation(U0=V_operator, operators=[A_l, A_m.adjoint()])

    def _get_gamma_lm(self, qc_ansatz, l, m):
        """gamma_lm = <0|B^d A_l V|0> <0|V^d A_m^d B|0>."""
        A_l = self.A_LCU[l].paulis[0]
        A_m = self.A_LCU[m].paulis[0]
        V_operator = Operator(qc_ansatz)
        V_dagger_operator = V_operator.adjoint()
        I_operator = SparsePauliOp('I' * self.n_qubits)

        if l == m:
            gamma_l = self._expectation(U0=I_operator, operators=[V_operator, A_l, self.B_LCU.adjoint()])
            # FIX: diagonal term is |gamma_l|^2, NOT (Re gamma_l)^2.
            return float(np.abs(gamma_l)**2)

        gamma_l = self._expectation(U0=I_operator, operators=[V_operator, A_l, self.B_LCU.adjoint()])
        gamma_m = self._expectation(U0=I_operator, operators=[self.B_LCU, A_m.adjoint(), V_dagger_operator])
        return gamma_l * gamma_m

    def calculate_psi_norm(self, qc_ansatz):
        """<psi|psi> = sum_lm a_m* a_l beta_lm."""
        inner_product = 0
        for l in range(len(self.A_LCU)):
            for m in range(l, len(self.A_LCU)):
                beta_lm = self._get_beta_lm(qc_ansatz, l, m)
                inner_product += self.A_LCU.coeffs[m].conjugate() * self.A_LCU.coeffs[l] * beta_lm
                if l != m:
                    # FIX: the (m,l) partner uses conj(beta_lm), since beta_ml = conj(beta_lm).
                    inner_product += self.A_LCU.coeffs[l].conjugate() * self.A_LCU.coeffs[m] * np.conjugate(beta_lm)
        return np.real(inner_product)

    def calculate_norm_b_psi_squared(self, qc_ansatz):
        """|<b|psi>|^2 = sum_lm a_m* a_l gamma_lm."""
        inner_product = 0
        for l in range(len(self.A_LCU)):
            for m in range(len(self.A_LCU)):
                gamma_lm = self._get_gamma_lm(qc_ansatz, l, m)
                inner_product += self.A_LCU.coeffs[m].conjugate() * self.A_LCU.coeffs[l] * gamma_lm
        return np.real(inner_product)

    def calculate_cost(self, params):
        """C_G = 1 - |<b|psi>|^2 / <psi|psi>."""
        qc = self.ansatz_function(self.n_qubits, self.d, params)
        den = self.calculate_psi_norm(qc_ansatz=qc)
        num = self.calculate_norm_b_psi_squared(qc_ansatz=qc)

        cost = np.squeeze(1 - (num / den)).real
        self.print_info(cost)
        self.cost_history.append(cost)
        return cost


class SamplerVQLSLocal(SamplerVQLS):

    def __init__(self, A, b, d, params0, ansatz_function, B=None, A_LCU=None, B_LCU=None, backend=None, num_shots=1024):
        super().__init__(A=A, b=b, d=d, params0=params0, ansatz_function=ansatz_function,
                         B=B, A_LCU=A_LCU, B_LCU=B_LCU, backend=backend, num_shots=num_shots)
        self.B_dagger_gate = qcu.make_gate(self.B_LCU.adjoint())

    def _get_beta_lm(self, qc_ansatz, l, m):
        """beta_lm = <0|V^d A_m^d A_l V |0>."""
        if l == m:
            return 1
        V_operator = Operator(qc_ansatz)
        A_l = self.A_LCU[l].paulis[0]
        A_m = self.A_LCU[m].paulis[0]
        return self._expectation(U0=V_operator, operators=[A_l, A_m.adjoint()])

    def _get_delta_lm_j(self, qc_ansatz, l, m, j):
        """delta_lm_j = <psi| A_m^d B (Z_j x I) B^d A_l |psi>."""
        A_l = self.A_LCU[l].paulis[0]
        A_m = self.A_LCU[m].paulis[0]
        V_operator = Operator(qc_ansatz)
        sp = ''.join('Z' if i == j else 'I' for i in range(self.n_qubits))
        j_term = SparsePauliOp(sp)
        operators = [A_l, self.B_LCU.adjoint(), j_term, self.B_LCU, A_m.adjoint()]
        return self._expectation(U0=V_operator, operators=operators)

    def calculate_psi_norm(self, qc_ansatz):
        """<psi|psi> = sum_lm a_m* a_l beta_lm."""
        inner_product = 0
        for l in range(len(self.A_LCU)):
            for m in range(l, len(self.A_LCU)):
                beta_lm = self._get_beta_lm(qc_ansatz, l, m)
                inner_product += self.A_LCU.coeffs[m].conjugate() * self.A_LCU.coeffs[l] * beta_lm
                if l != m:
                    # FIX: conj(beta_lm) for the (m,l) partner term.
                    inner_product += self.A_LCU.coeffs[l].conjugate() * self.A_LCU.coeffs[m] * np.conjugate(beta_lm)
        return np.real(inner_product)

    def calculate_cost(self, params):
        """C_L = 0.5 - (0.5 * sum_j delta_j) / (n * <psi|psi>)."""
        qc = self.ansatz_function(self.n_qubits, self.d, params)
        denom = self.calculate_psi_norm(qc_ansatz=qc)

        num = 0
        for j in range(self.n_qubits):
            for l in range(len(self.A_LCU)):
                for m in range(len(self.A_LCU)):
                    delta_lm_j = self._get_delta_lm_j(qc_ansatz=qc, l=l, m=m, j=j)
                    num += self.A_LCU.coeffs[l] * self.A_LCU.coeffs[m].conjugate() * delta_lm_j

        cost = np.real(0.5 - (0.5 * num / (self.n_qubits * denom)))
        self.print_info(cost)
        self.cost_history.append(cost)
        return cost
