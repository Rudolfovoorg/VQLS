# Variational Quantum Linear Solver (VQLS)

A Qiskit implementation of the **Variational Quantum Linear Solver** for solving
linear systems **A x = b** on (simulated) quantum hardware. It provides global
and local cost functions, each in an exact *Estimator* flavour and a shot-based
*Sampler* flavour.

This version has been **verified against exact classical solutions** and includes
three correctness/robustness fixes over the original (see
[Fixes applied](#fixes-applied)).

---

## 1. Problem Statement

Classically, **A x = b** is solved by inverting or factorizing `A`. VQLS instead
treats it as an optimization over a parameterized quantum circuit:

- Encode the right-hand side as a quantum state: `|b⟩ = B|0⟩`.
- Prepare a trial solution with a parameterized ansatz: `|x(θ)⟩ = V(θ)|0⟩`.
- Define the applied state `|ψ⟩ = A|x(θ)⟩`.
- Tune `θ` so that `|ψ⟩ ∝ |b⟩`. At that point `|x(θ)⟩` is the (normalized) solution.

A cost function measures how far `|ψ⟩` is from `|b⟩`; it reaches `0` at the solution.

**Input requirements**
- `A` must be **Hermitian**, of size `[2ⁿ × 2ⁿ]`.
- `b` must have length `2ⁿ` and should be a **normalized** statevector (it is
  encoded via state preparation).
- `n` (qubit count) is inferred as `log₂(A.shape[0])`.

**Cost functions** — with the LCU decomposition `A = Σₗ aₗ Aₗ` into Pauli strings:

Global:
```
C_G = 1 − |⟨b|ψ⟩|² / ⟨ψ|ψ⟩
```
Local (mitigates barren plateaus):
```
C_L = 1/2 − (1 / 2n) · Σⱼ ⟨ψ| B (Zⱼ ⊗ I) B† |ψ⟩ / ⟨ψ|ψ⟩
```

Based on Bravo-Prieto, LaRose, Cerezo, Subaşı, Cincio, Coles,
*Variational Quantum Linear Solver* (arXiv:1909.05820).

---

## 2. Project Structure

```
VQLS/
├── vqls.py            # BaseVQLS + 4 solvers (Estimator/Sampler × Global/Local)
├── qc_utils.py        # LCU decomposition, gate builders, Hadamard tests
├── ansatz.py          # HE_ansatz and fixed_ansatz trial circuits
├── example.py         # runnable end-to-end demo (verified)
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

### Solver variants

| Class | Cost | Evaluation |
|-------|------|------------|
| `EstimatorVQLSGlobal` | Global | Exact expectation values (`EstimatorV2`) |
| `EstimatorVQLSLocal`  | Local  | Exact expectation values (`EstimatorV2`) |
| `SamplerVQLSGlobal`   | Global | Shot-based Hadamard tests (`SamplerV2`) |
| `SamplerVQLSLocal`    | Local  | Shot-based Hadamard tests (`SamplerV2`) |

Estimator solvers are exact/noiseless (good for validating). Sampler solvers
estimate the same quantities from `num_shots` measurements (closer to hardware).

### Ansätze

| Function | Gates | Parameter count |
|----------|-------|-----------------|
| `HE_ansatz`    | RY, RZ + CNOT | `2 · n · d` |
| `fixed_ansatz` | RY + CZ       | `n · d`     |

`params0` **must** match the chosen ansatz's parameter count.

---

## 3. Installation

> Requires **Python ≥ 3.12**.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Run the demo

```bash
python example.py
```

Expected output (deterministic, `seed=1`):

```
final cost      : 0.00232
cost evaluations: 200
fidelity        : 0.9916   (1.0 = perfect match)
exact   x       : [ 0.962 -0.165 -0.166  0.138]
VQLS    x       : [0.975 0.163 0.129 0.08 ]  (magnitudes)
```

> Note: `python vqls.py` produces no output — it only defines classes. Use
> `example.py` or import the solvers in your own script.

---

## 5. Usage

```python
import numpy as np
from qiskit.quantum_info import Statevector
from ansatz import HE_ansatz
from vqls import EstimatorVQLSLocal     # any of the four solver classes

# 2-qubit (4x4) Hermitian system
M = np.random.randn(4, 4)
A = M @ M.T + 4 * np.eye(4)             # Hermitian, well-conditioned
b = np.array([1.0, 0.0, 0.0, 0.0])     # normalized |00>

n, d = 2, 2
params0 = np.random.uniform(0, 2*np.pi, 2 * n * d)   # HE_ansatz: 2*n*d params

solver = EstimatorVQLSLocal(A=A, b=b, d=d, params0=params0,
                            ansatz_function=HE_ansatz)
solver.run(method="COBYLA", options={"maxiter": 200})

x = solver.get_optimal_statevector()   # normalized solution |x>
result = solver.get_optimal_solution() # scipy OptimizeResult
print(solver.cost_history[-1])         # final cost
```

Shot-based simulation — swap in a Sampler class and set `num_shots`:
```python
from vqls import SamplerVQLSLocal
solver = SamplerVQLSLocal(A=A, b=b, d=d, params0=params0,
                          ansatz_function=HE_ansatz, num_shots=4096)
```

Lighter ansatz — note the different parameter count:
```python
from ansatz import fixed_ansatz
params0 = np.random.uniform(0, 2*np.pi, n * d)       # fixed_ansatz: n*d params
solver = EstimatorVQLSLocal(A=A, b=b, d=d, params0=params0,
                            ansatz_function=fixed_ansatz)
```

To watch convergence, set the logger to INFO:
```python
import logging; logging.getLogger("vqls_logger").setLevel(logging.INFO)
```

---

## 6. Fixes applied

This version corrects three issues found by comparing the solvers against exact
numpy solutions:

1. **`SamplerVQLSGlobal` diagonal term (real bug).** The diagonal `γ_ll` was
   computed as `(Re γ_l)²` instead of `|γ_l|²`, giving wrong global costs
   whenever the amplitude had an imaginary part (e.g. any RZ-containing ansatz).
   On a test case this gave `0.162` instead of the true `0.016`; the fix restores
   agreement to within shot noise.
2. **`run()` crashed on array inputs.** `if not initial_params` raises on numpy
   arrays ("ambiguous truth value"); changed to `if initial_params is None`.
3. **Missing conjugate in `calculate_psi_norm`.** The conjugate-partner term
   reused `beta_lm` instead of `conj(beta_lm)`. Harmless for Hermitian `A`
   (real coefficients) but incorrect in general; now explicitly conjugated.

Minor: the `EstimatorV2` instance is cached instead of re-created per cost
evaluation, and `get_optimal_params()` / `get_optimal_statevector()` helpers were
added.

The `EstimatorVQLSGlobal` and `EstimatorVQLSLocal` solvers were already correct
(global cost matched the exact value to machine precision; local solver converges
to fidelity > 0.99 with `numpy.linalg.solve`).

---

## 7. Caveats & scope

- This is a **simulation / proof-of-concept**. Several subroutines (full matrix
  Pauli decomposition, operator products via `Operator(circuit)`) scale
  exponentially in `n`, so it is intended for small systems and does **not**
  demonstrate a quantum speedup.
- VQLS in general has no proven advantage over classical solvers and is subject
  to barren plateaus and measurement overhead at scale. Use the **local** cost
  for larger `n`.
- Start with an Estimator solver to confirm convergence, then move to a Sampler
  solver to study shot-noise effects.

---

## 8. Push to GitHub

```bash
cd VQLS
git init
git add .
git commit -m "Initial commit: corrected VQLS implementation"

# create an empty repo named VQLS on GitHub first, then:
git branch -M main
git remote add origin https://github.com/<your-username>/VQLS.git
git push -u origin main
```

---

## References

- C. Bravo-Prieto, R. LaRose, M. Cerezo, Y. Subaşı, L. Cincio, P. J. Coles,
  *Variational Quantum Linear Solver*, arXiv:1909.05820 (2019).
