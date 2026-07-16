# Variational Quantum Linear Solver (VQLS)

A Qiskit implementation of the **Variational Quantum Linear Solver** for solving
linear systems **A x = b** on (simulated) quantum hardware. It provides global
and local cost functions, each in an exact *Estimator* flavour and a shot-based
*Sampler* flavour.

This version has been **verified against exact classical solutions** and includes
three correctness/robustness fixes over the original (see
[Fixes applied](#6-fixes-applied)).

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

## 8. SDP operators as VQLS inputs

The systems in `data/` are the linear systems that a semidefinite programming
solver actually produces while solving the basic SDP relaxation of Max-Cut on
Erdős–Rényi graphs,

```
max  (1/4) <L, X>     s.t.  diag(X) = 1,  X ⪰ 0
```

with `L` the graph Laplacian. They come from
[Quantum-resource-analysis](https://github.com/Rudolfovoorg/Quantum-resource-analysis)
and are exported with matrix, right-hand side, and Pauli decomposition together.
The `.npz` files are self-contained — this repository does not depend on that one.

| File | Purpose |
| ---- | ------- |
| `sdp_operators.py` | loader / adapter (`SDPSystem`, `load_system`, `available`) |
| `sdp_example.py`   | runnable demo, companion to `example.py` |
| `data/*.npz`       | the operators themselves (~220 KB total) |

---

### IPM Schur matrices — `ipm_schur_n*.npz`

Each Newton step of the primal-dual interior point method solves for the dual
step `dy` against the Schur matrix

```
M = Z⁻¹ ∘ X          (∘ = Hadamard product)
```

By the Schur product theorem `M` is **symmetric positive definite**. Two systems
are solved per iteration — *predictor* (`M dy₁ = −e`) and *corrector* — sharing
the same `M`, so one Pauli decomposition serves both. `M` has dimension `n`, so
for `n ∈ {4, 8, 16, 32}` it is already a power of two: `log₂(n)` qubits, no
padding.

### ADMM KKT matrices — `admm_kkt_n*.npz`

The ADMM `x`-update solves the saddle-point system

```
Q = [[ρI, Aᵀ],
     [A,  0 ]]        dimension n² + n
```

**`Q` does not depend on the graph.** `A` encodes `diag(X) = 1` and is a function
of `n` alone; the Laplacian enters only through the right-hand side. With
`A Aᵀ = I` the spectrum is exactly

```
{ ρ  (multiplicity n² − n),  (ρ ± √(ρ² + 4)) / 2 }
```

*independent of n*. For `ρ = 1` that is `{1, 1.618…, −0.618…}` and
`cond(Q) = 1.618/0.618 = 2.618… = φ²` — the golden ratio, for every `n`.

Two consequences:

- `Q` is **indefinite** (λ_min < 0). Still Hermitian, so `BaseVQLS` accepts it,
  but there is no SPD structure to lean on.
- Because `Q` is fixed for a given `(n, ρ)`, its Pauli decomposition can be
  computed **once** and reused across every graph and every ADMM iteration —
  only `b` changes. That is an advantage for VQLS, but it also means sweeping
  graph density tells you nothing about this operator.

---

### Why an adapter is needed

`BaseVQLS.__init__` imposes three things the raw solver output does not satisfy.

**`b` must be unit norm.** `BaseVQLS` passes `b` straight into
`qiskit.circuit.library.StatePreparation`, which rejects a non-normalized
vector. The stored RHS vectors are the raw ones the optimiser used.
`SDPSystem.b` returns the normalized copy; `SDPSystem.b_norm` keeps `‖b‖`.

**`A` must be `2ⁿ × 2ⁿ` and Hermitian.** The stored matrices are already padded
to a power of two with an identity block, and are real symmetric. The padded
system is block diagonal,

```
[ A  0 ] [ x ]   [ b ]
[ 0  I ] [ y ] = [ 0 ]
```

so the blocks decouple and `x` is untouched. The RHS padding is **zeros** — not
to keep `x` correct (the decoupling already does that) but because VQLS returns a
*normalized state*: any nonzero padding component would take up amplitude and
corrupt the solution after renormalisation.

**The scale of `x` is not recoverable from the quantum state.** VQLS returns a
unit-norm `|x⟩`, defined only up to a global phase — it solves for the
*direction* of `x`. `SDPSystem.recover_scale(x_hat)` re-fits the single scalar
classically by least squares:

```
α = ⟨A x̂, b⟩ / ‖A x̂‖²
```

This is an O(N²) matrix-vector product on the host, not part of the quantum
routine. It is the standard VQLS caveat, made explicit.

**Bonus.** `BaseVQLS` accepts a precomputed `A_LCU`. The `.npz` files ship the
Pauli decomposition, so `for_vqls()` passes it through and `qc_utils.get_LCU(A)`
— exponential in `n` — is skipped.

---

### Usage

```python
import numpy as np
from ansatz import HE_ansatz
from sdp_operators import load_system
from vqls import EstimatorVQLSLocal

s = load_system("ipm_schur_n16.npz", index=2)   # iteration 1, predictor
print(s.describe())
# ipm_schur  4q  dim 16->16  iter 1 predictor  cond 1.28  133 Pauli terms  SPD

n, d = s.n_qubits, 2
params0 = np.random.uniform(0, 2*np.pi, 2*n*d)   # HE_ansatz: 2*n*d

# for_vqls() supplies A, the NORMALIZED b, and the precomputed A_LCU
solver = EstimatorVQLSLocal(**s.for_vqls(), d=d, params0=params0,
                            ansatz_function=HE_ansatz)
solver.run(method="COBYLA", options={"maxiter": 200})

x_hat = solver.get_optimal_statevector()         # normalized direction
alpha, x = s.recover_scale(x_hat)                # actual solution of A x = b
print(s.residual(x))                             # ||A x - b|| / ||b||
```

Command line:

```bash
python sdp_example.py                                  # 2-qubit IPM system
python sdp_example.py --file ipm_schur_n16.npz --index 2
python sdp_example.py --file admm_kkt_n5.npz --index 0 --maxiter 400
python sdp_example.py --list                           # inventory
python sdp_operators.py                                # same inventory
```

> **Logging order matters.** `vqls.py` calls `logger.setLevel(logging.INFO)` at
> import time, so `logging.getLogger("vqls_logger").setLevel(...)` only takes
> effect if it runs *after* `from vqls import ...`. `example.py` and
> `sdp_example.py` both get this right; a script that sets the level before the
> import will print the cost at every evaluation.

---

### Data inventory

| File | Systems | Qubits | cond(A) range | Definiteness |
| ---- | ------- | ------ | ------------- | ------------ |
| `ipm_schur_n4.npz`  |  8 | 2 | 1.27 – 11.7   | SPD |
| `ipm_schur_n8.npz`  | 10 | 3 | 1.27 – 1.6e3  | SPD |
| `ipm_schur_n16.npz` | 14 | 4 | 1.28 – 1.5e4  | SPD |
| `ipm_schur_n32.npz` | 12 | 5 | 1.16 – 4.2e3  | SPD |
| `admm_kkt_n5.npz`   |  3 | 5 | 2.618 (fixed) | indefinite |
| `admm_kkt_n10.npz`  |  3 | 7 | 2.618 (fixed) | indefinite |

All generated with `p = 0.5`, `seed = 1`, `ρ = 1`.

Two traps when picking an index:

- **IPM iteration 0 is diagonal.** At the first Newton step `X = I`, so
  `M = Z⁻¹ ∘ I` is diagonal — only `2ⁿ` Pauli terms (`I`/`Z` only) and trivially
  solvable. Not a representative test; start at iteration 1 (`--index 2`).
- **The last IPM iterations are badly conditioned by design.** Interior point
  methods drive `μ → 0` and `cond(M)` grows accordingly. That is the method, not
  a defect — but expect VQLS to struggle there.

---

### Validated behaviour

`ipm_schur_n4.npz`, `EstimatorVQLSLocal`, `HE_ansatz`, `d=3`, COBYLA,
`maxiter=300`, seed 1 — accuracy tracks conditioning:

| IPM iter | cond(A) | Pauli terms | fidelity | residual after `recover_scale` |
| -------- | ------- | ----------- | -------- | ------------------------------ |
| 0 | 1.32 |  4 | 1.0000 | 1.26e-03 |
| 1 | 1.27 | 10 | 1.0000 | 2.95e-05 |
| 2 | 3.09 | 10 | 1.0000 | 3.30e-05 |
| 3 | 11.7 | 10 | 0.9966 | 5.79e-03 |

The scaling story these operators make available: conditioning along the IPM
path grows by ~4 orders of magnitude (`1.3 → 1.5e4` at `n = 16`) while the matrix
size and qubit count stay fixed. That isolates conditioning from problem size,
which a random-matrix benchmark cannot do.

---

### Caveats

- Simulation only. As §7 notes, `get_LCU` and `Operator(circuit)` scale
  exponentially in `n`; the shipped systems top out at 7 qubits deliberately.
  Passing the precomputed `A_LCU` removes the decomposition cost but not the
  circuit-simulation cost.
- The ADMM KKT systems are indefinite. They are included because they are what
  ADMM actually solves, not because they are good VQLS targets.
- `recover_scale` is classical. VQLS gives the direction of `x`; the magnitude
  comes from a host-side matrix-vector product. Any claim about end-to-end
  quantum advantage has to account for that step.
- All shipped systems use one graph (`p = 0.5`, `seed = 1`). For the IPM family
  the graph matters and other draws give different operators; for the ADMM family
  it provably does not.

---

## 9. Push to GitHub

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
- This is a modified version, the original source code is available at: https://github.com/omkarbihani/VQLS
- C. Bravo-Prieto, R. LaRose, M. Cerezo, Y. Subaşı, L. Cincio, P. J. Coles,
  *Variational Quantum Linear Solver*, arXiv:1909.05820 (2019).
