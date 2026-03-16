# DBF.jl

**Double Bracket Flow** methods for quantum Hamiltonian diagonalization, ground state preparation, and disentanglement using Pauli operator algebra.

DBF.jl works in the **Heisenberg picture**: rather than optimizing the quantum state, it iteratively applies unitary rotations to transform the Hamiltonian itself. Each rotation has the form $e^{i\theta G/2} H\, e^{-i\theta G/2}$ where $G$ is a Pauli generator. Operator truncation keeps the representation compact, enabling simulation of systems beyond the reach of exact methods. Truncation is controlled via PauliOperators.jl's composable `TruncationStrategy` type system.

Built on [PauliOperators.jl](https://github.com/nmayhall/PauliOperators.jl) for efficient symplectic Pauli algebra (up to 128 qubits).

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/nmayhall/DBF.jl")
```

## Quick Start

### Diagonalize a Hamiltonian

```julia
using DBF
using PauliOperators

# Build a 6-qubit Heisenberg Hamiltonian
N = 6
H = DBF.heisenberg_1D(N, 1.0, 1.0, 1.0, z=0.1)

# Diagonalize via double bracket flow
H_diag, generators, angles = dbf_diag(H,
    max_iter=1000,
    conv_thresh=1e-6,
    truncation=CoeffTruncation(1e-4))

# Or use composite truncation (coefficient + weight clipping):
H_diag, generators, angles = dbf_diag(H,
    max_iter=1000,
    conv_thresh=1e-6,
    truncation=CompositeTruncation(CoeffTruncation(1e-4), WeightTruncation(10)))

# H_diag is now approximately diagonal in the Z basis.
# generators and angles record the unitary circuit:
#   U = prod_i exp(i * angles[i]/2 * generators[i])
```

### Find the Ground State

```julia
using DBF
using PauliOperators

N = 8
H = DBF.heisenberg_1D(N, 1.0, 2.0, 3.0, z=0.1)

# Pick the best computational basis state as the reference
ψ = Ket{N}(argmin([real(expectation_value(H, Ket{N}(i))) for i in 0:2^N-1]))

# Optimize: transform H so that ψ becomes the ground state
res = dbf_groundstate(H, ψ,
    max_iter=50,
    conv_thresh=1e-4,
    operator_truncation=CoeffTruncation(1e-6),
    gradient_truncation=CoeffTruncation(1e-8))

# For more aggressive truncation, use composite strategies:
# res = dbf_groundstate(H, ψ,
#     operator_truncation=CompositeTruncation(CoeffTruncation(1e-6), WeightTruncation(8)),
#     gradient_truncation=CompositeTruncation(CoeffTruncation(1e-6), MajoranaWeightTruncation(4)))

H_opt = res["hamiltonian"]
E = real(expectation_value(H_opt, ψ))       # ground state energy
var = variance(H_opt, ψ)                     # should be near zero

# Optional: compute PT2 correction for remaining truncation error
e0, e2 = DBF.pt2(H_opt, ψ)
```

### ADAPT-VQE Style Optimization

```julia
using DBF
using PauliOperators

N = 6
H = DBF.heisenberg_1D(N, 1.0, 1.0, 1.0)
ψ = Ket{N}(0)

# Build an operator pool
pool = vcat(DBF.generate_pool_1_weight(N), DBF.generate_pool_2_weight(N))

# Run ADAPT optimization
H_opt, generators, angles = adapt(H, pool, ψ,
    max_iter=30,
    conv_thresh=1e-4,
    truncation=CoeffTruncation(1e-4))
```

### Block-Diagonalize (Disentangle)

```julia
using DBF
using PauliOperators

N = 8
H = DBF.heisenberg_1D(N, 1.0, 1.0, 1.0)

# Disentangle qubits 1:4 from qubits 5:8
H_disent, generators, angles = dbf_disentangle(H, 4,
    max_iter=100,
    conv_thresh=1e-4)
```

## Core Algorithms

### Double Bracket Flow for Diagonalization (`dbf_diag`)

Iteratively maximizes $\|\mathrm{diag}(H)\|$ by:

1. Computing a search direction from the commutator $[H, \mathrm{diag}(H)]$
2. Analytically optimizing the rotation angle $\theta$
3. Evolving $H \to e^{i\theta G/2}\, H\, e^{-i\theta G/2}$
4. Truncating small coefficients and high-weight terms

Returns the (approximately) diagonalized Hamiltonian plus the full circuit as generators and angles.

### Ground State Preparation (`dbf_groundstate`)

Transforms $H$ so that a computational basis state $|\psi\rangle$ becomes its ground state, minimizing $\langle\psi|H|\psi\rangle$. Uses an $n$-body Z-projector approximation to $|\psi\rangle\langle\psi|$ as the source operator. Supports:

- Adjustable projector body order (`n_body=1` to `6`)
- Separate `operator_truncation` and `gradient_truncation` strategies (any `TruncationStrategy`)
- Automatic truncation error tracking via `CorrectionAccumulator`
- PT2 energy corrections at each macro-iteration
- JLD2 checkpointing (`checkfile` kwarg)

### ADAPT Optimization (`adapt`)

Pool-based greedy optimization. At each step, the generator with the largest gradient $|\mathrm{Im}\langle\psi|[G_i, H]|\psi\rangle|$ is selected from a predefined pool, and the rotation angle is optimized analytically.

Built-in pool generators: `generate_pool_1_weight` through `generate_pool_6_weight`, and `qubitexcitationpool` for fermionic-style singles and doubles.

### Disentanglement (`dbf_disentangle`)

Block-diagonalizes $H$ with respect to a bipartition of the qubits, minimizing the norm of operators that act across both subsystems.

## Model Hamiltonians

DBF.jl includes several built-in Hamiltonians:

| Function | Model |
|----------|-------|
| `heisenberg_1D(N, Jx, Jy, Jz)` | 1D Heisenberg with periodic BC |
| `heisenberg_2D(Nx, Ny, Jx, Jy, Jz)` | 2D Heisenberg on square lattice |
| `heisenberg_2D_zigzag(Nx, Ny, Jx, Jy, Jz)` | 2D Heisenberg with snake ordering |
| `heisenberg_central_spin(N, Jx, Jy, Jz)` | Central spin model |
| `heisenberg_sparse(N, Jx, Jy, Jz, sparsity)` | Random sparse couplings |
| `hubbard_model_1D(L, t, U)` | 1D Fermi-Hubbard (Jordan-Wigner) |
| `fermi_hubbard_2D(Lx, Ly, t, U)` | 2D spinful Fermi-Hubbard |
| `S2(N)`, `Sz(N)` | Total spin operators |

All Hamiltonians are returned as `PauliSum` objects. Optional keyword arguments include magnetic field terms (`x`, `y`, `z`) and boundary conditions (`periodic`).

## Truncation

After each rotation, the Hamiltonian is truncated to control operator growth. All DBF functions accept `TruncationStrategy` objects from PauliOperators.jl, making truncation composable and swappable:

| Strategy | Description |
|----------|-------------|
| `CoeffTruncation(ε)` | Remove terms with $\|c_i\| \le \epsilon$ |
| `WeightTruncation(w)` | Remove terms with Pauli weight $> w$ |
| `MajoranaWeightTruncation(w)` | Remove terms with Majorana weight $> w$ |
| `CompositeTruncation(s1, s2, ...)` | Apply multiple strategies in sequence |
| `StochasticCoeffTruncation(ε)` | Unbiased Russian Roulette compression |
| `AdaptiveTruncation(max_terms, min_thresh)` | Adaptively increase threshold if too many terms |
| `NoTruncation()` | Identity (do nothing) |

**Composing strategies:**

```julia
# Coefficient clip, then weight clip, then Majorana weight clip
trunc = CompositeTruncation(
    CoeffTruncation(1e-6),
    WeightTruncation(8),
    MajoranaWeightTruncation(4))

res = dbf_groundstate(H, ψ, operator_truncation=trunc)
```

**Error tracking:** Functions that track truncation error (`dbf_groundstate`, `adapt`, sequence `evolve`) use PauliOperators' `CorrectionAccumulator` internally to measure $\Delta E = \langle\psi|H|\psi\rangle_\text{after} - \langle\psi|H|\psi\rangle_\text{before}$ at each truncation step. The accumulated error is returned in the results.

## Perturbation Theory & Subspace Methods

For post-processing or refinement, DBF.jl provides Schrodinger-picture utilities that work within a subspace defined by the first-order interacting space:

- **`pt2(H, ψ)`** -- Second-order Rayleigh-Schrodinger perturbation theory correction
- **`cepa(H, ψ)`** -- Coupled Electron Pair Approximation via iterative linear solve (KrylovKit)
- **`fois_ci(H, ψ)`** -- First-Order Interacting Space CI via iterative diagonalization (KrylovKit)

These use the `XZPauliSum` representation (Pauli terms grouped by X-bitstring) for efficient matrix-vector products without building dense matrices.

## Exported API

| Function | Description |
|----------|-------------|
| `dbf_diag(H; kwargs...)` | Diagonalize a Hamiltonian via double bracket flow |
| `dbf_groundstate(H, ψ; kwargs...)` | Ground state preparation |
| `dbf_disentangle(H, M; kwargs...)` | Block-diagonalize across a qubit partition |
| `adapt(H, pool, ψ; kwargs...)` | ADAPT-VQE style pool optimization |
| `pack_x_z(H)` | Convert `PauliSum` to X-bitstring-grouped representation |
| `project(k, basis)` | Project a `KetSum` onto a basis |

Many additional utilities (Hamiltonians, pool generators, theta optimizers, perturbation theory) are accessible via the `DBF.` prefix.

## Dependencies

- [PauliOperators.jl](https://github.com/nmayhall/PauliOperators.jl) -- Pauli operator algebra (evolution, clipping, statistics, gates)
- [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl) -- Iterative eigensolvers and linear solvers
- [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) -- 1D optimization for rotation angles
- [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) -- Matrix-free linear maps for subspace methods
- [JLD2.jl](https://github.com/JuliaIO/JLD2.jl) -- Checkpointing

## References

- Chinmay Shrikhande, Arnab Bachhar, Aaron Rodriguez Jimenez, Nicholas J. Mayhall. (2025) Rapid ground state energy estimation with a Sparse Pauli Dynamics-enabled Variational Double Bracket Flow. [arXiv:2511.21651](https://arxiv.org/abs/2511.21651)
