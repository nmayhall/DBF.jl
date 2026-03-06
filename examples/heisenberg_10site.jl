using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra

Random.seed!(42)

N = 10

# Build a 10-site antiferromagnetic Heisenberg chain with a transverse field
#   H = -2 sum_i (Jx Xi Xi+1 + Jy Yi Yi+1 + Jz Zi Zi+1) + x sum_i Xi
Jx, Jy, Jz = -1, -1, -1
H = DBF.heisenberg_1D(N, Jx, Jy, Jz)
DBF.coeff_clip!(H)

# Neel state reference
g,a = DBF.get_1d_neel_state_sequence(N)
for (gi, ai) in zip(g,a)
    global H = evolve(H, gi, ai)
end
ψ = Ket{N}(0)
display(ψ)

@printf(" N = %i\n", N)
@printf(" Number of Pauli terms in H: %i\n", length(H))
@printf(" ||H|| = %12.8f\n\n", norm(H))

e_ref = real(expectation_value(H, ψ))
@printf(" Reference state: %s\n", string(ψ))
@printf(" Reference energy: %12.8f\n", e_ref)
@printf(" Reference variance: %12.8f\n\n", real(DBF.variance(H, ψ)))

# --- Run 1: coefficient truncation only ---
println("="^60)
println(" Run 1: CoeffTruncation only (n_body=2)")
println("="^60)

res1 = dbf_groundstate(deepcopy(H), ψ,
    n_body            = 1,
    max_iter          = 20,
    conv_thresh       = 1e-3,
    evolve_truncation = CoeffTruncation(1e-4),
    grad_truncation   = CoeffTruncation(1e-6),
    # grad_coeff_thresh = 1e-4,
    correction        = EnergyVarianceCorrection(ψ),
    max_rots_per_grad = 50)

H1 = res1.hamiltonian
e1 = real(expectation_value(H1, ψ))
corr1 = res1.correction
@printf("\n Final energy:         %12.8f\n", e1)
@printf(" Accumulated error:    %12.8f\n", corr1.accumulated_energy)
@printf(" Corrected energy:     %12.8f\n", e1 - corr1.accumulated_energy)
@printf(" Final variance:       %12.8f\n", real(DBF.variance(H1, ψ)))
@printf(" Final # Pauli terms:  %i\n\n", length(H1))

# --- Run 2: composite truncation (coefficient + weight) ---
println("="^60)
println(" Run 2: CompositeTruncation (coeff + weight, n_body=2)")
println("="^60)

res2 = dbf_groundstate(deepcopy(H), ψ,
    n_body            = 1,
    max_iter          = 20,
    conv_thresh       = 1e-3,
    evolve_truncation = CompositeTruncation(CoeffTruncation(1e-4), WeightTruncation(6)),
    grad_truncation   = CompositeTruncation(CoeffTruncation(1e-4), WeightTruncation(6)),
    grad_coeff_thresh = 1e-4,
    correction        = EnergyVarianceCorrection(ψ),
    max_rots_per_grad = 50)

H2 = res2.hamiltonian
e2 = real(expectation_value(H2, ψ))
corr2 = res2.correction
@printf("\n Final energy:         %12.8f\n", e2)
@printf(" Accumulated energy err:  %12.8f\n", corr2.accumulated_energy)
@printf(" Accumulated var err:     %12.8f\n", corr2.accumulated_variance)
@printf(" Corrected energy:     %12.8f\n", e2 - corr2.accumulated_energy)
@printf(" Final variance:       %12.8f\n", real(DBF.variance(H2, ψ)))
@printf(" Final # Pauli terms:  %i\n\n", length(H2))

# --- Compare to exact diagonalization (feasible for N <= 14) ---
if N <= 14
    e_exact = minimum(real(eigvals(Matrix(H))))
    println("="^60)
    @printf(" Exact ground state energy: %12.8f\n", e_exact)
    @printf(" Run 1 energy error:        %12.8f\n", e1 - e_exact)
    @printf(" Run 2 energy error:        %12.8f\n", e2 - e_exact)
    println("="^60)
end
