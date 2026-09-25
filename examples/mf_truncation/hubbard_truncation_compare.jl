# =============================================================================
#  MeanFieldTruncation on the Fermi-Hubbard model with an HF (AFM) reference.
# =============================================================================
#
#  1D Fermi-Hubbard (site/JW basis, half-filling) with the antiferromagnetic
#  Hartree-Fock determinant as the reference — the mean-field product state the
#  fluctuation expansion is built around. This mirrors the data-vdbf1
#  hubbard_1d/check_run.jl setup (chemical potential μ = U/2 so half-filling is
#  the global ground state; the HF determinant is X-transformed to |0…0⟩), but
#  at a size small enough for a dense eigmin reference.
#
#  At weight budget k, compare against the exact ground state:
#     CoeffTruncation(ε)          |  WeightTruncation(k)  |  MeanFieldTruncation(k, ψ)
#
#  Run (from DBF/examples):
#     julia --project=. mf_truncation/hubbard_truncation_compare.jl [L] [U] [k] [eps] [max_iter]

using PauliOperators
using DBF
using LinearAlgebra
using Printf

acc_scalar(x) = x isa AbstractVector ? (isempty(x) ? NaN : real(x[end])) : real(x)

# 1D Hubbard chain of L sites (N = 2L qubits), t=1, given U, at half-filling
# via a μ = U/2 chemical potential (particle-hole symmetric point).
function build_hubbard(L, U)
    N = 2L
    H = DBF.fermi_hubbard_2D(1, L, 1.0, U)          # Lx=1, Ly=L -> 1D chain
    μ = U / 2
    for i in 1:N
        H += -μ * 0.5 * (Pauli(N) - Pauli(N, Z=[i]))
    end
    coeff_clip!(H, 1e-14)
    return H
end

# AFM Hartree-Fock determinant: site j occupied spin-up (odd j) / spin-down
# (even j). Mode order is [1↑,1↓,2↑,2↓,…]: up(j)=2j-1, dn(j)=2j.
function hf_afm_occupation(L)
    occ = Int[]
    for j in 1:L
        push!(occ, isodd(j) ? 2j-1 : 2j)
    end
    return occ                                       # occupied qubit indices
end

# Similarity-transform H so the HF determinant maps to |0…0⟩.
function to_zero_reference!(H, N, occ)
    for q in occ
        H = Pauli(N, X=[q]) * H * Pauli(N, X=[q])
    end
    return H
end

function run_case(Oin, ψ, label, oper_trunc; max_iter)
    res = DBF.dbf_groundstate(Oin, ψ;
            n_body                 = 1,
            max_iter               = max_iter,
            verbose                = 0,
            conv_thresh            = 1e-6,
            operator_truncation    = oper_trunc,
            gradient_truncation    = CoeffTruncation(1e-8),
            adaptive_truncation    = false,
            compute_var_error      = true,
            energy_lowering_thresh = 1e-6)
    E    = real(res["energies"][end])
    accE = acc_scalar(get(res, "accumulated_error", NaN))
    var  = real(res["variances"][end])
    return (label=label, E=E, Ecorr=E-accE, var=var,
            nterms=length(res["hamiltonian"]), accE=accE)
end

function main(L, U, k, ϵ, max_iter)
    N = 2L
    H = build_hubbard(L, U)
    Eexact = eigmin(Hermitian(Matrix(H)))            # dense 2^N x 2^N (N<=12 ok)

    occ = hf_afm_occupation(L)
    H = to_zero_reference!(H, N, occ)
    ψ = Ket{N}(0)
    e_hf = real(expectation_value(H, ψ))

    @printf("\n1D Fermi-Hubbard, L=%d sites (N=%d qubits), t=1, U=%.1f, half-filling\n", L, N, U)
    @printf("weight budget k = %d, coeff threshold ε = %.0e, max_iter = %d\n", k, ϵ, max_iter)
    @printf("exact ground-state energy        = %14.8f\n", Eexact)
    @printf("HF (AFM determinant) reference   = %14.8f   (corr. energy to recover: %.4f)\n\n",
            e_hf, e_hf - Eexact)

    cases = [
        run_case(SparsePauliVector(H), ψ, "CoeffTruncation($ϵ)",       CoeffTruncation(ϵ);        max_iter),
        run_case(SparsePauliVector(H), ψ, "WeightTruncation($k)",       WeightTruncation(k);       max_iter),
        run_case(SparsePauliVector(H), ψ, "MeanFieldTruncation($k, ψ)", MeanFieldTruncation(k, ψ); max_iter),
    ]

    @printf("%-28s %13s %13s %12s %12s %8s\n",
            "strategy", "E_bare", "E_corrected", "err(corr)", "variance", "nterms")
    for c in cases
        @printf("%-28s %13.8f %13.8f %12.2e %12.6f %8d\n",
                c.label, c.E, c.Ecorr, abs(c.Ecorr - Eexact), c.var, c.nterms)
    end
    @printf("\n(err(corr) = |E_corrected - E_exact|; E_corrected subtracts the tracked truncation loss)\n")
end

L        = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 6
U        = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 4.0
k        = length(ARGS) >= 3 ? parse(Int, ARGS[3])     : 2
ϵ        = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-2
max_iter = length(ARGS) >= 5 ? parse(Int, ARGS[5])     : 60
main(L, U, k, ϵ, max_iter)
