# =============================================================================
#  Validation: does folding by MAJORANA weight (MajoranaMeanFieldTruncation) fix
#  the k=4 blowup that the Pauli-weight MeanFieldTruncation hit on fermionic
#  (JW-mapped) Hubbard? 1D Hubbard L=6, U=4, half-filling; HF (AFM determinant)
#  X-transformed to |0…0⟩ (which is HF under particle-hole). Exact GS from eigmin.
#
#  Run (from DBF/examples): julia --project=. mf_truncation/hubbard_majorana_validation.jl [max_iter]
# =============================================================================

using PauliOperators
using DBF
using LinearAlgebra
using Printf

acc_scalar(x) = x isa AbstractVector ? (isempty(x) ? NaN : real(x[end])) : real(x)

function build_hubbard(L, U)
    N = 2L
    H = DBF.fermi_hubbard_2D(1, L, 1.0, U)
    μ = U / 2
    for i in 1:N
        H += -μ * 0.5 * (Pauli(N) - Pauli(N, Z=[i]))
    end
    coeff_clip!(H, 1e-14)
    return H
end
hf_afm_occupation(L) = [isodd(j) ? 2j-1 : 2j for j in 1:L]

function run_case(Oin, ψ, oper_trunc; max_iter)
    res = DBF.dbf_groundstate(Oin, ψ; n_body=1, max_iter=max_iter, verbose=0,
            conv_thresh=1e-6, operator_truncation=oper_trunc,
            gradient_truncation=CoeffTruncation(1e-8), adaptive_truncation=false,
            compute_var_error=true, energy_lowering_thresh=1e-6)
    E = real(res["energies"][end]); accE = acc_scalar(get(res,"accumulated_error",NaN))
    return (Ecorr = E - accE, nterms = length(res["hamiltonian"]))
end

function main(max_iter)
    L, U = 6, 4.0
    N = 2L
    H = build_hubbard(L, U)
    Eexact = eigmin(Hermitian(Matrix(H)))
    occ = hf_afm_occupation(L)
    for q in occ
        H = Pauli(N, X=[q]) * H * Pauli(N, X=[q])   # HF determinant → |0…0⟩
    end
    ψ = Ket{N}(0)

    @printf("\n1D Hubbard L=%d (N=%d qubits), U=%.1f, half-filling.  exact GS = %.6f\n", L, N, U, Eexact)
    @printf("HF reference ⟨0|H|0⟩ = %.6f\n\n", real(expectation_value(H, ψ)))
    @printf("%3s | %-16s | %-22s | %-22s\n",
            "k", "MeanField(Pauli)", "MajoranaWeight (drop)", "MajoranaMeanField (fold)")
    @printf("%3s | %12s %3s | %12s %8s | %12s %8s\n",
            "", "err", "nt", "err", "nterms", "err", "nterms")
    for k in 2:5
        mf  = run_case(SparsePauliVector(H), ψ, MeanFieldTruncation(k, ψ);          max_iter)
        mwd = run_case(SparsePauliVector(H), ψ, MajoranaWeightTruncation(k);        max_iter)
        mmf = run_case(SparsePauliVector(H), ψ, MajoranaMeanFieldTruncation(k, ψ);  max_iter)
        @printf("%3d | %12.2e %3d | %12.2e %8d | %12.2e %8d\n",
                k, abs(mf.Ecorr - Eexact), mf.nterms,
                   abs(mwd.Ecorr - Eexact), mwd.nterms,
                   abs(mmf.Ecorr - Eexact), mmf.nterms)
    end
    @printf("\n(err=|E_corrected-E_exact|; MajoranaWeight vs MajoranaMeanField = drop vs fold at equal Majorana budget)\n")
end

main(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 40)
