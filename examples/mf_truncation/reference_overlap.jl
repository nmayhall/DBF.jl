# Reference-state FCI overlap |⟨HF|GS⟩|² for the 1D Hubbard L=6 at half-filling.
# HF = AFM determinant X-transformed to |0…0⟩ (index 1); GS = lowest eigenvector.
# Run: julia --project=. mf_truncation/reference_overlap.jl
using PauliOperators, DBF, LinearAlgebra, Printf

function build_hubbard(L, U)
    N = 2L
    H = DBF.fermi_hubbard_2D(1, L, 1.0, U)
    μ = U/2
    for i in 1:N; H += -μ*0.5*(Pauli(N) - Pauli(N, Z=[i])); end
    coeff_clip!(H, 1e-14); return H
end
hf_afm_occupation(L) = [isodd(j) ? 2j-1 : 2j for j in 1:L]

function main()
    L = 6; N = 2L
    @printf("%6s %14s %14s\n", "U", "E_GS(FCI)", "|<HF|GS>|^2")
    for U in (0.2, 0.4, 0.8, 1.6, 2.0, 3.2, 4.0)
        H = build_hubbard(L, U)
        for q in hf_afm_occupation(L); H = Pauli(N, X=[q])*H*Pauli(N, X=[q]); end
        F = eigen(Hermitian(Matrix(H)))
        Egs = F.values[1]; gs = F.vectors[:,1]
        ovlp = abs2(gs[1])                        # |<0…0|GS>|^2, |0…0⟩ is index 1
        @printf("%6.1f %14.6f %14.6f\n", U, Egs, ovlp)
    end
end
main()
