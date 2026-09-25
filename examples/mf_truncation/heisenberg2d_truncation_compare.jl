# =============================================================================
#  MeanFieldTruncation on a 2D Heisenberg model, inside DBF's dbf_groundstate.
# =============================================================================
#
#  2D antiferromagnetic Heisenberg on an Nx x Ny square lattice (J2 = 0, no
#  frustration), computational-basis reference, NO warmup — just dbf_groundstate
#  with the operator-truncation strategy varied. At a fixed weight budget k we
#  compare, against the exact ground-state energy (dense eigmin):
#
#     CoeffTruncation(ε)         drop terms with |c| <= ε        (the usual knob)
#     WeightTruncation(k)        drop terms with weight > k       (hard cutoff)
#     MeanFieldTruncation(k, ψ)  fold weight > k back onto ≤ k    (preserves ⟨ψ|O|ψ⟩)
#
#  N is kept at 12 so a dense eigmin gives a genuine exact reference; the same
#  script scales to larger N by dropping the exact column.
#
#  Run (from DBF/examples):
#     julia --project=. mf_truncation/heisenberg2d_truncation_compare.jl [k] [eps] [max_iter]

using PauliOperators
using DBF
using LinearAlgebra
using Printf

const Nx = 4
const Ny = 3                       # N = 12 qubits (dense eigmin is cheap here)

acc_scalar(x) = x isa AbstractVector ? (isempty(x) ? NaN : real(x[end])) : real(x)

# X-gate similarity so the Néel state maps to |0…0⟩ (the reference dbf_groundstate's
# source projector assumes). Flip the (i+j)-odd sublattice; coord=i+j*Nx+1.
function neel_transform!(H, Nx, Ny)
    N = Nx * Ny
    for j in 0:Ny-1, i in 0:Nx-1
        (i + j) % 2 == 1 && (H = Pauli(N, X=[i + j*Nx + 1]) * H * Pauli(N, X=[i + j*Nx + 1]))
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
    return (label = label, E = E, Ecorr = E - accE, var = var,
            nterms = length(res["hamiltonian"]), accE = accE)
end

function main(k, ϵ, max_iter)
    N = Nx * Ny
    # AFM S·S Heisenberg + on-site fields. The fields break the spin-flip parity
    # of the pure Heisenberg model, giving dbf_groundstate a nonzero 1-body
    # gradient from a product-state reference (so no warmup layer is needed —
    # a pure Heisenberg model from a Néel state has zero 1-body gradient).
    H = DBF.heisenberg_2D(Nx, Ny, -1/8, -1/8, -1/8; x=0.5, z=0.2, periodic=false)
    coeff_clip!(H, 1e-16)

    Eexact = eigmin(Hermitian(Matrix(H)))                     # dense 2^N x 2^N
    H = neel_transform!(H, Nx, Ny)                            # |0…0⟩ ≡ Néel reference
    ψ = Ket{N}(0)

    @printf("\n2D Heisenberg %dx%d (OBC), N = %d qubits\n", Nx, Ny, N)
    @printf("weight budget k = %d,  coeff threshold ε = %.0e,  max_iter = %d\n", k, ϵ, max_iter)
    @printf("exact ground-state energy   = %14.8f\n", Eexact)
    @printf("reference <ψ|H|ψ>           = %14.8f  (bits %s)\n\n",
            real(expectation_value(H, ψ)), string(ψ))

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

k        = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 2
ϵ        = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1e-2
max_iter = length(ARGS) >= 3 ? parse(Int, ARGS[3])     : 60
main(k, ϵ, max_iter)
