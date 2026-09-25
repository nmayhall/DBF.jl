# =============================================================================
#  MeanFieldTruncation on a larger 2D Heisenberg+field spin model (no exact ref).
# =============================================================================
#
#  Scaled-up version of heisenberg2d_truncation_compare.jl. At ~24 qubits a dense
#  eigmin is impossible, so we drop the exact column and compare on the honest
#  convergence metrics: the correction-tracked energy (E_corrected) and the
#  residual variance (→ 0 at an eigenstate; lower = closer to the ground state),
#  at a fixed operator budget. Reference is the Néel checkerboard product state.
#
#  Run (from DBF/examples):
#    julia --project=. mf_truncation/heisenberg2d_scale.jl [Nx] [Ny] [k] [eps] [max_iter]

using PauliOperators
using DBF
using LinearAlgebra
using Printf

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
    t = @elapsed res = DBF.dbf_groundstate(Oin, ψ;
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
            nterms=length(res["hamiltonian"]), secs=t)
end

function main(Nx, Ny, k, ϵ, max_iter)
    N = Nx * Ny
    H = DBF.heisenberg_2D(Nx, Ny, -1/8, -1/8, -1/8; x=0.5, z=0.2, periodic=false)
    coeff_clip!(H, 1e-16)
    H = neel_transform!(H, Nx, Ny)      # |0…0⟩ ≡ Néel reference
    ψ = Ket{N}(0)

    @printf("\n2D Heisenberg+field %dx%d (OBC), N = %d qubits\n", Nx, Ny, N)
    @printf("weight budget k = %d, coeff threshold ε = %.0e, max_iter = %d\n", k, ϵ, max_iter)
    @printf("Néel reference <ψ|H|ψ> = %14.8f\n\n", real(expectation_value(H, ψ)))

    cases = [
        run_case(SparsePauliVector(H), ψ, "CoeffTruncation($ϵ)",       CoeffTruncation(ϵ);        max_iter),
        run_case(SparsePauliVector(H), ψ, "WeightTruncation($k)",       WeightTruncation(k);       max_iter),
        run_case(SparsePauliVector(H), ψ, "MeanFieldTruncation($k, ψ)", MeanFieldTruncation(k, ψ); max_iter),
    ]

    @printf("%-28s %14s %14s %13s %10s %8s\n",
            "strategy", "E_bare", "E_corrected", "variance", "nterms", "secs")
    for c in cases
        @printf("%-28s %14.8f %14.8f %13.6f %10d %8.1f\n",
                c.label, c.E, c.Ecorr, c.var, c.nterms, c.secs)
    end
    @printf("\n(no exact ref at N=%d; lower variance at lower E_corrected ⇒ closer to ground state)\n", N)
end

Nx       = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 4
Ny       = length(ARGS) >= 2 ? parse(Int, ARGS[2])     : 6
k        = length(ARGS) >= 3 ? parse(Int, ARGS[3])     : 3
ϵ        = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-2
max_iter = length(ARGS) >= 5 ? parse(Int, ARGS[5])     : 40
main(Nx, Ny, k, ϵ, max_iter)
