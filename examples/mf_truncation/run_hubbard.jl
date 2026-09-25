# =============================================================================
#  Run the Hubbard truncation comparison ONCE per strategy and SAVE TRAJECTORIES:
#    logs/<tag>_<strategy>.out  — verbose=1 per-iteration table (full trajectory)
#    logs/<tag>_<strategy>.csv  — per-grad: igrad,energy,variance,acc_err,acc_var_err
#
#  1D Hubbard L=6, U=4, half-filling; HF (AFM determinant) X-transformed to |0…0⟩.
#  Strategies compared at budget k (Majorana weight for the Majorana ones):
#    Coeff(ε) | MeanField(k) [Pauli wt] | MajoranaWeight(k) [drop] | MajoranaMeanField(k) [fold]
#
#  Run (from DBF/examples):  julia --project=. mf_truncation/run_hubbard.jl [k] [eps] [max_iter]
# =============================================================================

using PauliOperators
using DBF
using LinearAlgebra
using Printf

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

function save_traj(path, label, res, meta)
    E  = real.(res["energies_per_grad"]);  V  = real.(res["variance_per_grad"])
    ae = real.(res["accumulated_error_per_grad"])
    av = real.(res["accumulated_var_error_per_grad"])
    n  = min(length(E), length(V), length(ae), length(av))
    open(path, "w") do io
        println(io, "# label=$label ", join(["$k=$v" for (k,v) in meta], " "),
                    " nterms=", length(res["hamiltonian"]))
        println(io, "igrad,energy,variance,acc_err,acc_var_err")
        for i in 1:n
            @printf(io, "%d,%.10g,%.10g,%.10g,%.10g\n", i, E[i], V[i], ae[i], av[i])
        end
    end
    println("  wrote ", path)
end

function main(k, ϵ, max_iter, U)
    L = 6
    N = 2L
    tag = "hubbardL$(L)U$(U)_k$(k)"
    logdir = joinpath(@__DIR__, "logs"); mkpath(logdir)

    H = build_hubbard(L, U)
    Eexact = eigmin(Hermitian(Matrix(H)))
    for q in hf_afm_occupation(L)
        H = Pauli(N, X=[q]) * H * Pauli(N, X=[q])         # HF determinant → |0…0⟩
    end
    ψ = Ket{N}(0)
    meta = ("eexact"=>Eexact, "N"=>N, "L"=>L, "U"=>U, "k"=>k, "eps"=>ϵ, "max_iter"=>max_iter)

    strategies = [
        ("Coeff($ϵ)",            CoeffTruncation(ϵ)),
        ("MajoranaWeight($k)",   MajoranaWeightTruncation(k)),
        ("MajoranaMeanField($k)",MajoranaMeanFieldTruncation(k, ψ)),
    ]
    println("running $tag (Eexact=$(round(Eexact,digits=6))) ...")
    for (label, strat) in strategies
        outpath = joinpath(logdir, "$(tag)_$(label).out")
        res = open(outpath, "w") do io
            redirect_stdout(io) do
                DBF.dbf_groundstate(SparsePauliVector(H), ψ;
                    n_body=1, max_iter=max_iter, verbose=1, conv_thresh=1e-6,
                    operator_truncation=strat, gradient_truncation=CoeffTruncation(1e-8),
                    adaptive_truncation=false, compute_var_error=true, energy_lowering_thresh=1e-6)
            end
        end
        println("  wrote ", outpath)
        save_traj(joinpath(logdir, "$(tag)_$(label).csv"), label, res, meta)
    end
    println("done $tag.")
end

k        = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 4
ϵ        = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1e-2
max_iter = length(ARGS) >= 3 ? parse(Int, ARGS[3])     : 40
U        = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 4.0
main(k, ϵ, max_iter, U)
