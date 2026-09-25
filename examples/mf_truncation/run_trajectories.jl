# =============================================================================
#  Run the truncation-comparison calculations ONCE and save their trajectories
#  to CSV, so plots can be regenerated without re-running the jobs.
#
#  For each strategy writes BOTH:
#    logs/<tag>_<strategy>.out  — the formatted verbose=1 per-iteration table
#                                 (params header + trajectory + timer), i.e. the
#                                 same stdout log format as data-vdbf1.
#    logs/<tag>_<strategy>.csv  — machine-readable per-grad columns:
#                                 igrad, energy, variance, acc_err, acc_var_err
#                                 (bare energy/variance + cumulative corrections;
#                                 corrected = bare - cumulative).
#
#  Run (from DBF/examples):  julia --project=. mf_truncation/run_trajectories.jl [k] [eps] [max_iter]
# =============================================================================

using PauliOperators
using DBF
using LinearAlgebra
using Printf

# X-gate (Clifford) similarity so the classical Néel state maps to |0…0⟩, which
# is the reference dbf_groundstate's create_0_projector source assumes. Flip the
# (i+j)-odd checkerboard sublattice. coord_to_index(i,j)=i+j*Nx+1 (matches heisenberg_2D).
function neel_transform!(H, Nx, Ny)
    N = Nx * Ny
    for j in 0:Ny-1, i in 0:Nx-1
        if (i + j) % 2 == 1
            s = i + j*Nx + 1
            H = Pauli(N, X=[s]) * H * Pauli(N, X=[s])
        end
    end
    return H
end

function save_traj(path, label, res, meta)
    E  = real.(res["energies_per_grad"])
    V  = real.(res["variance_per_grad"])
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

function main(k, ϵ, max_iter, zzr)
    N = 12
    tag = zzr == 1.0 ? "heis12_k$(k)" : "heis12zz$(zzr)_k$(k)"
    logdir = joinpath(@__DIR__, "logs"); mkpath(logdir)

    # XXZ: ZZ coupling is zzr× the XX/YY coupling (zzr=2 ⇒ ZZ twice XX,YY).
    Jx = -1/8
    H = DBF.heisenberg_2D(4, 3, Jx, Jx, zzr*Jx; x=0.5, z=0.2, periodic=false)
    coeff_clip!(H, 1e-16)
    Eexact = eigmin(Hermitian(Matrix(H)))       # spectrum is invariant under the transform
    H = neel_transform!(H, 4, 3)                 # |0…0⟩ is now the Néel reference
    ψ = Ket{N}(0)
    meta = ("eexact"=>Eexact, "N"=>N, "k"=>k, "eps"=>ϵ, "max_iter"=>max_iter)

    strategies = [
        ("Coeff($ϵ)",       CoeffTruncation(ϵ)),
        ("Weight($k)",       WeightTruncation(k)),
        ("MeanField($k)",    MeanFieldTruncation(k, ψ)),
    ]
    println("running $tag (Eexact=$(round(Eexact,digits=5))) ...")
    for (label, strat) in strategies
        outpath = joinpath(logdir, "$(tag)_$(label).out")
        # capture the verbose=1 formatted table to the .out file
        res = open(outpath, "w") do io
            redirect_stdout(io) do
                DBF.dbf_groundstate(SparsePauliVector(H), ψ;
                    n_body=1, max_iter=max_iter, verbose=1, conv_thresh=1e-8,
                    operator_truncation=strat, gradient_truncation=CoeffTruncation(1e-8),
                    adaptive_truncation=false, compute_var_error=true, energy_lowering_thresh=1e-8)
            end
        end
        println("  wrote ", outpath)
        save_traj(joinpath(logdir, "$(tag)_$(label).csv"), label, res, meta)
    end
    println("done. plot with:  julia --project=. mf_truncation/plot_from_csv.jl $tag")
end

k        = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 3
ϵ        = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1e-2
max_iter = length(ARGS) >= 3 ? parse(Int, ARGS[3])     : 80
zzr      = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0   # ZZ/XX coupling ratio
main(k, ϵ, max_iter, zzr)
