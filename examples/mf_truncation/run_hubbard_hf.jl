# =============================================================================
#  Fold vs drop in the UHF-MO basis (HF determinant → |0…0⟩), saving trajectories.
#  Uses hubbard_hf_basis.jl (adopts the doped-Hubbard LNO integrals→Pauli+PH
#  pipeline). Records the reference FCI overlap |⟨HF|GS⟩|² in the CSV meta.
#
#  Run (from DBF/examples): julia --project=. mf_truncation/run_hubbard_hf.jl [k] [eps] [max_iter] [U]
# =============================================================================

using PauliOperators, DBF, LinearAlgebra, Printf
include(joinpath(@__DIR__, "hubbard_hf_basis.jl"))

function save_traj(path, label, res, meta)
    E=real.(res["energies_per_grad"]); V=real.(res["variance_per_grad"])
    ae=real.(res["accumulated_error_per_grad"]); av=real.(res["accumulated_var_error_per_grad"])
    m=min(length(E),length(V),length(ae),length(av))
    open(path,"w") do io
        println(io, "# label=$label ", join(["$k=$v" for (k,v) in meta]," "), " nterms=", length(res["hamiltonian"]))
        println(io, "igrad,energy,variance,acc_err,acc_var_err")
        for i in 1:m; @printf(io, "%d,%.10g,%.10g,%.10g,%.10g\n", i, E[i], V[i], ae[i], av[i]); end
    end
    println("  wrote ", path)
end

function main(k, ϵ, max_iter, U)
    L = 6
    tag = "hubbardHFL$(L)U$(U)_k$(k)"
    logdir = joinpath(@__DIR__,"logs"); mkpath(logdir)
    H, na, nb, n = build_hf_hamiltonian(L, U)
    F = eigen(Hermitian(Matrix(H))); Eexact = F.values[1]; overlap = abs2(F.vectors[1,1])
    ψ = Ket{n}(0)
    @printf("HF basis: U=%.3f  E_HF=%.6f  E_exact=%.6f  |<HF|GS>|^2=%.6f  nterms(H)=%d\n",
            U, real(expectation_value(H,ψ)), Eexact, overlap, length(H))
    meta = ("eexact"=>Eexact, "overlap"=>overlap, "N"=>n, "U"=>U, "k"=>k)
    strategies = [
        ("Coeff($ϵ)",             CoeffTruncation(ϵ)),
        ("MajoranaWeight($k)",    MajoranaWeightTruncation(k)),
        ("MajoranaMeanField($k)", MajoranaMeanFieldTruncation(k, ψ)),
    ]
    println("running $tag ...")
    for (label, strat) in strategies
        outp = joinpath(logdir,"$(tag)_$(label).out")
        res = open(outp,"w") do io
            redirect_stdout(io) do
                DBF.dbf_groundstate(SparsePauliVector(H), ψ; n_body=1, max_iter=max_iter, verbose=1,
                    conv_thresh=1e-6, operator_truncation=strat, gradient_truncation=CoeffTruncation(1e-8),
                    adaptive_truncation=false, compute_var_error=true, energy_lowering_thresh=1e-6)
            end
        end
        println("  wrote ", outp)
        save_traj(joinpath(logdir,"$(tag)_$(label).csv"), label, res, meta)
    end
    println("done $tag.")
end

k=length(ARGS)>=1 ? parse(Int,ARGS[1]) : 4
ϵ=length(ARGS)>=2 ? parse(Float64,ARGS[2]) : 1e-2
mi=length(ARGS)>=3 ? parse(Int,ARGS[3]) : 40
U=length(ARGS)>=4 ? parse(Float64,ARGS[4]) : 2.0
main(k,ϵ,mi,U)
