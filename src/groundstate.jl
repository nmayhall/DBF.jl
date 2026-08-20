using JLD2
using TimerOutputs

"""
    optimize_theta_expval(O::PauliSum{N,T}, G::PauliBasis{N}, ψ::Ket{N}; stepsize=.001, verbose=1) where {N,T}

Find the optimal θ that minimizes `<ψ|exp(iθ/2 G) O exp(-iθ/2 G)|ψ>`

Return the optimal angle, as well as the continious function that maps θ to the expectation value.
"""
function optimize_theta_expval(O::AnyPauliSum{N,T}, G::PauliBasis{N}, ψ::Ket{N}; verbose=1) where {N,T}
    cg,ψg = G*ψ
    Oeval = expectation_value(O, ψ)
    OGeval = matrix_element(ψ', O, ψg)*cg
    GOGeval = expectation_value(O, ψg)*cg'*cg
    function cost(θ)
        # Cost function for <ψ| U(θ)' O U(θ)|ψ>
        return real(cos(θ/2)^2 * Oeval + sin(θ/2)^2 * GOGeval - 2im*cos(θ/2)*sin(θ/2)*OGeval)
    end
    
    # Tight tolerances so θ is reproducible well below coefficient-truncation
    # thresholds (Brent's default rel_tol of √eps ≈ 1.5e-8 lets fp noise in the
    # inputs shift θ enough to flip downstream threshold decisions)
    result = optimize(cost, 0.0, 2π, Optim.Brent(); rel_tol=1e-12, abs_tol=1e-12)
    # result = optimize(negative_cost, [0.0, π], Brent())
    # result = optimize(negative_cost, [0.0, π], LBFGS())
    θ = result.minimizer
    # f_min = result.minimum

    if Optim.iteration_limit_reached(result)
        @show Optim.abs_tol(result), Optim.rel_tol(result)
        @warn " minimization failed"
    end

    # Make sure bounds are respected
    θ < 2π || throw(DomainError)
    θ > 0 || throw(DomainError)

    # if cost(θ) > cost(0)
    #     @warn " optimal θ worse than zero" θ cost(θ)  cost(0) cost(θ) - cost(0) "resetting"
    #     θ = 0
    # end
    stepsize = 1e-5
    # idx = argmax([cost(i*π) for i in 0:stepsize:1-stepsize])
    # θ = (idx-1) * stepsize * π
    if cost(θ+stepsize) < cost(θ) || cost(θ-stepsize) < cost(θ)
        @show cost(θ+stepsize) , cost(θ) , cost(θ-stepsize), θ
        @show cost(θ+stepsize) - cost(θ)
        @show cost(θ-stepsize) - cost(θ)
        throw(ErrorException) 
    end
    
    verbose < 1 || @show θ, sqrt(cost(θ))
    return θ, cost
    
    # idx = argmin([cost(i*2π) for i in 0:stepsize:1-stepsize])
    # # for i in 0:stepsize:1-stepsize
    # #     @show i*2π, cost(i*2π)
    # # end
    # θ = (idx-1) * stepsize * 2π
    # return θ, cost
end



"""
    compute_gradient!(grad_vec, grad_ops, G, σv, ψ, thresh)

Fill `grad_vec`/`grad_ops` with the gradient components `2 Re(⟨σ|H|ψ⟩ c cᵢ)`
of `G`'s terms whose magnitude exceeds `thresh`.

This lives in its own function as a type-stability barrier: inlined under
`@elapsed`/`@timeit` in `dbf_groundstate`, the loop variables get boxed and
every iteration pays dynamic dispatch plus a heap-allocated pair — that was
~80% of the gradient section's cost in profiling.
"""
function compute_gradient!(grad_vec::Vector{Float64}, grad_ops::Vector{PauliBasis{N}},
                           G::AnyPauliSum{N,T}, σv, ψ::Ket{N}, thresh) where {N,T}
    for (p, c) in G
        ci, σ = p * ψ
        gi = 2 * real(get(σv, σ, T(0)) * c * ci)
        if abs(gi) > thresh
            push!(grad_vec, gi)
            push!(grad_ops, p)
        end
    end
    return nothing
end

"""
    dbf_groundstate(Oin::PauliSum{N,T}, ψ::Ket{N};
        operator_truncation=CoeffTruncation(1e-6),
        gradient_truncation=CoeffTruncation(1e-6),
        energy_lowering_thresh=1e-6,
        ...) where {N,T}

Transform `Oin` so that computational basis state `ψ` becomes the ground state,
minimizing `⟨ψ|H|ψ⟩`. Uses an n-body Z-projector approximation as the source operator.

# Truncation
- `operator_truncation`: `TruncationStrategy` applied to H after each rotation
- `gradient_truncation`: `TruncationStrategy` applied to the gradient/commutator

Both accept any `TruncationStrategy` from PauliOperators (e.g., `CoeffTruncation`,
`WeightTruncation`, `CompositeTruncation`, etc.).

# Checkpointing
`checkfile=nothing` (default) disables checkpointing. Otherwise
`<checkfile>.jld2` is rewritten every iteration with a replay payload: `H0`
(stored as a plain `PauliSum`), `state`, `generators`, `angles` and the
per-rotation / per-iteration histories -- the same keys as the returned `out`
except that the live operator's scratch buffers are never written.

# Timing
The `Time` column is the wall time of the full iteration (commutator through
checkpoint); the `TimerOutputs` table printed at the end breaks it down.
"""
# =============================================================================
#  Resumable operator snapshots
# =============================================================================
#
#  `checkfile` stores a REPLAY record -- H0 plus the (generator, angle)
#  sequence -- so restarting from it means re-running every rotation. For the
#  tight-truncation runs that is a large fraction of the run being resumed.
#  These snapshots instead capture the flowed operator itself, so a job stopped
#  by a walltime limit continues from where it stopped.
#
#  A SparsePauliVector is written as its LIVE SLICES ONLY (z[1:n], x[1:n],
#  c[1:n]). Serializing the struct whole would also write the append, scratch
#  and workspace buffers -- several times the live term count -- and converting
#  to PauliSum would pay to build a Dict of ~10^8 entries. Both are avoided.
#  The slices are already in the canonical sorted order the type maintains, and
#  are restored in that same order, so the invariant survives the round trip.

_snapshot_operator(O::SparsePauliVector{N,W,T}) where {N,W,T} =
    Dict{String,Any}("kind" => "spv", "N" => N, "n" => O.n,
                     "z" => O.z[1:O.n], "x" => O.x[1:O.n], "c" => O.c[1:O.n])

_snapshot_operator(O::PauliSum) = Dict{String,Any}("kind" => "psum", "op" => O)

function _restore_operator(p::Dict)
    p["kind"] == "psum" && return p["op"]
    n = p["n"]::Int
    v = SparsePauliVector(p["N"]::Int, eltype(p["c"]); capacity = max(n, 1))
    copyto!(v.z, 1, p["z"], 1, n)
    copyto!(v.x, 1, p["x"], 1, n)
    copyto!(v.c, 1, p["c"], 1, n)
    v.n = n
    return v
end

"""
    save_resume_state(path, O, corr, out, iter, acc)

Atomically write a resumable snapshot. The write goes to `path * ".tmp"` and is
then renamed, so a job killed mid-write leaves the previous good snapshot
intact rather than a truncated file.

`out["H0"]` is dropped: it is re-supplied from `Oin` by the resuming call, and
storing a SparsePauliVector H0 would drag its scratch buffers along.
"""
function save_resume_state(path, O, corr, out, iter, acc)
    hist = copy(out)                 # shallow: shares the history arrays
    delete!(hist, "H0")
    tmp = path * ".tmp"
    jldsave(tmp;
            op   = _snapshot_operator(O),
            out  = hist,
            iter = iter,
            acc  = acc,
            accumulated_energy   = real(corr.accumulated_energy),
            accumulated_variance = hasproperty(corr, :accumulated_variance) ?
                                   real(corr.accumulated_variance) : 0.0)
    mv(tmp, path; force = true)
    return path
end


function dbf_groundstate(Oin::AnyPauliSum{N,T}, ψ::Ket{N};
            n_body=1,
            initial_error = 0,
            initial_norm_error = 0,
            max_iter=10,
            verbose=1,
            conv_thresh=1e-3,
            operator_truncation::TruncationStrategy=CoeffTruncation(1e-6),
            gradient_truncation::TruncationStrategy=CoeffTruncation(1e-6),
            adaptive_truncation=false,
            energy_lowering_thresh=1e-6,
            max_rots_per_grad = 100,
            clifford_check = false,
            compute_var_error = true,
            compute_pt2 = false,
            compute_pt2_error = false,
            checkfile=nothing,
            resume_file=nothing,
            resume_stride=100) where {N,T}

    # the pt2-error probes imply computing pt2
    compute_pt2 |= compute_pt2_error

    # Document the exact settings of this run in the output stream
    if verbose >= 1
        println("\n ===== dbf_groundstate parameters =====")
        @printf("   %-24s %s\n", "N (qubits)", string(N))
        @printf("   %-24s %s\n", "coeff type", string(T))
        @printf("   %-24s %s\n", "len(H0)", string(length(Oin)))
        @printf("   %-24s %s\n", "n_body", string(n_body))
        @printf("   %-24s %s\n", "max_iter", string(max_iter))
        @printf("   %-24s %s\n", "conv_thresh", string(conv_thresh))
        @printf("   %-24s %s\n", "operator_truncation", string(operator_truncation))
        @printf("   %-24s %s\n", "gradient_truncation", string(gradient_truncation))
        @printf("   %-24s %s\n", "adaptive_truncation", string(adaptive_truncation))
        @printf("   %-24s %s\n", "energy_lowering_thresh", string(energy_lowering_thresh))
        @printf("   %-24s %s\n", "max_rots_per_grad", string(max_rots_per_grad))
        @printf("   %-24s %s\n", "clifford_check", string(clifford_check))
        @printf("   %-24s %s\n", "compute_var_error", string(compute_var_error))
        @printf("   %-24s %s\n", "compute_pt2", string(compute_pt2))
        @printf("   %-24s %s\n", "compute_pt2_error", string(compute_pt2_error))
        @printf("   %-24s %s\n", "initial_error", string(initial_error))
        @printf("   %-24s %s\n", "initial_norm_error", string(initial_norm_error))
        @printf("   %-24s %s\n", "checkfile", string(checkfile))
        println(" ======================================\n")
    end

    O = deepcopy(Oin)
    generators = Vector{PauliBasis{N}}([])
    angles = Vector{Float64}([])

    to = TimerOutput()

    ecurr = expectation_value(O, ψ)

    # Set up correction accumulator for truncation error tracking
    corr = compute_var_error ? EnergyVarianceCorrection(ψ) : EnergyCorrection(ψ)
    corr.accumulated_energy = Float64(initial_error)
    if compute_var_error
        corr.accumulated_variance = 0.0
    end
    accumulated_pt2_error = 0
    accumulated_norm_error = initial_norm_error
        
    e0, e2 = 0.0, 0.0
    if compute_pt2
        verbose < 2 || println("\n Compute PT2 correction")
        @show e0, e2 = pt2(O, ψ)
    end
   
    # 
    # Initialize data collection
    out = Dict()

    out["state"] = ψ
    out["H0"] = Oin
    out["energies"] = Vector{Float64}([])
    out["variances"] = Vector{Float64}([])
    out["accumulated_error"] = Vector{Float64}([])
    out["accumulated_var_error"] = Vector{Float64}([])
    out["norms"] = Vector{Float64}([])
    out["generators"] = Vector{PauliBasis{N}}([])
    out["angles"] = Vector{Float64}([])
    
    out["energies_per_grad"] = Vector{Float64}([])
    out["accumulated_error_per_grad"] = Vector{Float64}([])
    out["norms_per_grad"] = Vector{Float64}([])
    out["pt2_per_grad"] = Vector{Float64}([])
    out["variance_per_grad"] = Vector{Float64}([])
    out["accumulated_var_error_per_grad"] = Vector{Float64}([])

    push!(out["energies"], ecurr)
    push!(out["variances"], variance(O,ψ))
    push!(out["accumulated_error"], initial_error)
    push!(out["accumulated_var_error"], initial_error)
    push!(out["norms"], norm(O))
    
    push!(out["energies_per_grad"], ecurr)
    push!(out["accumulated_error_per_grad"], initial_error)
    push!(out["pt2_per_grad"], real(e2))
    push!(out["variance_per_grad"], variance(O,ψ))
    push!(out["accumulated_var_error_per_grad"], compute_var_error ? real(corr.accumulated_variance) : 0.0)
    push!(out["norms_per_grad"], norm(O))
   
    verbose < 1 || @printf(" %6s", "Iter")
    verbose < 1 || @printf(" %14s", "<ψ|H|ψ>")
    verbose < 1 || @printf(" %12s", "total_error")
    if compute_pt2_error
        verbose < 1 || @printf(" %12s", "PT_error")
    end
    if compute_pt2
        verbose < 1 || @printf(" %10s", "E(2)")
    end
    verbose < 1 || @printf(" %12s", "norm_err")
    verbose < 1 || @printf(" %9s", "norm(G)")
    verbose < 1 || @printf(" %10s", "len([H,Z])")
    verbose < 1 || @printf(" %8s", "len(G)")
    verbose < 1 || @printf(" %8s", "len(H)")
    verbose < 1 || @printf(" %4s", "#Rot")
    verbose < 1 || @printf(" %10s", "variance")
    if compute_var_error
        verbose < 1 || @printf(" %12s", "var_error")
    end
    verbose < 1 || @printf(" %8s", "Entropy")
    verbose < 1 || @printf(" %8s", "Time")
    verbose < 1 || @printf("\n")

    P = create_0_projector(N, n_body)
    # Match the projector's container to the input so commutator_clipped
    # dispatches to the SparsePauliVector kernel
    Oin isa SparsePauliVector && (P = SparsePauliVector(P; T=T))

    # Resume from a previous snapshot, if one is there. This must happen before
    # `out_ckpt` is built below -- that is a shallow copy, and rebinding `out`
    # afterwards would leave it pointing at the discarded history arrays.
    iter_start = 1
    if resume_file !== nothing && isfile(resume_file)
        r = load(resume_file)
        O = _restore_operator(r["op"])
        out = r["out"]
        out["H0"] = Oin              # dropped from the snapshot on purpose
        iter_start = r["iter"] + 1
        corr.accumulated_energy = r["accumulated_energy"]
        compute_var_error && (corr.accumulated_variance = r["accumulated_variance"])
        acc = r["acc"]
        accumulated_norm_error = acc.norm_error
        accumulated_pt2_error  = acc.pt2_error
        ecurr = acc.ecurr
        if verbose >= 1
            @printf("\n RESUMED from %s at iteration %d  (len(H) = %d, E = %.8f)\n",
                    resume_file, iter_start, length(O), real(ecurr))
        end
    end

    # Checkpoint payload: what is needed to replay the trajectory -- H0, the
    # state and the (generator, angle) sequence plus the small per-rotation
    # histories. It shares the history arrays with `out` (shallow copy), so
    # saving it each iteration is up to date, but H0 is stored once as a plain
    # PauliSum: a SparsePauliVector serializes all of its scratch buffers
    # (~8x the live terms; 25 GB for a 65M-term H), which made the per-
    # iteration @save the dominant cost of large runs.
    out_ckpt = nothing
    if checkfile !== nothing
        out_ckpt = copy(out)
        out_ckpt["H0"] = Oin isa SparsePauliVector ? PauliSum(Oin) : Oin
    end

    # Norm of O after the most recent truncation. A Pauli rotation preserves
    # the coefficient 2-norm exactly, so this is also the pre-truncation norm
    # of the next rotation — no need to recompute it after each evolve!
    n_curr = norm(O)

    operator_truncation_save = deepcopy(operator_truncation)

    for iter in iter_start:max_iter
        
        # Wall time of the whole iteration (commutator through checkpoint);
        # this is what the "Time" column reports.
        t_iter = time_ns()
       
        # Create the iteration dependent pool
        @timeit to "commutator" G = commutator_clipped(P,O)
        
        len_comm = length(G)
        verbose < 2 || @printf(" length of commutator: %i\n", len_comm)
        @timeit to "clip" truncate!(G, gradient_truncation)
       
        if length(G) == 0
            @warn " No search direction found. Loosen `gradient_truncation`."
            break
        end

        grad_vec = Vector{Float64}([])
        grad_ops = Vector{PauliBasis{N}}([])
      
        @timeit to "matvec" σv = matvec(O, ψ)

        # Compute gradient vector
        @timeit to "gradient" compute_gradient!(
            grad_vec, grad_ops, G, σv, ψ, energy_lowering_thresh)
        
        
        # Descending |gradient|, quantized to 12 significant digits so that
        # symmetry-degenerate gradients (equal in exact arithmetic, split only
        # by fp rounding noise) compare as exact ties regardless of kernel /
        # build / summation-order changes; ties then break canonically by the
        # generator's (z,x) identity, independent of container iteration order.
        # Keys are precomputed (sort's `by` runs per COMPARISON, and round
        # with sigdigits is expensive), and only the top max_rots_per_grad
        # entries are selected -- the rotation loop below consumes exactly
        # that many and never skips, so a full sort of the pool is wasted.
        # NOTE: if the loop ever re-grows a skip/continue branch, widen this.
        @timeit to "sort" begin
            sort_keys = [(-round(abs(grad_vec[i]), sigdigits=12),
                          grad_ops[i].z, grad_ops[i].x) for i in eachindex(grad_vec)]
            sorted_idx = partialsortperm(sort_keys,
                                         1:min(max_rots_per_grad, length(sort_keys)))
        end
        
        verbose < 2 || @printf("     %8s %12s %12s", "G idx", "||O||", "<ψ|H|ψ>")
        verbose < 2 || @printf(" %12s %12s", "len(O)", "θi")
        verbose < 2 || @printf("\n")
        n_rots = 0
        for gi in sorted_idx
            
            Gi = grad_ops[gi]
            @timeit to "opt_theta" θi, costi = DBF.optimize_theta_expval(O, Gi, ψ, verbose=0)

            if adaptive_truncation
                operator_truncation = CoeffTruncation(operator_truncation_save.thresh * abs(sin(θi)))
            end

            if clifford_check
                # See if we can do a cheap clifford operation
                if costi(0) - costi(π / 2) > energy_lowering_thresh
                    θi = π / 2
                end
            end

            # #
            # # make sure energy lowering is large enough to warrent evolving
            # costi(0) - costi(θi) > energy_lowering_thresh || continue


            n1 = n_curr   # == norm(O) after evolve!, by unitarity
            pt2_1 = 0
            pt2_2 = 0

            # Rotate and truncate. For SparsePauliVector the sequence evolve!
            # folds the truncation filter into the merge pass (one sweep
            # instead of merge + separate compaction); PauliOperators
            # documents window = 1 as exactly equivalent to
            # evolve!(O, Gi, θi); truncate!(O, strategy, corr).
            # The pre-truncation pt2 probe needs the unfused path.
            if O isa SparsePauliVector && !compute_pt2_error
                @timeit to "evolve" evolve!(O, [Gi], [θi]; window=1,
                                            truncation=operator_truncation,
                                            correction=corr)
            else
                @timeit to "evolve" evolve!(O, Gi, θi)
                if compute_pt2_error
                    @timeit to "pt2" _, pt2_1 = pt2(O, ψ)
                end
                @timeit to "clip" truncate!(O, operator_truncation, corr)
            end

            @timeit to "expval" e2 = expectation_value(O,ψ)
            n2 = norm(O)
            n_curr = n2
            if compute_pt2_error
                @timeit to "pt2" _, pt2_2 = pt2(O, ψ)
            end

            accumulated_pt2_error += pt2_2 - pt2_1
            accumulated_norm_error += n2^2 - n1^2

            ecurr = e2
            verbose < 2 || @printf("     %8i %12.8f %12.8f", gi, norm(O), ecurr)
            verbose < 2 || @printf(" %12i %12.8f %s", length(O), θi, string(G))
            verbose < 2 || @printf("\n")
            n_rots += 1
            flush(stdout)
            
            push!(out["accumulated_error"], real(corr.accumulated_energy))
            push!(out["accumulated_var_error"], compute_var_error ? real(corr.accumulated_variance) : 0.0)
            push!(out["energies"], ecurr)
            if compute_var_error
                @timeit to "variance" push!(out["variances"], real(variance(O, ψ)))
            end
            push!(out["norms"], n2)
            push!(out["generators"], Gi) 
            push!(out["angles"], θi)

            if n_rots >= max_rots_per_grad
                break
            end
        end
        if compute_pt2
            verbose < 2 || println("\n Compute PT2 correction")
            @timeit to "pt2" e0, e2 = pt2(O, ψ)
            verbose < 2 || @printf(" E0 = %12.8f E2 = %12.8f EPT2 = %12.8f \n", e0, e2, e0+e2)
        end

        @timeit to "variance" var_curr = variance(O,ψ)
        verbose < 1 || @printf("*%6i", iter)
        verbose < 1 || @printf(" %14.8f", ecurr)
        verbose < 1 || @printf(" %12.8f", real(corr.accumulated_energy))
        if compute_pt2_error
            verbose < 1 || @printf(" %12.8f", real(accumulated_pt2_error))
        end
        if compute_pt2
            verbose < 1 || @printf(" %10.6f", real(e2))
        end
        verbose < 1 || @printf(" %12.8f", accumulated_norm_error)
        verbose < 1 || @printf(" %8.3e", norm(grad_vec))
        verbose < 1 || @printf(" %10i", len_comm)
        verbose < 1 || @printf(" %8i", length(grad_vec))
        verbose < 1 || @printf(" %8i", length(O))
        verbose < 1 || @printf(" %4i", n_rots)
        verbose < 1 || @printf(" %10.6f", real(var_curr))
        if compute_var_error
            verbose < 1 || @printf(" %12.8f", compute_var_error ? real(corr.accumulated_variance) : 0.0)
        end
        verbose < 1 || @printf(" %8.4f", @timeit(to, "entropy", entropy(O)))
        # Time column is filled in below, after the checkpoint save
        
        push!(out["pt2_per_grad"], real(e2))
        push!(out["accumulated_error_per_grad"], corr.accumulated_energy)
        push!(out["energies_per_grad"], ecurr)
        push!(out["variance_per_grad"], var_curr)
        push!(out["accumulated_var_error_per_grad"], compute_var_error ? real(corr.accumulated_variance) : 0.0)
        push!(out["norms_per_grad"], norm(O))

        # Checkpoint `out` only: it holds H0, the state, and the full
        # (generator, angle) sequence, so the current operator is exactly
        # reconstructible by replaying the rotations with the same
        # truncation -- no need to pay GB-scale writes for O every iteration.
        if checkfile !== nothing
            @timeit to "checkpoint" @save "$(checkfile).jld2" out=out_ckpt
        end

        # Resumable snapshot: strided, because it writes the operator itself
        # rather than the small replay record above.
        if resume_file !== nothing && (iter % resume_stride == 0 || iter == max_iter)
            @timeit to "resume_snapshot" save_resume_state(
                resume_file, O, corr, out, iter,
                (norm_error = accumulated_norm_error,
                 pt2_error  = accumulated_pt2_error,
                 ecurr      = ecurr,
                 var_curr   = var_curr))
        end

        verbose < 1 || @printf(" %8.2f", (time_ns() - t_iter) / 1e9)
        verbose < 1 || @printf("\n")
        flush(stdout)

        if norm(grad_vec) < conv_thresh
            verbose < 1 || @printf(" Converged.\n")
            break
        end

        if iter == max_iter
            verbose < 1 || @printf(" Not Converged.\n")
        end
        
        if n_rots == 0
            @warn """ No search directions found. 
                    Tighten `grad_coeff_thresh` or `energy_lowering_thresh`"""
            break
        end
        
    end
    out["hamiltonian"] = O 
    show(to) 
    println() 
    return out 
end

function commute_with_Zs(O::PauliSum{N}; thresh=1e-12) where N
    out_tot = PauliSum(N)
   
    for i in 1:N
        zi = PauliBasis(Pauli(N, Z=[i]))
        
        out = PauliSum(N)
        sizehint!(out, min(1000, length(O)//2)) # assume half commute

        for (p, c) in O
            
            !PauliOperators.commute(zi,p) || continue
            # out += c*(zi*p - p*zi) 
            zp = zi*p 
            curr = get(out, PauliBasis(zp), 0.0) 
            out[PauliBasis(zp)] = curr + 2*coeff(zp)*c
        end
        coeff_clip!(out, thresh)
        sum!(out_tot, out)
        coeff_clip!(out_tot, thresh)
    end
    return out_tot
end

"""
    commutator_clipped(O1::PauliSum{N}, O2::PauliSum{N}; thresh=1e-12)

Compute [O1, O2] with intermediate coefficient clipping for numerical stability
with large operators. Unlike PauliOperators.commutator, this clips after each
outer-loop iteration to control intermediate term growth.
"""
function commutator_clipped(O1::PauliSum{N}, O2::PauliSum{N}; thresh=1e-12) where N
    out_tot = PauliSum(N)
   
    for (p1, c1) in O1
        
        out = PauliSum(N)
        sizehint!(out, min(1000, length(O2)//2)) # assume half commute

        for (p2, c2) in O2
            
            !PauliOperators.commute(p1,p2) || continue
            p3 = p1*p2 
            curr = get(out, PauliBasis(p3), 0.0) 
            out[PauliBasis(p3)] = curr + 2*coeff(p3)*c1*c2
        end
        coeff_clip!(out, thresh)
        sum!(out_tot, out)
        coeff_clip!(out_tot, thresh)
    end
    return out_tot
end

"""
    commutator_clipped(O1::SparsePauliVector{N,W,T}, O2::SparsePauliVector{N,W,T}; thresh=1e-12)

SparsePauliVector method of `commutator_clipped`, restructured around a key
property: for a fixed left term `p1`, the product key map
`p2 ↦ (z₂⊻z₁, x₂⊻x₁)` is a bijection, so a single-term partial commutator
never contains duplicate keys. The per-`O1`-term clip can therefore be
applied per triple at generation time (exactly equivalent to the `PauliSum`
method's per-term `coeff_clip!` of the deduplicated partial), and all
surviving triples are collected into one buffer for a single sort and a
single duplicate-summing merge, instead of |O1| sorts and |O1| incremental
merges into a growing accumulator.

The only semantic difference from the `PauliSum` method is that the running
total is not re-clipped between `O1` terms; this only matters for
accumulated coefficients at the `thresh` (default 1e-12) dust level, far
below any physically meaningful `gradient_truncation`.
"""
function commutator_clipped(O1::SparsePauliVector{N,W,T}, O2::SparsePauliVector{N,W,T}; thresh=1e-12) where {N,W,T}
    O1.an == 0 && O2.an == 0 || error("commutator_clipped requires merged operands (no pending appends)")

    out_tot = SparsePauliVector(N, T, capacity=max(16, 2 * O2.n))
    A = SparsePauliVector(N, T, capacity=1)
    A.n = 1
    pair_ws = Vector{Tuple{W,W,T}}(undef, max(16, O2.n))

    ws = out_tot.ws
    m = 0
    @inbounds for i in 1:O1.n
        A.z[1] = O1.z[i]
        A.x[1] = O1.x[i]
        A.c[1] = O1.c[i]
        mi, ovf = PauliOperators._commutator_triples!(pair_ws, A, O2, false)
        ovf && error("commutator pair workspace overflow — this is a bug")
        for j in 1:mi
            t = pair_ws[j]
            abs(t[3]) > thresh || continue
            m += 1
            m > length(ws) && resize!(ws, max(2 * length(ws), m))
            ws[m] = t
        end
    end
    PauliOperators._sort_ws!(ws, 1, m)
    PauliOperators._merge_spv!(out_tot, m, PauliOperators._compile_filter(CoeffTruncation(Float64(thresh))))
    return out_tot
end



"""
    groundstate_diffeq(Oin::PauliSum{N,T}, ψ::Ket{N};
        operator_truncation=CoeffTruncation(1e-12),
        gradient_truncation=CoeffTruncation(1e-8),
        ...) where {N,T}

d/dt H = [H,[H,P]]

where P = |000...><000...| = equal sum of all diagonal paulis
"""
function groundstate_diffeq(Oin::PauliSum{N,T}, ψ::Ket{N};
            n_body = 2,
            max_iter=10, verbose=1, conv_thresh=1e-3,
            operator_truncation::TruncationStrategy=CoeffTruncation(1e-12),
            gradient_truncation::TruncationStrategy=CoeffTruncation(1e-8),
            stepsize = .01) where {N,T}

    O = deepcopy(Oin)
    generators = Vector{PauliBasis{N}}([])
    angles = Vector{Float64}([])
    norm_old = norm(offdiag(O))

    ecurr = expectation_value(O, ψ)
    corr = EnergyCorrection(ψ)

    # Define the source operator that is an n-body approximation to |00><00|
    S = create_0_projector(N,n_body)
    @printf(" Number of terms in approx projector: %i\n", length(S))
    
    verbose < 1 || @printf(" %6s", "Iter")
    verbose < 1 || @printf(" %12s", "<ψ|H|ψ>")
    verbose < 1 || @printf(" %12s", "||<[H,Gi]>||")
    verbose < 1 || @printf(" %12s", "total_error")
    verbose < 1 || @printf(" %12s", "E(2)")
    verbose < 1 || @printf(" %12s", "|H|")
    verbose < 1 || @printf(" %8s", "#Grad_Ops")
    verbose < 1 || @printf(" %4s", "#Rot")
    verbose < 1 || @printf(" %8s", "len(H)")
    verbose < 1 || @printf(" %12s", "variance")
    verbose < 1 || @printf(" %12s", "Sh Entropy")
    verbose < 1 || @printf("\n")

    for iter in 1:max_iter
        
       
        # Create the iteration dependent pool
        # pool = max_of_commutator2(S, O, n_top=search_n_top)
        pool = S*O - O*S
        # pool = commute_with_Zs(O)
        truncate!(pool, gradient_truncation)
       
        if length(pool) == 0
            @warn " No search direction found. Increase `n_top` or decrease `clip`"
            break
        end

        grad_vec = Vector{Float64}([])
        grad_ops = Vector{PauliBasis{N}}([])
       
        # # @show norm(pool), norm(pool*O - O*pool)
        # # Compute gradient vector
        for (p,c) in pool
            # dyad = (ψ * ψ') * p'
            # grad_vec[pi] = 2*imag(expectation_value(O,dyad))
            # ci, σ = p*ψ
            # gi = 2*real(matrix_element(σ', O, ψ)*c*ci)
            # # @show expectation_value(O*p*c - c*p*O, ψ)
            # if abs(gi) > grad_coeff_thresh
                push!(grad_vec, -imag(c))
                push!(grad_ops, p)
            # end
        end
        # for (p,c) in O*pool - pool*O
        #     @show p,c
        # end
       
        # @show length(pool), norm(grad_vec)
        # n_pool = length(grad_vec)
        
        
        norm_new = norm(grad_vec)
        
        sorted_idx = reverse(sortperm(abs.(grad_vec)))

        verbose < 2 || @printf("     %8s %12s %12s", "pool idx", "||O||", "<ψ|H|ψ>")
        verbose < 2 || @printf(" %12s %12s", "len(O)", "θi")
        verbose < 2 || @printf("\n")
        n_rots = 0
        for i in sorted_idx
            
            gi = grad_ops[i]
            θi = grad_vec[i]
            # θi, costi = DBF.optimize_theta_expval(O, G, ψ, verbose=0)
           
            # #
            # # make sure energy lowering is large enough to warrent evolving
            # costi(0) - costi(θi) > grad_coeff_thresh || continue

            # n_rots < search_n_top || break 
            #See if we can do a cheap clifford operation
            # if costi(0) - costi(π/2) > evolve_coeff_thresh 
            #     θi = π/2
            #     @warn "clifford", costi(0) - costi(π/2)
            # end 
          
            

            O = PauliOperators.evolve(O,gi,θi*stepsize)
            truncate!(O, operator_truncation, corr)
            # if norm_new - costi(θi) > 1e-12
            #     @show norm_new - costi(θi)
            #     throw(ErrorException)
            # end
            # norm_new = costi(θi)/O_norm
            ecurr = expectation_value(O, ψ) 
            verbose < 2 || @printf("     %8i %12.8f %12.8f", gi, norm(O), ecurr)
            verbose < 2 || @printf(" %12i %12.8f %s", length(O), θi, string(G))
            verbose < 2 || @printf("\n")
            push!(generators, gi)
            push!(angles, θi)
            n_rots += 1
            flush(stdout)
        end
        verbose < 2 || println("\n Compute PT2 correction")
        e0, e2 = pt2(O, ψ)
        verbose < 2 || @printf(" E0 = %12.8f E2 = %12.8f EPT2 = %12.8f \n", e0, e2, e0+e2)
        
        var_curr = variance(O,ψ)
        verbose < 1 || @printf("*%6i", iter)
        verbose < 1 || @printf(" %12.8f", ecurr)
        verbose < 1 || @printf(" %12.8f", norm_new)
        verbose < 1 || @printf(" %12.8f", real(corr.accumulated_energy))
        verbose < 1 || @printf(" %12.8f", real(e2))
        verbose < 1 || @printf(" %12.8f", norm(O))
        verbose < 1 || @printf(" %8i", length(pool))
        verbose < 1 || @printf(" %4i", n_rots)
        verbose < 1 || @printf(" %8i", length(O))
        verbose < 1 || @printf(" %12.8f", real(var_curr))
        verbose < 1 || @printf(" %12.8f", entropy(O))
        verbose < 1 || @printf("\n")
        
        if norm_new < conv_thresh
            verbose < 1 || @printf(" Converged.\n")
            break
        end

       
        if iter == max_iter
            verbose < 1 || @printf(" Not Converged.\n")
        end
        
        if n_rots == 0
            @warn """ No search directions found. 
                    Tighten `grad_coeff_thresh` or expand pool"""
            break
        end
        
        norm_old = norm_new
    end
    return O, generators, angles
end

function create_0_projector(N, n_body)
    S = PauliSum(N)
    S += Pauli(N)
    for i in 1:N
        S += Pauli(N, Z=[i])
        # S += Pauli(N, X=[i])

        n_body > 1 || continue

        for j in i+1:N
            S += Pauli(N, Z=[i, j])

            n_body > 2 || continue

            for k in j+1:N
                S += Pauli(N, Z=[i, j, k])

                n_body > 3 || continue

                for l in k+1:N
                    S += Pauli(N, Z=[i, j, k, l])
                
                    n_body > 4 || continue

                    for m in l+1:N
                        S += Pauli(N, Z=[i, j, k, l, m])
                        
                        n_body > 5 || continue

                        for n in m+1:N
                            S += Pauli(N, Z=[i, j, k, l, m, n])
                        end
                    end
                end
            end
        end
    end
    # The real approx projector would have a factor of 2^-N, but we'll ignore that constant
    # which basically amounts to a scaled up stepsize when integrating. 
    # S = S * (1/2^N)
    
    return S
end