# =============================================================================
#  VQE variant of the DBF flow.
#
#  Same outer loop as `dbf_groundstate` -- H is carried forward and truncated
#  each iteration -- but the angles of one iteration's rotations are optimized
#  JOINTLY instead of one at a time.
#
#  `dbf_groundstate` takes the selected generators in gradient order and, for
#  each in turn, finds the angle that minimizes ⟨ψ|H|ψ⟩ for that generator
#  alone, applies it, and moves on. Every angle is therefore chosen against a
#  different operator and none is revisited. Here the whole batch is handed to
#  LBFGS and all n_rots angles are fit together against the current H, which
#  `expectation_value_gradient` makes affordable: one adjoint sweep returns the
#  energy and the derivative with respect to every angle, at ~3x the cost of
#  the energy alone, and the generators need not commute.
# =============================================================================

"""
    optimize_angles(H::PauliSum{N,T}, generators::Vector{PauliBasis{N}}, ψ::Ket{N};
                    initial_angles=zeros(length(generators)),
                    truncation::TruncationStrategy=NoTruncation(),
                    method::Symbol=:dynamic,
                    maxiter=100, g_tol=1e-8, verbose=1) where {N,T}

Fit every angle of a generator sequence at once, minimizing

    C(θ) = ⟨ψ| U_M'⋯U_1' H U_1⋯U_M |ψ⟩,   U_k = exp(-iθ_k/2 G_k)

with the same ordering as `evolve(H, generators, angles)` (generators applied
in index order). Cost and full gradient come from one
`expectation_value_gradient` call per LBFGS step.

`expectation_value_gradient` truncates monotonically -- a Pauli is never
deleted once present -- so the quantity minimized is the monotone-truncated
cost, recoverable as
`expectation_value(H, generators, angles, ψ; truncation, monotone=true)`.

Returns `(angles, energy, result, iters, f_calls)` -- `iters` is the number
of LBFGS optimization cycles, `f_calls` the number of cost/gradient evaluations
(higher, since the line search probes within a cycle).
"""
function optimize_angles(H::PauliSum{N,T}, generators::Vector{PauliBasis{N}}, ψ::Ket{N};
                         initial_angles::Vector{Float64}=zeros(length(generators)),
                         truncation::TruncationStrategy=NoTruncation(),
                         method::Symbol=:dynamic,
                         maxiter::Int=100,
                         g_tol::Float64=1e-8,
                         verbose::Int=1) where {N,T}

    n = length(generators)
    length(initial_angles) == n || throw(DimensionMismatch(
        "generators ($n) and initial_angles ($(length(initial_angles))) must have same length"))
    n == 0 && return (angles=Float64[], energy=real(expectation_value(H, ψ)),
                      result=nothing, iters=0, f_calls=0)

    ncall = Ref(0)
    # cost and gradient arrive together, so the fused form avoids paying for the
    # adjoint sweep twice on every line-search probe
    function fg!(F, G, x)
        ncall[] += 1
        c, g = expectation_value_gradient(H, generators, collect(x), ψ;
                                          truncation=truncation, method=method)
        G === nothing || (G .= g)
        return c
    end

    opts = Optim.Options(iterations=maxiter, g_tol=g_tol, x_reltol=1e-10, f_reltol=1e-12)
    res = Optim.optimize(Optim.only_fg!(fg!), collect(initial_angles), Optim.LBFGS(), opts)

    verbose < 1 || @printf("   VQE: %3d angles  E = %16.10f  (%d iters, %d f/g)\n",
                           n, res.minimum, Optim.iterations(res), ncall[])
    return (angles=res.minimizer, energy=res.minimum, result=res,
            iters=Optim.iterations(res), f_calls=ncall[])
end

"""
    forward_sweep_angles(H, generators, ψ; truncation, verbose)

Initial angles from one forward pass: optimize each generator's angle on its
own against the running operator, in index order, applying it before moving to
the next. This is exactly the `dbf_groundstate` procedure, used here only to
seed the joint optimization (`initialization=:forwardsweep`).
"""
function forward_sweep_angles(H::PauliSum{N,T}, generators::Vector{PauliBasis{N}}, ψ::Ket{N};
                              truncation::TruncationStrategy=NoTruncation(),
                              verbose::Int=0) where {N,T}
    W = deepcopy(H)
    θ = zeros(length(generators))
    for (k, G) in enumerate(generators)
        θ[k], _ = optimize_theta_expval(W, G, ψ; verbose=0)
        evolve!(W, G, θ[k])
        truncation isa NoTruncation || truncate!(W, truncation)
    end
    verbose < 1 || @printf("   forward sweep seed: E = %16.10f\n", real(expectation_value(W, ψ)))
    return θ
end

"""
    dbf_vqe(Oin::PauliSum{N,T}, ψ::Ket{N}; kwargs...) where {N,T}

Each iteration:

1. form the commutator `[H, P]` with the n-body Z-projector `P`;
2. score its terms by the energy gradient at `ψ` and keep the best `n_rots`;
3. fit all `n_rots` angles jointly (`optimize_angles`), started either at zero
   (`initialization=:zero`) or from a single independent-angle forward sweep
   (`initialization=:forwardsweep`);
4. apply the whole sequence to H and truncate.

# Keywords
- `n_body=1`, `max_iter=10`, `n_rots=10`
- `operator_truncation`, `gradient_truncation`, `energy_lowering_thresh`
- `initialization=:zero` | `:forwardsweep`
- `opt_maxiter=100`, `g_tol=1e-8`, `conv_thresh=1e-6`, `compute_var_error=true`

Returns `out::Dict` with `energies`, `accumulated_error`, `variances`,
`generators`, `angles`, `grad_norms`, `f_calls`, `hamiltonian`, `H0`.
"""
function dbf_vqe(Oin::PauliSum{N,T}, ψ::Ket{N};
                 n_body::Int=1,
                 max_iter::Int=10,
                 n_rots::Int=10,
                 operator_truncation::TruncationStrategy=CoeffTruncation(1e-6),
                 gradient_truncation::TruncationStrategy=CoeffTruncation(1e-6),
                 energy_lowering_thresh::Float64=1e-8,
                 initialization::Symbol=:zero,
                 opt_maxiter::Int=100,
                 g_tol::Float64=1e-8,
                 conv_thresh::Float64=1e-6,
                 compute_var_error::Bool=true,
                 method::Symbol=:dynamic,
                 verbose::Int=1) where {N,T}

    initialization in (:zero, :forwardsweep) || throw(ArgumentError(
        "initialization must be :zero or :forwardsweep, got :$initialization"))

    O = deepcopy(Oin)
    P = create_0_projector(N, n_body)
    corr = compute_var_error ? EnergyVarianceCorrection(ψ) : EnergyCorrection(ψ)
    corr.accumulated_energy = 0.0
    compute_var_error && (corr.accumulated_variance = 0.0)
    accumulated_norm_error = 0.0
    ecurr = real(expectation_value(O, ψ))

    if verbose >= 1
        println("\n ===== dbf_vqe parameters =====")
        @printf("   %-24s %s\n", "N (qubits)", string(N))
        @printf("   %-24s %s\n", "coeff type", string(T))
        @printf("   %-24s %s\n", "len(H0)", string(length(Oin)))
        @printf("   %-24s %s\n", "n_body", string(n_body))
        @printf("   %-24s %s\n", "max_iter", string(max_iter))
        @printf("   %-24s %s\n", "n_rots", string(n_rots))
        @printf("   %-24s %s\n", "initialization", string(initialization))
        @printf("   %-24s %s\n", "opt_maxiter", string(opt_maxiter))
        @printf("   %-24s %s\n", "g_tol", string(g_tol))
        @printf("   %-24s %s\n", "conv_thresh", string(conv_thresh))
        @printf("   %-24s %s\n", "operator_truncation", string(operator_truncation))
        @printf("   %-24s %s\n", "gradient_truncation", string(gradient_truncation))
        @printf("   %-24s %s\n", "energy_lowering_thresh", string(energy_lowering_thresh))
        @printf("   %-24s %s\n", "compute_var_error", string(compute_var_error))
        println(" ======================================\n")
    end

    out = Dict()
    out["state"] = ψ
    out["H0"] = Oin
    out["energies"] = Vector{Float64}([])
    out["variances"] = Vector{Float64}([])
    out["accumulated_error"] = Vector{Float64}([])
    out["accumulated_var_error"] = Vector{Float64}([])
    out["norms"] = Vector{Float64}([])
    out["norm_error"] = Vector{Float64}([])
    out["generators"] = Vector{PauliBasis{N}}([])
    out["angles"] = Vector{Float64}([])
    out["grad_norms"] = Vector{Float64}([])
    out["f_calls"] = Vector{Int}([])
    out["opt_iters"] = Vector{Int}([])

    push!(out["energies"], ecurr)
    push!(out["variances"], real(variance(O, ψ)))
    push!(out["accumulated_error"], 0.0)
    push!(out["accumulated_var_error"], 0.0)
    push!(out["norms"], norm(O))
    push!(out["norm_error"], 0.0)

    verbose < 1 || @printf(" %6s", "Iter")
    verbose < 1 || @printf(" %14s", "<ψ|H|ψ>")
    verbose < 1 || @printf(" %12s", "total_error")
    verbose < 1 || @printf(" %12s", "norm_err")
    verbose < 1 || @printf(" %9s", "norm(G)")
    verbose < 1 || @printf(" %10s", "len([H,Z])")
    verbose < 1 || @printf(" %8s", "len(G)")
    verbose < 1 || @printf(" %8s", "len(H)")
    verbose < 1 || @printf(" %4s", "#Rot")
    verbose < 1 || @printf(" %5s", "#Opt")
    verbose < 1 || @printf(" %10s", "variance")
    if compute_var_error
        verbose < 1 || @printf(" %12s", "var_error")
    end
    verbose < 1 || @printf(" %8s", "Entropy")
    verbose < 1 || @printf(" %8s", "Time")
    verbose < 1 || @printf("\n")

    for iter in 1:max_iter
        t_iter = time_ns()

        # 1. commutator pool
        G = commutator_clipped(P, O)
        len_comm = length(G)
        truncate!(G, gradient_truncation)
        if length(G) == 0
            verbose < 1 || @warn "No search direction found. Loosen `gradient_truncation`."
            break
        end

        # 2. sort by |0> energy gradient, keep n_rots. Ties break on the
        # generator's (z,x) so the choice is independent of iteration order.
        grad_vec = Vector{Float64}([])
        grad_ops = Vector{PauliBasis{N}}([])
        σv = matvec(O, ψ)
        compute_gradient!(grad_vec, grad_ops, G, σv, ψ, energy_lowering_thresh)
        if isempty(grad_vec)
            verbose < 1 || @warn "All candidate gradients below energy_lowering_thresh."
            break
        end
        len_g = length(grad_vec)
        sort_keys = [(-round(abs(grad_vec[i]), sigdigits=12), grad_ops[i].z, grad_ops[i].x)
                     for i in eachindex(grad_vec)]
        pick = partialsortperm(sort_keys, 1:min(n_rots, length(sort_keys)))
        gens = grad_ops[pick]
        gnorm = norm(grad_vec[pick])

        # 3. joint angle optimization over the whole batch
        θ0 = initialization === :zero ? zeros(length(gens)) :
             forward_sweep_angles(O, gens, ψ; truncation=operator_truncation,
                                  verbose=verbose - 2)
        r = optimize_angles(O, gens, ψ; initial_angles=θ0,
                            truncation=operator_truncation, method=method,
                            maxiter=opt_maxiter, g_tol=g_tol, verbose=verbose - 2)

        # 4. apply the sequence and truncate. Truncating after each rotation
        # (rather than once at the end of the layer) keeps H from growing
        # mid-layer; `corr` accumulates the energy dropped, and since a
        # rotation preserves the coefficient 2-norm exactly, any change in
        # ||O|| is truncation.
        for (Gk, θk) in zip(gens, r.angles)
            n1 = norm(O)
            evolve!(O, Gk, θk)
            truncate!(O, operator_truncation, corr)
            n2 = norm(O)
            accumulated_norm_error += n2^2 - n1^2
        end

        eprev = ecurr
        ecurr = real(expectation_value(O, ψ))
        var_curr = real(variance(O, ψ))

        append!(out["generators"], gens)
        append!(out["angles"], r.angles)
        push!(out["energies"], ecurr)
        push!(out["variances"], var_curr)
        push!(out["accumulated_error"], real(corr.accumulated_energy))
        push!(out["accumulated_var_error"],
              compute_var_error ? real(corr.accumulated_variance) : 0.0)
        push!(out["norms"], norm(O))
        push!(out["norm_error"], accumulated_norm_error)
        push!(out["grad_norms"], gnorm)
        push!(out["f_calls"], r.f_calls)
        push!(out["opt_iters"], r.iters)

        verbose < 1 || @printf("*%6i", iter)
        verbose < 1 || @printf(" %14.8f", ecurr)
        verbose < 1 || @printf(" %12.8f", real(corr.accumulated_energy))
        verbose < 1 || @printf(" %12.8f", accumulated_norm_error)
        verbose < 1 || @printf(" %8.3e", gnorm)
        verbose < 1 || @printf(" %10i", len_comm)
        verbose < 1 || @printf(" %8i", len_g)
        verbose < 1 || @printf(" %8i", length(O))
        verbose < 1 || @printf(" %4i", length(gens))
        verbose < 1 || @printf(" %5i", r.iters)
        verbose < 1 || @printf(" %10.6f", var_curr)
        if compute_var_error
            verbose < 1 || @printf(" %12.8f", real(corr.accumulated_variance))
        end
        verbose < 1 || @printf(" %8.4f", entropy(O))
        verbose < 1 || @printf(" %8.2f", (time_ns() - t_iter) / 1e9)
        verbose < 1 || @printf("\n")
        flush(stdout)

        if abs(ecurr - eprev) < conv_thresh
            verbose < 1 || println(" Converged.")
            break
        end
    end

    out["hamiltonian"] = O
    return out
end

"""
    dbf_vqe_from_h0(Oin::PauliSum{N,T}, ψ::Ket{N}; kwargs...) where {N,T}

Variant kept for comparison with [`dbf_vqe`](@ref). H is **not** carried
forward: only the generator list and its angles are kept, and each iteration
re-derives the operator from `H0` and re-optimizes *every* angle accumulated so
far, not just the current batch.

Two differences follow. Angles chosen early keep moving as the list grows, and
an energy evaluation is a single truncated evolution from `H0` rather than a
chain of per-rotation truncations — so truncation error does not compound
across iterations.

The cost is that the evolution runs through all `M` rotations every time, and
that operator grows geometrically in `M`: measured on naphthalene at
`CoeffTruncation(1e-6)`, ~1.17x per generator — 7,151 terms at `M=0` reaching
2.3M by `M=40`, with one gradient call then taking 7.6 s. `dbf_vqe` has no such
growth because it truncates its carried operator every iteration. Useful for
shallow sequences and for checking how much the greedy angle choice costs.

Keywords as `dbf_vqe`, with `adds_per_iter` in place of `n_rots` and
`warm_start=true` seeding the optimizer with the previous angles.
"""
function dbf_vqe_from_h0(Oin::PauliSum{N,T}, ψ::Ket{N};
                         n_body::Int=1,
                         max_iter::Int=20,
                         adds_per_iter::Int=1,
                         operator_truncation::TruncationStrategy=NoTruncation(),
                         gradient_truncation::TruncationStrategy=CoeffTruncation(1e-6),
                         energy_lowering_thresh::Float64=1e-8,
                         opt_maxiter::Int=100,
                         g_tol::Float64=1e-8,
                         conv_thresh::Float64=1e-6,
                         warm_start::Bool=true,
                         method::Symbol=:dynamic,
                         verbose::Int=1) where {N,T}

    H0 = deepcopy(Oin)
    P  = create_0_projector(N, n_body)
    generators = PauliBasis{N}[]; angles = Float64[]
    ecurr = real(expectation_value(H0, ψ))

    out = Dict{String,Any}()
    out["H0"] = H0; out["state"] = ψ
    out["energies"] = Float64[ecurr]
    out["n_params"] = Int[0]; out["grad_norms"] = Float64[]; out["f_calls"] = Int[]

    verbose < 1 || @printf(" %5s %16s %13s %9s %7s %8s %8s %7s\n",
            "Iter", "<ψ|H|ψ>", "ΔE", "norm(g)", "#param", "len(H)", "f/g", "Time")

    for iter in 1:max_iter
        t0 = time_ns()
        O = isempty(generators) ? H0 :
            evolve(H0, generators, angles; truncation=operator_truncation)

        G = commutator_clipped(P, O); truncate!(G, gradient_truncation)
        if length(G) == 0
            verbose < 1 || @warn "No search direction found."
            break
        end
        grad_vec = Float64[]; grad_ops = PauliBasis{N}[]
        σv = matvec(O, ψ)
        compute_gradient!(grad_vec, grad_ops, G, σv, ψ, energy_lowering_thresh)
        if isempty(grad_vec)
            verbose < 1 || @warn "All gradients below threshold."
            break
        end

        keys_ = [(-round(abs(grad_vec[i]), sigdigits=12), grad_ops[i].z, grad_ops[i].x)
                 for i in eachindex(grad_vec)]
        pick = partialsortperm(keys_, 1:min(adds_per_iter, length(keys_)))
        for gi in pick; push!(generators, grad_ops[gi]); push!(angles, 0.0); end
        gnorm = norm(grad_vec[pick])

        x0 = warm_start ? angles : zeros(length(generators))
        r = optimize_angles(H0, generators, ψ; initial_angles=x0,
                            truncation=operator_truncation, method=method,
                            maxiter=opt_maxiter, g_tol=g_tol, verbose=verbose - 1)
        angles = collect(r.angles)
        eprev, ecurr = ecurr, r.energy

        push!(out["energies"], ecurr); push!(out["n_params"], length(generators))
        push!(out["grad_norms"], gnorm); push!(out["f_calls"], r.f_calls)
        verbose < 1 || @printf(" %5i %16.10f %13.4e %9.3e %7i %8i %8i %6.2fs\n",
                iter, ecurr, ecurr - eprev, gnorm, length(generators), length(O),
                r.f_calls, (time_ns() - t0) / 1e9)
        if abs(ecurr - eprev) < conv_thresh
            verbose < 1 || println(" Converged: |ΔE| < conv_thresh")
            break
        end
    end

    out["generators"] = generators; out["angles"] = angles
    out["hamiltonian"] = isempty(generators) ? H0 :
        evolve(H0, generators, angles; truncation=operator_truncation)
    return out
end
