"""
    coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}

Non-mutating version of coeff_clip! — returns a filtered copy.
"""
function coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter(p->abs(p.second) > thresh, ps)
end

function reduce_by_1body(p::PauliBasis{N}, ψ) where N
    out = PauliSum(N)
    # for i in 1:N
    n_terms = length(PauliOperators.get_on_bits(p.z|p.x)) 
    for i in PauliOperators.get_on_bits(p.z|p.x) 
        mask = 1 << (i - 1) 
        tmp1 = PauliBasis{N}(p.z & ~mask, p.x & ~mask)
        tmp2 = PauliBasis{N}(p.z & mask, p.x & mask)
        tmp3 = tmp1*tmp2
        if isapprox(coeff(tmp3), 1) == false || PauliBasis(tmp3) != p
            throw(ErrorException)
        end 
        # println(string(tmp1), "*", string(tmp2), "=", string(tmp1*tmp2))
        out += tmp1 * (expectation_value(tmp2, ψ) / n_terms)
        out += tmp1 * (expectation_value(tmp2, ψ) / n_terms)
        # display(PauliBasis{N}(p.z & ~mask, p.x & ~mask))
    end
    out = out * (1/norm(out))
    # for (p,c) in out
    #     println(string(p), " ", weight(p))
    # end
    return out
end

function meanfield_reduce!(O::PauliSum{N},s, weightclip) where N
    tmp = PauliSum(N)
    for (p,c) in O
        if weight(p) > weightclip 
            tmp += reduce_by_1body(p,s)
            O[p] = 0
        end
    end 
    O += tmp
end


function find_top_k_offdiag(dict, k=10)
    """Optimized for when k << length(dict)"""
    
    # Pre-allocate arrays
    top_keys = Vector{keytype(dict)}(undef, k)
    top_vals = Vector{valtype(dict)}(undef, k) 
    top_abs = Vector{Float64}(undef, k)
    
    n_found = 0
    min_val = 0.0
    min_idx = 1
    
    @inbounds for (key, val) in dict
        key.x != 0 || continue
        abs_val = abs(val)
        
        if n_found < k
            # Still filling up
            n_found += 1
            top_keys[n_found] = key
            top_vals[n_found] = val  
            top_abs[n_found] = abs_val
            
            # Update minimum
            if abs_val < min_val || n_found == 1
                min_val = abs_val
                min_idx = n_found
            end
            
        elseif abs_val > min_val
            # Replace minimum
            top_keys[min_idx] = key
            top_vals[min_idx] = val
            top_abs[min_idx] = abs_val
            
            # Find new minimum
            min_val = top_abs[1]
            min_idx = 1
            for i in 2:k
                if top_abs[i] < min_val
                    min_val = top_abs[i]
                    min_idx = i
                end
            end
        end
    end
    
    # Sort the results
    p = sortperm(view(top_abs, 1:n_found), rev=true)
    return [top_keys[p[i]] => top_vals[p[i]] for i in 1:n_found]
end



# Weight distribution analysis functions are now in PauliOperators:
# get_weight_counts, get_weight_probs, get_majorana_weight_counts, get_majorana_weight_probs
# Note: DBF's get_mweight_counts/get_mweight_probs → PauliOperators' get_majorana_weight_counts/get_majorana_weight_probs

function add_single_excitations(k::Ket{N}) where N
    s = KetSum(N)
    s[k] = 1
    for i in 1:N
        for j in 1:N
            i != j || continue
            c,b = Pauli(N, X=[i, j]) * k
            # count_ones(k.v) == count_ones(b.v) || continue 
            coeff = get(s, b, 0)
            s[b] = coeff + c
        end
    end
    # for (k,c) in s 
    #     @show count_ones(k.v)
    # end
    return s
end

# Base.Matrix(O::PauliSum{N,T}, S::Vector{Ket{N}}) is now in PauliOperators analysis.jl

function Base.Matrix(O::XZPauliSum{T}, basis::Vector{Ket{N}}) where {N,T}
    n = length(basis)
    
    def = Dict{Int128, Float64}()

    M = zeros(ComplexF64,n,n)
    for (i, keti) in enumerate(basis)
        # M[i,i] = expectation_value(O,keti)

        for (j, ketj) in enumerate(basis)
            j >= i || continue
            x = keti.v ⊻ ketj.v
            ox = get(O, x, def)
            for (z,c) in ox
            # for (z,c) in o[x]
                p = PauliBasis{N}(z,x)
                phase,_ = p*ketj
                M[i,j] += phase*c

                j > i || continue
                phase,_  = p*keti
                M[j,i] += phase*c
            end
        end
    end
    return M
end

function Base.Matrix(k::KetSum{N,T}, S::Vector{Ket{N}}) where {N,T}
    nS = length(S)
    v = zeros(T,nS,1)
    length(k) == length(S) || throw(DimensionMismatch)
    for (i,keti) in enumerate(S)
        v[i,1] = k[keti]
    end
    return v
end

# Base.Vector(K::KetSum{N,T}, S::Vector{Ket{N}}) is now in PauliOperators analysis.jl




# Base.sum!(O::KetSum, k::KetSum) is now in PauliOperators addition.jl

# Base.:*(PauliSum, Ket) is now in PauliOperators multiplication.jl

# expectation_value(O::PauliSum, v::KetSum) is now in PauliOperators expectation_value.jl

function PauliOperators.expectation_value(O::XZPauliSum, v::KetSum{N,T}) where {N,T}
    ev = 0
    for (x,zs) in O
        for (z,c) in zs 
            p = PauliBasis{N}(z,x)
            for (k1,c1) in v
                ev += expectation_value(p,k1)*c*c1'*c1
                for (k2,c2) in v
                    k2 != k1 || continue
                    ev += matrix_element(k2', p, k1)*c*c2'*c1
                end
            end
        end
    end
    return ev
end
function PauliOperators.expectation_value(O::XZPauliSum, v::Ket{N}) where {N}
    ev = 0
    haskey(O,0) || return 0.0
    for (z,c) in O[0]
        p = PauliBasis{N}(z,Int128(0))
        ev += expectation_value(p,v)*c
    end
    return ev
end

"""
    pack_x_z(H::PauliSum{N,T}) where {N,T}

Convert PauliSum into a Dict{Int128,Vector{Tuple{Int128,Float64}}}
This allows us to access Pauli's by first specifying `x`, 
then 'z'. 
"""
function pack_x_z(H::PauliSum{N,T}) where {N,T}
    # Build [X][Z] container
    h = Dict{Int128,Vector{Tuple{Int128,T}}}()
    for (p,c) in H
        dx = get(h, p.x, Vector{Tuple{Int128,T}}())
        push!(dx, (p.z,c))
        h[p.x] = dx
    end
    return h
end


"""
    extrapolate_energy(out::Dict; use_per_grad=false, min_points=5, r2_thresh=0.8, verbose=1)

Perform energy-variance extrapolation on the output of `dbf_groundstate`.

Fits both linear and quadratic models to (corrected_variance, corrected_energy) data,
and extrapolates to zero variance. The extrapolated energy is the average of the two
intercepts: E(v=0) = (b₁ + b₂)/2, with uncertainty ±|b₁ - b₂|/2.

The optimal data window is selected by scanning backwards from the final iteration,
choosing the cutoff that minimizes uncertainty while maintaining R² > `r2_thresh`.

# Arguments
- `out::Dict`: output dictionary from `dbf_groundstate`
- `use_per_grad::Bool=false`: if true, use per-macro-iteration data instead of per-rotation
- `min_points::Int=5`: minimum number of data points for fitting
- `r2_thresh::Float64=0.8`: minimum R² for the linear fit
- `verbose::Int=1`: verbosity level (0=silent, 1=summary)

# Returns
A `NamedTuple` with fields:
- `energy`: extrapolated energy at zero variance
- `uncertainty`: half the difference between linear and quadratic intercepts
- `cutoff`: index of the first data point included in the fit
- `r2_linear`: R² of the linear fit
- `fit_linear`: the linear polynomial fit
- `fit_quadratic`: the quadratic polynomial fit
- `corrected_energies`: vector of corrected energies
- `corrected_variances`: vector of corrected variances
"""
function extrapolate_energy(out::Dict;
    use_per_grad::Bool=false,
    min_points::Int=5,
    r2_thresh::Float64=0.8,
    verbose::Int=1)

    if use_per_grad
        energies = out["energies_per_grad"]
        variances = out["variance_per_grad"]
        acc_err = out["accumulated_error_per_grad"]
        acc_var_err = get(out, "accumulated_var_error_per_grad", zeros(length(energies)))
    else
        energies = out["energies"]
        variances = out["variances"]
        acc_err = out["accumulated_error"]
        acc_var_err = out["accumulated_var_error"]
    end

    ce = real.(energies .- acc_err)
    cv = real.(variances .- acc_var_err)

    N = length(ce)
    if N < min_points
        @warn "Only $N data points available, need at least $min_points for fitting."
        return (energy=NaN, uncertainty=NaN, cutoff=1, r2_linear=NaN,
                fit_linear=nothing, fit_quadratic=nothing,
                corrected_energies=ce, corrected_variances=cv)
    end

    best_cutoff = 1
    best_uncertainty = Inf
    best_r2 = -Inf
    fallback_cutoff = 1
    fallback_r2 = -Inf

    for c in (N - min_points + 1):-1:1
        npts = N - c + 1
        npts >= 3 || continue  # need at least 3 points for quadratic fit

        x = cv[c:end]
        y = ce[c:end]

        fit_lin = Polynomials.fit(x, y, 1)
        fit_quad = Polynomials.fit(x, y, 2)

        b1 = fit_lin(0.0)
        b2 = fit_quad(0.0)
        uncertainty = abs(b1 - b2) / 2.0

        # R² for linear fit
        y_pred = fit_lin.(x)
        ss_res = sum((y .- y_pred).^2)
        y_mean = sum(y) / length(y)
        ss_tot = sum((y .- y_mean).^2)
        r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 1.0

        # Track best fallback (highest R²)
        if r2 > fallback_r2
            fallback_r2 = r2
            fallback_cutoff = c
        end

        # Among cutoffs meeting R² threshold, pick minimum uncertainty
        if r2 > r2_thresh && uncertainty < best_uncertainty
            best_uncertainty = uncertainty
            best_r2 = r2
            best_cutoff = c
        end
    end

    # Use fallback if no cutoff met the R² threshold
    if best_uncertainty == Inf
        best_cutoff = fallback_cutoff
        @warn "No cutoff met R² threshold of $r2_thresh. Using best R² cutoff."
    end

    x = cv[best_cutoff:end]
    y = ce[best_cutoff:end]
    fit_lin = Polynomials.fit(x, y, 1)
    fit_quad = Polynomials.fit(x, y, 2)
    b1 = fit_lin(0.0)
    b2 = fit_quad(0.0)
    energy = (b1 + b2) / 2.0
    uncertainty = abs(b1 - b2) / 2.0

    y_pred = fit_lin.(x)
    ss_res = sum((y .- y_pred).^2)
    y_mean = sum(y) / length(y)
    ss_tot = sum((y .- y_mean).^2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 1.0

    if verbose >= 1
        @printf(" Extrapolated energy: %12.8f ± %12.8f\n", energy, uncertainty)
        @printf(" Linear intercept:    %12.8f\n", b1)
        @printf(" Quadratic intercept: %12.8f\n", b2)
        @printf(" R² (linear):         %12.8f\n", r2)
        @printf(" Data window:         %i:%i (%i points)\n", best_cutoff, N, N - best_cutoff + 1)
    end

    return (energy=energy, uncertainty=uncertainty, cutoff=best_cutoff, r2_linear=r2,
            fit_linear=fit_lin, fit_quadratic=fit_quad,
            corrected_energies=ce, corrected_variances=cv)
end

"""
    plot_extrapolation(out::Dict; kwargs...)

Plot energy vs variance with linear/quadratic extrapolation to zero variance.
Requires `using Plots` to be called first (implemented as a package extension).
"""
function plot_extrapolation end

"""
    optimize_rotation_sequence(H::PauliSum{N,T}, generators::Vector{PauliBasis{N}}, ψ::Ket{N};
        initial_angles=zeros(length(generators)), maxiter=100, g_tol=1e-8, verbose=1)

Variationally minimize `⟨ψ|U†HU|ψ⟩` over all rotation angles simultaneously,
where `U = exp(-iθ₁/2 G₁) exp(-iθ₂/2 G₂) ⋯ exp(-iθₙ/2 Gₙ)`.

Uses a mixed Heisenberg/Schrödinger picture: the cost function evolves the state
(Schrödinger), and gradients are computed via a backpropagation-style backward pass.
Optimization is performed with LBFGS.

# Arguments
- `H::PauliSum{N,T}`: the Hamiltonian operator
- `generators::Vector{PauliBasis{N}}`: sequence of Pauli generators defining the rotations
- `ψ::Ket{N}`: reference state

# Keywords
- `initial_angles::Vector{Float64}`: starting angles (default: all zeros)
- `maxiter::Int=100`: maximum LBFGS iterations
- `g_tol::Float64=1e-8`: gradient convergence tolerance
- `verbose::Int=1`: verbosity (0=silent, 1=summary, 2=per-iteration trace)

# Returns
A `NamedTuple` with fields:
- `angles::Vector{Float64}`: optimized rotation angles
- `energy::Float64`: final energy `⟨ψ|U†HU|ψ⟩`
- `result`: the full `Optim.Result` object
"""
function optimize_rotation_sequence(H::PauliSum{N,T}, generators::Vector{PauliBasis{N}}, ψ::Ket{N};
    initial_angles::Vector{Float64}=zeros(length(generators)),
    maxiter::Int=100,
    g_tol::Float64=1e-8,
    verbose::Int=1) where {N,T}

    length(generators) == length(initial_angles) || throw(DimensionMismatch(
        "generators ($(length(generators))) and initial_angles ($(length(initial_angles))) must have same length"))

    Hxz = pack_x_z(H)

    # Heisenberg convention: U_1†...U_n† H U_n...U_1 (generators applied left-to-right)
    # Schrodinger equivalent: |ψ(θ)⟩ = U_n...U_1|ψ⟩ (generators applied in reverse)
    rgens = reverse(generators)

    # Cost function: evolve |ψ⟩ through the rotation sequence, compute ⟨ψ(θ)|H|ψ(θ)⟩
    function _cost(angles)
        rangs = reverse(angles)
        ψt = KetSum(ψ, T=ComplexF64)
        for (gi, θi) in zip(rgens, rangs)
            evolve!(ψt, gi, θi)
        end
        return real(expectation_value(Hxz, ψt))
    end

    # Gradient via backpropagation:
    #   1. Forward: evolve |ψ⟩ → |ψ(θ)⟩  (reversed generator order)
    #   2. Compute |σ⟩ = H|ψ(θ)⟩
    #   3. Backward: unwind rotations (original order), accumulating ∂E/∂θᵢ = Im⟨σ|gᵢ|ψ⟩
    function _gradient(angles)
        rangs = reverse(angles)
        ψt = KetSum(ψ, T=ComplexF64)
        for (gi, θi) in zip(rgens, rangs)
            evolve!(ψt, gi, θi)
        end

        σt = matvec(Hxz, ψt)
        gt = zeros(length(angles))
        idx = 1
        for (gi, θi) in zip(generators, angles)
            gt[idx] = imag(matrix_element(σt, gi, ψt))
            evolve!(ψt, gi, -θi)
            evolve!(σt, gi, -θi)
            idx += 1
        end
        return gt
    end

    options = Optim.Options(
        iterations = maxiter,
        x_reltol = 1e-8,
        f_reltol = 1e-8,
        g_tol = g_tol,
        store_trace = verbose >= 2,
    )

    result = Optim.optimize(_cost, (G, x) -> G .= _gradient(x), initial_angles, Optim.LBFGS(), options)

    if verbose >= 2 && Optim.Options().store_trace
        @printf(" %4s %14s %12s\n", "iter", "energy", "g_norm")
        for t in Optim.trace(result)
            @printf(" %4i %14.8f %12.8f\n", t.iteration, t.value, t.g_norm)
        end
    end

    if verbose >= 1
        @printf(" Optimized energy: %14.8f  (iterations: %i)\n", result.minimum, Optim.iterations(result))
        if Optim.iteration_limit_reached(result)
            @warn "LBFGS iteration limit reached. Consider increasing `maxiter`."
        end
    end

    return (angles=result.minimizer, energy=result.minimum, result=result)
end

# KetSum +/- KetSum are now in PauliOperators addition.jl