using JLD2

"""
    optimize_theta_expval(O::PauliSum{N,T}, G::PauliBasis{N}, ψ::Ket{N}; stepsize=.001, verbose=1) where {N,T}

Find the optimal θ that minimizes `<ψ|exp(iθ/2 G) O exp(-iθ/2 G)|ψ>`

Return the optimal angle, as well as the continious function that maps θ to the expectation value.
"""
function optimize_theta_expval(O::PauliSum{N,T}, G::PauliBasis{N}, ψ::Ket{N}; verbose=1) where {N,T}
    cg,ψg = G*ψ
    Oeval = expectation_value(O, ψ)
    OGeval = matrix_element(ψ', O, ψg)*cg
    GOGeval = expectation_value(O, ψg)*cg'*cg
    function cost(θ)
        # Cost function for <ψ| U(θ)' O U(θ)|ψ>
        return real(cos(θ/2)^2 * Oeval + sin(θ/2)^2 * GOGeval - 2im*cos(θ/2)*sin(θ/2)*OGeval)
    end
    
    options = Optim.Options(
        x_reltol = 1e-12, # A tight relative tolerance for changes in the solution vector
        f_reltol = 1e-12, # A tight relative tolerance for changes in the objective function value
        g_tol = 1e-10,    # A tighter absolute tolerance for the gradient
    )
    result = optimize(cost, 0.0, 2π)
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
    dbf_eval(Oin::PauliSum{N,T}, ψ::Ket{N}; 
    max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
    evolve_coeff_thresh=1e-12) where {N,T}

d/dt H = [H,[H,P]]

where P = |000...><000...| = equal sum of all diagonal paulis
"""
function dbf_groundstate(Oin::PauliSum{N,T}, ψ::Ket{N}; 
            initial_error = 0,
            initial_norm_error = 0,
            max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
            evolve_coeff_thresh=1e-12,
            evolve_weight_thresh=nothing,
            grad_coeff_thresh=1e-8,
            grad_weight_thresh=nothing,
            energy_lowering_thresh=1e-3,
            max_rots_per_grad = 100,
            clifford_check = false,
            compute_pt2_error = false,
            checkfile=nothing) where {N,T}
       

    if grad_weight_thresh === nothing
        grad_weight_thresh = N
    end
    if evolve_weight_thresh === nothing
        evolve_weight_thresh = N
    end

    O = deepcopy(Oin)
    generators = Vector{PauliBasis{N}}([])
    angles = Vector{Float64}([])

    ecurr = expectation_value(O, ψ) 
    accumulated_error = initial_error 
    accumulated_pt2_error = 0 
    accumulated_norm_error = initial_norm_error 
        
    verbose < 2 || println("\n Compute PT2 correction")
    e0, e2 = pt2(O, ψ)
   
    # 
    # Initialize data collection
    out = Dict()
    out["energies"] = Vector{Float64}([])
    out["accumulated_error"] = Vector{Float64}([])
    out["norms"] = Vector{Float64}([])
    out["generators"] = Vector{PauliBasis{N}}([])
    out["angles"] = Vector{Float64}([])
    
    out["energies_per_grad"] = Vector{Float64}([])
    out["accumulated_error_per_grad"] = Vector{Float64}([])
    out["norms_per_grad"] = Vector{Float64}([])
    out["pt2_per_grad"] = Vector{Float64}([])
    out["variance_per_grad"] = Vector{Float64}([])
    
    push!(out["energies"], ecurr)
    push!(out["accumulated_error"], initial_error)
    push!(out["norms"], norm(O))
    
    push!(out["energies_per_grad"], ecurr)
    push!(out["accumulated_error_per_grad"], initial_error)
    push!(out["pt2_per_grad"], e2)
    push!(out["variance_per_grad"], variance(O,ψ))
    push!(out["norms_per_grad"], norm(O))
   
    verbose < 1 || @printf(" %6s", "Iter")
    verbose < 1 || @printf(" %14s", "<ψ|H|ψ>")
    verbose < 1 || @printf(" %12s", "total_error")
    if compute_pt2_error
        verbose < 1 || @printf(" %12s", "PT_error")
    end
    verbose < 1 || @printf(" %10s", "E(2)")
    verbose < 1 || @printf(" %8s", "norm_err")
    verbose < 1 || @printf(" %8s", "norm(G)")
    verbose < 1 || @printf(" %10s", "len([H,Z])")
    verbose < 1 || @printf(" %8s", "len(G)")
    verbose < 1 || @printf(" %8s", "len(H)")
    verbose < 1 || @printf(" %4s", "#Rot")
    verbose < 1 || @printf(" %8s", "variance")
    verbose < 1 || @printf(" %8s", "Entropy")
    verbose < 1 || @printf(" %8s", "Time")
    verbose < 1 || @printf("\n")

    for iter in 1:max_iter
        
       
        # Create the iteration dependent pool
        pool = commute_with_Zs(O)

        len_comm = length(pool)
        verbose < 2 || @printf(" length of commutator: %i\n", len_comm)
        coeff_clip!(pool, thresh=grad_coeff_thresh)
        weight_clip!(pool, grad_weight_thresh)
        # pool = find_top_k(pool, search_n_top)
       
        if length(pool) == 0
            @warn " No search direction found. Increase `n_top` or decrease `clip`"
            break
        end

        grad_vec = Vector{Float64}([])
        grad_ops = Vector{PauliBasis{N}}([])
      
        xzO = pack_x_z(O)
        σv = matvec(xzO, ψ)

        # Compute gradient vector
        for (p,c) in pool
            # dyad = (ψ * ψ') * p'
            # grad_vec[pi] = 2*imag(expectation_value(O,dyad))
            ci, σ = p*ψ
            gi = 2*real(get(σv, σ, T(0)) * c * ci)
            # gi = 2*real(matrix_element(σ', O, ψ)*c*ci)
            # @show expectation_value(O*p*c - c*p*O, ψ)
            if abs(gi) > grad_coeff_thresh
                push!(grad_vec, gi)
                push!(grad_ops, p)
            end
        end
        
        
        sorted_idx = reverse(sortperm(abs.(grad_vec)))

        verbose < 2 || @printf("     %8s %12s %12s", "pool idx", "||O||", "<ψ|H|ψ>")
        verbose < 2 || @printf(" %12s %12s", "len(O)", "θi")
        verbose < 2 || @printf("\n")
        n_rots = 0
        time = @elapsed for gi in sorted_idx
            
            G = grad_ops[gi]
            θi, costi = DBF.optimize_theta_expval(O, G, ψ, verbose=0)
         
            if clifford_check
                # See if we can do a cheap clifford operation
                if costi(0) - costi(π / 2) > energy_lowering_thresh
                    θi = π / 2
                    println("clifford:", string(G))
                end
            end

            #
            # make sure energy lowering is large enough to warrent evolving
            costi(0) - costi(θi) > energy_lowering_thresh || continue


            O = evolve(O,G,θi)

            e1 = expectation_value(O,ψ)
            n1 = norm(O)
            pt2_1 = 0
            pt2_2 = 0
            if compute_pt2_error
                _, pt2_1 = pt2(O, ψ)
            end
            
            #
            # Truncate operator
            coeff_clip!(O, thresh=evolve_coeff_thresh)
            weight_clip!(O, evolve_weight_thresh)
            e2 = expectation_value(O,ψ)
            n2 = norm(O)
            if compute_pt2_error
                _, pt2_2 = pt2(O, ψ)
            end

            accumulated_error += e2 - e1
            accumulated_pt2_error += pt2_2 - pt2_1
            accumulated_norm_error += n2^2 - n1^2
            
            push!(out["accumulated_error"], real(accumulated_error))
            push!(out["energies"], ecurr)
            push!(out["norms"], n2)
            push!(out["generators"], G) 
            push!(out["angles"], θi)
            ecurr = expectation_value(O, ψ) 
            verbose < 2 || @printf("     %8i %12.8f %12.8f", gi, norm(O), ecurr)
            verbose < 2 || @printf(" %12i %12.8f %s", length(O), θi, string(G))
            verbose < 2 || @printf("\n")
            n_rots += 1
            flush(stdout)

            if n_rots >= max_rots_per_grad
                break
            end
        end
        verbose < 2 || println("\n Compute PT2 correction")
        e0, e2 = pt2(O, ψ)
        verbose < 2 || @printf(" E0 = %12.8f E2 = %12.8f EPT2 = %12.8f \n", e0, e2, e0+e2)
        
        var_curr = variance(O,ψ)
        verbose < 1 || @printf("*%6i", iter)
        verbose < 1 || @printf(" %14.8f", ecurr)
        verbose < 1 || @printf(" %12.8f", real(accumulated_error))
        if compute_pt2_error
            verbose < 1 || @printf(" %12.8f", real(accumulated_pt2_error))
        end
        verbose < 1 || @printf(" %10.6f", real(e2))
        verbose < 1 || @printf(" %8.1e", accumulated_norm_error)
        verbose < 1 || @printf(" %8.1e", norm(grad_vec))
        verbose < 1 || @printf(" %10.1e", len_comm)
        verbose < 1 || @printf(" %8.1e", length(grad_vec))
        verbose < 1 || @printf(" %8i", length(O))
        verbose < 1 || @printf(" %4i", n_rots)
        verbose < 1 || @printf(" %8.4f", real(var_curr))
        verbose < 1 || @printf(" %8.4f", entropy(O))
        verbose < 1 || @printf(" %8.2f", time)
        verbose < 1 || @printf("\n")
        
        push!(out["pt2_per_grad"], e2)
        push!(out["accumulated_error_per_grad"], accumulated_error)
        push!(out["energies_per_grad"], ecurr)
        push!(out["variance_per_grad"], var_curr)
        push!(out["norms_per_grad"], norm(O))

        if checkfile !== nothing
            @save "$(checkfile).jld2" O out
        end
    
        
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
        coeff_clip!(out, thresh=thresh)
        sum!(out_tot, out)
        coeff_clip!(out_tot, thresh=thresh)
    end
    return out_tot
end



"""
    dbf_eval(Oin::PauliSum{N,T}, ψ::Ket{N}; 
    max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
    evolve_coeff_thresh=1e-12) where {N,T}

d/dt H = [H,[H,P]]

where P = |000...><000...| = equal sum of all diagonal paulis
"""
function groundstate_diffeq(Oin::PauliSum{N,T}, ψ::Ket{N}; 
            n_body = 2,
            max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
            evolve_coeff_thresh=1e-12,
            evolve_weight_thresh=nothing,
            grad_coeff_thresh=1e-8,
            grad_weight_thresh=nothing,
            stepsize = .01) where {N,T}
        
    if grad_weight_thresh === nothing
        grad_weight_thresh = N
    end
    if evolve_weight_thresh === nothing
        evolve_weight_thresh = N
    end

    O = deepcopy(Oin)
    generators = Vector{PauliBasis{N}}([])
    angles = Vector{Float64}([])
    norm_old = norm(offdiag(O))

    ecurr = expectation_value(O, ψ) 
    accumulated_error = 0
   
    # Define the source operator that is an n-body approximation to |00><00|
    S = create_0_projector(N,n_body)
    @printf(" Number of terms in approx projector: %i\n", length(S))
    
    verbose < 1 || @printf(" %6s", "Iter")
    verbose < 1 || @printf(" %12s", "<ψ|H|ψ>")
    verbose < 1 || @printf(" %12s", "||<[H,Gi]>||")
    verbose < 1 || @printf(" %12s", "total_error")
    verbose < 1 || @printf(" %12s", "E(2)")
    verbose < 1 || @printf(" %12s", "|H|")
    verbose < 1 || @printf(" %8s", "#PoolOps")
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
        coeff_clip!(pool, thresh=grad_coeff_thresh)
        # weight_clip!(pool, grad_weight_thresh)
        # pool = find_top_k(pool, search_n_top)
       
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
          
            

            O = evolve(O,gi,θi*stepsize)
            e1 = expectation_value(O,ψ)
            coeff_clip!(O, thresh=evolve_coeff_thresh)
            weight_clip!(O, evolve_weight_thresh)
            e2 = expectation_value(O,ψ)

            accumulated_error += e2 - e1
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
        verbose < 1 || @printf(" %12.8f", real(accumulated_error))
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