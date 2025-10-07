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

TBW
"""
function dbf_groundstate_old(Oin::PauliSum{N,T}, ψ::Ket{N}; 
            max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
            evolve_coeff_thresh=1e-12,
            evolve_weight_thresh=20,
            search_n_top=100,
            extra_diag=nothing) where {N,T}
    O = deepcopy(Oin)
    generators = Vector{PauliBasis}([])
    angles = Vector{Float64}([])
    norm_old = norm(offdiag(O))

    G_old = Pauli(N)


    verbose < 1 || @printf(" %6s %12s %12s", "Iter", "|O|", "<ψ|O|ψ>")
    verbose < 1 || @printf(" %12s %12s %12s G\n", "norm(g)", "len(O)", "θ")
    for iter in 1:max_iter
        
        S = diag(O)
        
        # add extra diagonal operators to help 
        if extra_diag !== nothing
            S += extra_diag
        end
        
        # com = O*S - S*O
        SO_OS = max_of_commutator2(S, O, n_top=search_n_top)
        
        if length(SO_OS) == 0
            @warn " No search direction found. Increase `n_top` or decrease `clip`"
            break
        end

        # coeff_clip!(O, thresh=evolve_coeff_thresh)
        # com = O*com - com*O
        # f(k) = expectation_value(k*O-O*k, ψ)
        function f(k)
            # <ψ| k*O - O*k |psi> = imag( tr(O|ψ><ψ|k)) 
            dyad = (ψ * ψ') * k'
            return 2*imag(expectation_value(O,dyad))
        end
        # G = argmax(k -> abs(f(k)), keys(SO_OS))
        G = argmax(k -> abs(f(k) * SO_OS[k]), keys(SO_OS))
        
        # for k in keys(SO_OS)
        #     # SO_OS[k] = abs(expectation_value(k*O-O*k, ψ)*SO_OS[k])
        #     SO_OS[k] = abs(f(k)*SO_OS[k])
        # end
        
        # G = argmax(k->abs(SO_OS[k]), keys(SO_OS))

        norm_new = norm(SO_OS)
        # G = argmax(k -> abs(com[k]), keys(com))

        if G == G_old
            println(" Trapped? ", string(G), " ", coeff)
            θi, costi = DBF.optimize_theta_expval(O, G, ψ, verbose=1)
            # θi, costi = DBF.optimize_theta_expval(O, G, ψ, stepsize=.000001, verbose=1)
            step = .1
            for i in 0:.01:1
                θ = i*step*2π
                @printf(" θ=%12.8f cost=%12.8f\n", θ, costi(θ))
            end
            break
        end
        
        # θi, costi = DBF.optimize_theta_expval(O, G, ψ, stepsize=.000001, verbose=0)
        θi, costi = DBF.optimize_theta_expval(O, G, ψ, verbose=0)
        O = evolve(O,G,θi)
        coeff_clip!(O, thresh=evolve_coeff_thresh)

        # if norm_new - costi(θi) > 1e-12
        #     @show norm_new - costi(θi)
        #     throw(ErrorException)
        # end
        # norm_new = costi(θi)/O_norm
        ecurr = expectation_value(O, ψ) 
        verbose < 1 || @printf(" %6i %12.8f %12.8f %12.8f", iter, norm(O), ecurr, norm_new)
        verbose < 1 || @printf(" %12i %12.8f %s", length(O), θi, string(G))
        verbose < 1 || @printf("\n")
        push!(generators, G)
        push!(angles, θi)

        # if norm_new - norm_old < conv_thresh
        if norm_new < conv_thresh
            verbose < 1 || @printf(" Converged.\n")
            break
        end
       
        # if norm_new > norm_old
        #     println(" Norm increased?")
        #     throw(ErrorException)
        # end
        if iter == max_iter
            verbose < 1 || @printf(" Not Converged.\n")
        end
        
        norm_old = norm_new
        G_old = G
    end
    return O, generators, angles
end

"""
    dbf_eval(Oin::PauliSum{N,T}, ψ::Ket{N}; 
    max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
    evolve_coeff_thresh=1e-12) where {N,T}

d/dt H = [H,[H,P]]

where P = |000...><000...| = equal sum of all diagonal paulis
"""
function dbf_groundstate(Oin::PauliSum{N,T}, ψ::Ket{N}; 
            n_body = 2,
            max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
            evolve_coeff_thresh=1e-12,
            evolve_weight_thresh=20,
            grad_coeff_thresh=1e-8,
            grad_weight_thresh=10,
            search_n_top=1000,
            extra_diag=nothing) where {N,T}
    O = deepcopy(Oin)
    generators = Vector{PauliBasis}([])
    angles = Vector{Float64}([])
    norm_old = norm(offdiag(O))

    ecurr = expectation_value(O, ψ) 
    accumulated_error = 0
   
    # Define the source operator that is an n-body approximation to |00><00|
    S = PauliSum(N)
    for i in 1:N 
        S += Pauli(N, Z=[i])

        n_body > 1 || continue

        for j in i+1:N 
            S += Pauli(N, Z=[i,j])
        
            n_body > 2 || continue

            for k in j+1:N 
                S += Pauli(N, Z=[i,j,k])
            
                n_body > 3 || continue

                for l in k+1:N 
                    S += Pauli(N, Z=[i,j,k,l])
                
                    n_body == 4 || throw("NotYetImplemented") 
                    # n_body > 3 || continue

                end
            end
        end
    end
    # S = S * (1/length(S))
    
    verbose < 1 || @printf(" %6s", "Iter")
    verbose < 1 || @printf(" %12s", "<ψ|H|ψ>")
    verbose < 1 || @printf(" %12s", "||<[H,Gi]>||")
    verbose < 1 || @printf(" %12s", "total_error")
    verbose < 1 || @printf(" %12s", "|H|")
    verbose < 1 || @printf(" %8s", "#PoolOps")
    verbose < 1 || @printf(" %4s", "#Rot")
    verbose < 1 || @printf(" %8s", "len(H)")
    verbose < 1 || @printf(" %12s", "variance")
    verbose < 1 || @printf(" %12s", "Sh Entropy")
    verbose < 1 || @printf("\n")


    for iter in 1:max_iter
        
       
        # Create the iteration dependent pool
        pool = max_of_commutator2(S, O, n_top=search_n_top)
        # pool = S*O - O*S
        coeff_clip!(pool, thresh=grad_coeff_thresh)
        # weight_clip!(pool, grad_weight_thresh)
        pool = find_top_k(pool, search_n_top)

        if length(pool) == 0
            @warn " No search direction found. Increase `n_top` or decrease `clip`"
            break
        end

        grad_vec = Vector{Float64}([])
        grad_ops = Vector{PauliBasis{N}}([])
       
        # @show norm(pool), norm(pool*O - O*pool)
        # Compute gradient vector
        for (p,c) in pool
            dyad = (ψ * ψ') * p'
            # grad_vec[pi] = 2*imag(expectation_value(O,dyad))
            ci, σ = p*ψ
            gi = 2*real(matrix_element(σ', O, ψ)*c*ci)
            # @show expectation_value(O*p*c - c*p*O, ψ)
            if abs(gi) > grad_coeff_thresh
                push!(grad_vec, gi)
                push!(grad_ops, p)
            end
        end
        # for (p,c) in O*pool - pool*O
        #     @show p,c
        # end
       
        # @show length(pool), norm(grad_vec)
        n_pool = length(grad_vec)
        
        
        norm_new = norm(grad_vec)
        
        sorted_idx = reverse(sortperm(abs.(grad_vec)))

        verbose < 2 || @printf("     %8s %12s %12s", "pool idx", "||O||", "<ψ|H|ψ>")
        verbose < 2 || @printf(" %12s %12s %s", "len(O)", "θi", string(G))
        verbose < 2 || @printf("\n")
        n_rots = 0
        for gi in sorted_idx
            
            G = grad_ops[gi]
            θi, costi = DBF.optimize_theta_expval(O, G, ψ, verbose=0)
           
            #
            # make sure energy lowering is large enough to warrent evolving
            # costi(0) - costi(θi) > evolve_coeff_thresh || continue

            n_rots < search_n_top || break 
            
            #See if we can do a cheap clifford operation
            # if costi(0) - costi(π/2) > evolve_coeff_thresh 
            #     θi = π/2
            #     @warn "clifford", costi(0) - costi(π/2)
            # end 
          
            

            O = evolve(O,G,θi)

            e1 = expectation_value(O,ψ)
            #
            # Truncate operator
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
            push!(generators, G)
            push!(angles, θi)
            n_rots += 1
            flush(stdout)
        end
        
        var_curr = variance(O,ψ)
        verbose < 1 || @printf("*%6i", iter)
        verbose < 1 || @printf(" %12.8f", ecurr)
        verbose < 1 || @printf(" %12.8f", norm_new)
        verbose < 1 || @printf(" %12.8f", real(accumulated_error))
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