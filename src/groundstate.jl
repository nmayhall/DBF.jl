"""
    optimize_theta_expval(O::PauliSum{N,T}, G::PauliBasis{N}, ψ::Ket{N}; stepsize=.001, verbose=1) where {N,T}

Find the optimal θ that minimizes `<ψ|exp(iθ/2 G) O exp(-iθ/2 G)|ψ>`

Return the optimal angle, as well as the continious function that maps θ to the expectation value.
"""
function optimize_theta_expval(O::PauliSum{N,T}, G::PauliBasis{N}, ψ::Ket{N}; verbose=1) where {N,T}
    Oeval = expectation_value(O, ψ)
    OG = O*G
    OGeval = expectation_value(OG, ψ)
    GOGeval = expectation_value(G*O*G, ψ)
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
function dbf_groundstate(Oin::PauliSum{N,T}, ψ::Ket{N}; 
            max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
            evolve_coeff_thresh=1e-12,
            evolve_weigth_thresh=20,
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