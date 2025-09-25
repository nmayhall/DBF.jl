"""
    optimize_theta_expval(O::PauliSum{N,T}, G::PauliBasis{N}, ψ::Ket{N}; stepsize=.001, verbose=1) where {N,T}

Find the optimal θ that minimizes `<ψ|exp(iθ/2 G) O exp(-iθ/2 G)|ψ>`

Return the optimal angle, as well as the continious function that maps θ to the expectation value.
"""
function optimize_theta_expval(O::PauliSum{N,T}, G::PauliBasis{N}, ψ::Ket{N}; stepsize=.001, verbose=1) where {N,T}
    Oeval = expectation_value(O, ψ)
    OG = O*G
    OGeval = expectation_value(OG, ψ)
    GOGeval = expectation_value(G*O*G, ψ)
    function cost(θ)
        # Cost function for <ψ| U(θ)' O U(θ)|ψ>
        return real(cos(θ/2)^2 * Oeval + sin(θ/2)^2 * GOGeval - 2im*cos(θ/2)*sin(θ/2)*OGeval)
    end
    
    idx = argmin([cost(i*2π) for i in 0:stepsize:1-stepsize])
    # for i in 0:stepsize:1-stepsize
    #     @show i*2π, cost(i*2π)
    # end
    θ = (idx-1) * stepsize * 2π
    return θ, cost
end


"""
    dbf_eval(Oin::PauliSum{N,T}, ψ::Ket{N}; 
    max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
    evolve_coeff_thresh=1e-12) where {N,T}

TBW
"""
function dbf_eval(Oin::PauliSum{N,T}, ψ::Ket{N}; 
    max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
    evolve_coeff_thresh=1e-12) where {N,T}
    O = deepcopy(Oin)
    generators = Vector{PauliBasis}([])
    angles = Vector{Float64}([])
    norm_old = norm(offdiag(O))

    local_Z = PauliSum(N)
    for i in 1:N 
        local_Z += Pauli(N, Z=[i])
        for j in i+1:N 
            local_Z += Pauli(N, Z=[i,j])
        end
    end
    G_old = Pauli(N)

    display(local_Z)

    verbose < 1 || @printf(" %6s %12s %12s", "Iter", "|O|", "<ψ|O|ψ>")
    verbose < 1 || @printf(" %12s %12s %12s G\n", "norm(g)", "len(O)", "θ")
    for iter in 1:max_iter
        source = local_Z
        source = diag(O) + local_Z
        # source = diag(O)
        com = O*source - source*O
        coeff_clip!(O, thresh=evolve_coeff_thresh)
        if length(com) == 0
            println(" [H,diag] == 0 Exiting. ")
            break
        end
        # com = O*com - com*O
        f(k) = imag(expectation_value(k*O-O*k, ψ))
        G = argmax(k -> abs(f(k) * com[k]), keys(com))
        
        for k in keys(com)
            com[k] = abs(imag(expectation_value(k*O-O*k, ψ))*com[k])
        end
        
        G = argmax(k->abs(com[k]), keys(com))

        norm_new = norm(com)
        # G = argmax(k -> abs(com[k]), keys(com))

        if G == G_old
            println(" Trapped? ", string(G), " ", coeff)
            θi, costi = DBF.optimize_theta_expval(O, G, ψ, stepsize=.000001, verbose=1)
            step = .1
            for i in 0:.01:1
                θ = i*step*2π
                @printf(" θ=%12.8f cost=%12.8f\n", θ, costi(θ))
            end
            break
        end
        
        θi, costi = DBF.optimize_theta_expval(O, G, ψ, stepsize=.000001, verbose=0)
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