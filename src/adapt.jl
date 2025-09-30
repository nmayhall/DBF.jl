"""
    dbf_eval(Oin::PauliSum{N,T}, ψ::Ket{N}; 
    max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
    evolve_coeff_thresh=1e-12) where {N,T}

TBW
"""
function adapt(Oin::PauliSum{N,T}, pool::Vector{PauliBasis{N}}, ψ::Ket{N}; 
            max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
            evolve_coeff_thresh=1e-12,
            evolve_weight_thresh=20,
            extra_diag=nothing) where {N,T}
    O = deepcopy(Oin)
    generators = Vector{PauliBasis}([])
    angles = Vector{Float64}([])
    norm_old = norm(offdiag(O))

    G_old = Pauli(N)

    grad_vec = zeros(length(pool))

    verbose < 1 || @printf(" %6s %12s %12s", "Iter", "|O|", "<ψ|O|ψ>")
    verbose < 1 || @printf(" %12s %12s %12s G\n", "norm(g)", "len(O)", "θ")
    for iter in 1:max_iter
        

        # Compute gradient vector 
        for (pi,p) in enumerate(pool)
            dyad = (ψ * ψ') * p'
            grad_vec[pi] = 2*imag(expectation_value(O,dyad))
        end
        
        Gidx = argmax(abs.(grad_vec))
        G = pool[Gidx]
        # G = argmax(k -> abs(f(k)), pool)

        norm_new = norm(grad_vec)
        # G = argmax(k -> abs(com[k]), keys(com))

        # if G == G_old
        #     println(" Trapped? ", string(G), " ", coeff)
        #     θi, costi = DBF.optimize_theta_expval(O, G, ψ, verbose=1)
        #     # θi, costi = DBF.optimize_theta_expval(O, G, ψ, stepsize=.000001, verbose=1)
        #     step = .1
        #     for i in 0:.01:1
        #         θ = i*step*2π
        #         @printf(" θ=%12.8f cost=%12.8f\n", θ, costi(θ))
        #     end
        #     break
        # end
        
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




function generate_pool_1_weight(N)
    pool = Vector{PauliBasis{N}}([])
    for i in 1:N
        push!(pool,PauliBasis(Pauli(N,X=[i])))
        push!(pool,PauliBasis(Pauli(N,Y=[i])))
        push!(pool,PauliBasis(Pauli(N,Z=[i])))
    end
    return pool
end

function generate_pool_2_weight(N)
    pool = Vector{PauliBasis{N}}([])
    for i in 1:N
        for j in i+1:N
            push!(pool,PauliBasis(Pauli(N,X=[i,j])))
            push!(pool,PauliBasis(Pauli(N,Y=[i,j])))
            push!(pool,PauliBasis(Pauli(N,Z=[i,j])))
            push!(pool,PauliBasis(Pauli(N,X=[i],Y=[j])))
            push!(pool,PauliBasis(Pauli(N,X=[i],Z=[j])))
            push!(pool,PauliBasis(Pauli(N,Y=[i],Z=[j])))
            push!(pool,PauliBasis(Pauli(N,Y=[i],X=[j])))
            push!(pool,PauliBasis(Pauli(N,Z=[i],X=[j])))
            push!(pool,PauliBasis(Pauli(N,Z=[i],Y=[j])))
        end
    end
    return pool
end