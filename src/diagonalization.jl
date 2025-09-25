
function optimize_theta_diagonalization(O,G; stepsize=.001, verbose=1)
    Od = diag(O)
    GOGd = diag(G*O*G)
    comd = diag(G*O-O*G)
    A = inner_product(Od,Od) 
    B = inner_product(GOGd,GOGd) 
    C = inner_product(comd,comd) 
    D = inner_product(Od,GOGd) 
    E = inner_product(Od,comd) 
    F = inner_product(GOGd,comd) 

    verbose < 2 || @show A, B, C, D, E

    function cost(θ)
        # C(x) = ||diag(U(x)' O U(x))||
        #      = ||diag((cos(x/2)+isin(x/2)G) O (cos(x/2)-isin(x/2)G)) ||
        #      = ||cos(x/2)^2diag(O) + sin(x/2)^2 diag(GOG) + isin(x/2)cos(x/2)diag([G,O]))||
        #
        #      = tr(cos(x/2)^2diag(O)' + sin(x/2)^2 diag(GOG)' - isin(x/2)cos(x/2)diag([G,O])')
        #          *cos(x/2)^2diag(O) + sin(x/2)^2 diag(GOG) + isin(x/2)cos(x/2)diag([G,O])) )
        #
        #      = cos(x/2)^4 (Od|Od) + sin(x/2)^4 (GOGd|GOGd) + sin(x/2)^2cos(x/2)^2 ([G,O]d|[G,O]d)
        #      + 2 cos(x/2)^2 sin(x/2)^2 (Od|GOGd)
        #      + i cos(x/2)^3 sin(x/2)^1 (Od|[G,O]d) 
        #      - i cos(x/2)^3 sin(x/2)^1 ([G,O]d|Od)    
        #      + i cos(x/2)^1 sin(x/2)^3 (GOGd|[G,O]d) 
        #      - i cos(x/2)^1 sin(x/2)^3 ([G,O]d|GOGd)   
        
        #      = cos(x/2)^4 (Od|Od) + sin(x/2)^4 (GOGd|GOGd) + sin(x/2)^2cos(x/2)^2 ([G,O]d|[G,O]d)
        #      + 2 cos(x/2)^2 sin(x/2)^2 (Od|GOGd) 
        #      + 2i cos(x/2)^3 sin(x/2)^1 (Od|comd) 
        #      + 2i cos(x/2)^1 sin(x/2)^3 (GOGd|comd) 
        
        #      = cos(x/2)^4 A + sin(x/2)^4 B + sin(x/2)^2cos(x/2)^2 C
        #      + 2 cos(x/2)^2 sin(x/2)^2  D 
        #      + 2i cos(x/2)^3 sin(x/2)^1 E 
        #      + 2i cos(x/2)^1 sin(x/2)^3 F
        
        #       where:
        #           A = (Od|Od)
        #           B = (GOGd|GOGd)
        #           C = ([G,O]d|[G,O]d)
        #           D = (Od|GOGd)
        #           E = (Od|[G,O]d)
        #           F = (GOGd|[G,O]d)
        #      
        #       Eventually we can simplify:
        #       since (Od|Od) == (GOGd|GOGd)
        #      = (cos^4 + sin^4) (Od|Od)  
        #       + sin^2 * cos^2 ((comd|comd) + 2 (Od|GOGd) )
        #       + 2i cos^3sin (Od|comd)
        #       + 2i cos sin^3 (GOGd|comd)
        #      
        #       since (GOGd|comd) == -(Od|comd)
        #      = (cos^4 + sin^4) (Od|Od)  
        #       + sin^2 * cos^2 ((comd|comd) + 2 (Od|GOGd) )
        #       + 2i (cos^3sin - cos sin^3) (Od|comd)
        #      
        #      
        c = cos(θ/2)
        s = sin(θ/2)
        return real(c^4 * A + s^4 * B + s^2*c^2*(C+2D) + 2im *c^3*s*E + 2im *c*s^3*F)
    end
    
    idx = argmax([real(cost(i*π)) for i in 0:stepsize:1-stepsize])
    θ = (idx-1) * stepsize * π
    
    verbose < 1 || @show θ, sqrt(cost(θ))
    return θ, cost
end



"""
    dbf_diag(Oin::PauliSum{N,T}; 
    max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
    evolve_coeff_thresh=1e-12) where {N,T}

Compute the double bracket flow for diagonalization of `Oin`
"""
function dbf_diag(Oin::PauliSum{N,T}; 
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

    verbose < 1 || @printf(" %6s %12s %12s %12s %12s G\n", "Iter", "θ", "|O|", "|od(O)|", "len(O)")
    for iter in 1:max_iter
        source = local_Z
        source = diag(O) + local_Z
        # source = diag(O)
        com = O*source - source*O
        coeff_clip!(com)
        if length(com) == 0
            println(" [H,diag] == 0 Exiting. ")
            break
        end
        coeff, G = findmax(v -> abs(v), com) 
        θi, costi = DBF.optimize_theta_diagonalization(O,G,stepsize=.000001, verbose=0)
        O = evolve(O,G,θi)
        coeff_clip!(O, thresh=evolve_coeff_thresh)

        norm_new = norm(offdiag(O))
        # if norm_new - costi(θi) > 1e-12
        #     @show norm_new - costi(θi)
        #     throw(ErrorException)
        # end
        # norm_new = costi(θi)/O_norm 
        verbose < 1 || @printf(" %6i %12.8f %12.8f %12.8f %12i", iter, θi, norm(O), norm_new, length(O))
        verbose < 1 || @printf(" %s", string(G))
        verbose < 1 || @printf("\n")
        push!(generators, G)
        push!(angles, θi)

        # if norm_new - norm_old < conv_thresh
        if norm_new < conv_thresh
            verbose < 1 || @printf(" Converged.\n")
            break
        end
       
        if G == G_old
            println(" Trapped?")
            break
        end
        if norm_new > norm_old
            println(" Norm increased?")
            throw(ErrorException)
        end
        if iter == max_iter
            verbose < 1 || @printf(" Not Converged.\n")
        end
        
        norm_old = norm_new
        G_old = G
    end
    return O, generators, angles
end

