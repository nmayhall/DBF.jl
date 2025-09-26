using Optim

function LinearAlgebra.isdiag(P::Pauli)
    return P.x == 0
end

function diag_GOG(G::PauliBasis{N},O::PauliSum{N}) where N
    out = PauliSum(N)
    sizehint!(out, min(1000, length(O)))
    for (p,c) in O
        gpg = c*(G*p*G)
        if isdiag(gpg)
            curr = get(out, PauliBasis(gpg), 0.0) + PauliOperators.coeff(gpg)
            out[PauliBasis(gpg)] = curr 
            # sum!(out,gpg)
        end
    end
    return out
end

"""
    diag_commutator(G::Pauli{N}, O::PauliSum{N}) where N

Evaluate diag(GO-OG)
"""
function diag_commutator(G::PauliBasis{N}, O::PauliSum{N}) where N 
    out = PauliSum(N)
    sizehint!(out, min(1000, length(O)))
    for (p,c) in O
        if PauliOperators.commute(G,p) == false
            term = 2*c*(G*p)
            if isdiag(term)
                curr = get(out, PauliBasis(term), 0.0) + PauliOperators.coeff(term)
                out[PauliBasis(term)] = curr
            end
        end
    end
    return out
end

function optimize_theta_diagonalization(O,G; verbose=1)
    Od = diag(O)
    # GOGd = diag(G*O*G)
    # comd = diag(G*O-O*G)
    GOGd = diag_GOG(G,O)
    comd = diag_commutator(G,O)
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
    
    negative_cost(θ) = -cost(θ)
    result = optimize(negative_cost, 0.0, 2π, Brent())
    θ = result.minimizer
    # f_min = result.minimum
    if Optim.iteration_limit_reached(result)
        @show Optim.abs_tol(result), Optim.rel_tol(result)
        println(" minimization failed")
    end

    # Make sure bounds are respected
    θ < 2π || throw(ErrorException)
    θ > 0 || throw(ErrorException)

    if cost(θ) < cost(0)
        @show cost(θ) , cost(0)
        println(" WTF")
    end
    stepsize = 1e-5
    # idx = argmin([real(cost(i*π)) for i in 0:stepsize:1-stepsize])
    # θ = (idx-1) * stepsize * π
    if cost(θ+stepsize) > cost(θ) || cost(θ-stepsize) > cost(θ)
        @show cost(θ+stepsize) , cost(θ) , cost(θ-stepsize), θ
        @show cost(θ+stepsize) - cost(θ)
        @show cost(θ-stepsize) - cost(θ)
        println(" WTF")
    end
    
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
    evolve_coeff_thresh=1e-12,
    evolve_weigth_thresh=20,
    bracket_thresh=1e-8) where {N,T}
    O = deepcopy(Oin)
    generators = Vector{PauliBasis}([])
    angles = Vector{Float64}([])
    norm_old = norm(offdiag(O))

    local_Z = PauliSum(N)
    for i in 1:N 
        local_Z += Pauli(N, Z=[i])*rand()
        for j in i+1:N 
            local_Z += Pauli(N, Z=[i,j])*rand()
        end
    end
    G_old = Pauli(N)

    # display(local_Z)

    verbose < 1 || @printf(" %6s %12s %12s %12s %12s G\n", "Iter", "θ", "|O|", "|od(O)|", "len(O)")
    for iter in 1:max_iter
        S = local_Z
        # S = diag(O) + local_Z
        S = diag(O)

        # this commutator seems to be the most expensive part at first.
        # Simplest thing to do is to just clip both before computing the 
        # Commutator, as the max commutator coomponent will most likely 
        # still be in here
        # old: 
        # com = O*S - S*O
        # com = max_of_commutator(find_top_k(O,100),S,clip=bracket_thresh)
        com = max_of_commutator2(S,O,clip=bracket_thresh, n_top=100)
        
        coeff_clip!(com)
        if length(com) == 0
            println(" [H,diag] == 0 Exiting. ")
            break
        end
        coeff, G = findmax(v -> abs(v), com) 
        θi, costi = DBF.optimize_theta_diagonalization(O,G,verbose=0)
        
        #
        # Evolve
        O = evolve(O,G,θi)

        if norm(O) - costi(θi) > 1e-12
            @show norm(O), θi, costi(θi)
            throw(ErrorException)
        end
        coeff_clip!(O, thresh=evolve_coeff_thresh)
        weight_clip!(O, evolve_weigth_thresh)
        

        norm_new = norm(offdiag(O))
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
            @show norm_new - norm_old
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

function max_of_commutator(A::PauliSum{N},B::PauliSum{N};clip=1e-8) where N
    curr_key = Pauli{N}(0,0,0)
    curr_val = 0
    out = PauliSum(N)
    sizehint!(out, min(1000, length(A) * length(B) ÷ 4))
    @inbounds for (p1,c1) in A
        abs(c1) > clip || continue

        @inbounds for (p2,c2) in B
            abs(c2) > clip || continue

            if PauliOperators.commute(p1,p2) == false
                prod = 2*c1*c2*(p1*p2)
                curr = get(out, PauliBasis(prod), 0.0) + PauliOperators.coeff(prod)
                
                out[PauliBasis(prod)] = curr 
            end
        end
    end
    coeff, G = findmax(v -> abs(v), out) 
    return PauliSum(out[G]*G)
    # return PauliSum(coeff*G)
    # return PauliSum(curr_key*curr_val)
end


function max_of_commutator2(A::PauliSum{N},B::PauliSum{N};
                            clip=1e-8,
                            n_top=1000) where N
    curr_key = Pauli{N}(0,0,0)
    curr_val = 0
    out = PauliSum(N)
    sizehint!(out, min(1000, length(A) * length(B) ÷ 4))
    Btop = find_top_k(B,n_top)
    # @show length(Btop) 
    @inbounds for (p1,c1) in find_top_k(A,n_top)
        # abs(c1) > clip || continue

        @inbounds for (p2,c2) in Btop
        # @inbounds for (p2,c2) in B
            # abs(c2) > clip || continue

            if PauliOperators.commute(p1,p2) == false
                prod = 2*c1*c2*(p1*p2)
                curr = get(out, PauliBasis(prod), 0.0) + PauliOperators.coeff(prod)
                
                out[PauliBasis(prod)] = curr 
            end
        end
    end
    coeff, G = findmax(v -> abs(v), out) 
    return PauliSum(out[G]*G)
    # return PauliSum(coeff*G)
    # return PauliSum(curr_key*curr_val)
end

