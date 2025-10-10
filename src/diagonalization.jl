using Optim



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
            search_n_top=100,
            extra_diag=nothing) where {N,T}

    O = deepcopy(Oin)
    generators = Vector{PauliBasis}([])
    angles = Vector{Float64}([])
    norm_old = norm(diag(O))

    G_old = Pauli(N)


    verbose < 1 || @printf(" %6s %12s %12s %12s %12s G\n", "Iter", "θ", "|O|", "|diag(O)|", "len(O)")
    for iter in 1:max_iter
        S = diag(O)

        # add extra diagonal operators to help 
        if extra_diag !== nothing
            S += extra_diag
        end

        # Find Search Direction:
        # this commutator seems to be the most expensive part at first.
        # Simplest thing to do is to just clip both before computing the 
        # Commutator, as the max commutator coomponent will most likely 
        # still be in here
        # old: 
        # com = O*S - S*O
        # com = max_of_commutator(O,S,clip=bracket_thresh)
        # SO_OS = max_of_commutator2(S,O,clip=bracket_thresh, n_top=100)
        SO_OS = max_of_commutator2(S, O, n_top=search_n_top)
        
        if length(SO_OS) == 0
            @warn " No search direction found. Increase `n_top` or decrease `clip`"
            break
        end

        coeff, G = findmax(v -> abs(v), SO_OS) 
        
        #
        # Find the optimal rotation angle along generator G 
        θi, costi = DBF.optimize_theta_diagonalization(O,G,verbose=0)
        push!(generators, G)
        push!(angles, θi)
        
        #
        # Evolve
        O = evolve(O,G,θi)

        # if norm(O) - costi(θi) > 1e-12
        #     @warn " Cost function not accurate: "
        #     @show norm(O), θi, costi(θi)
        #     # throw(ErrorException)
        # end

        #
        # Here's where we will do our truncating
        coeff_clip!(O, thresh=evolve_coeff_thresh)
        weight_clip!(O, evolve_weigth_thresh)
        

        norm_new = norm(diag(O))
        # norm_new = costi(θi)/O_norm 
        verbose < 1 || @printf(" %6i %12.8f %12.8f %12.8f %12i", iter, θi, norm(O), norm_new, length(O))
        verbose < 1 || @printf(" %s", string(G))
        verbose < 1 || @printf("\n")

        #
        # Check for convergence
        # if norm_new < conv_thresh
        # # if norm_new < conv_thresh
        #     verbose < 1 || @printf(" Converged.\n")
        # end
       
        if G == G_old
            @warn " Operator repeated" string(G) string(G_old) θi
            break
        end
        
        if norm_new < norm_old
            @warn " Norm increased?" norm_new norm_old norm_new-norm_old
            break
        end
        if iter == max_iter
            @warn " Not converged" iter max_iter
        end
        
        norm_old = norm_new
        G_old = G
    end
    return O, generators, angles
end


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
    
    options = Optim.Options(
        x_reltol = 1e-12, # A tight relative tolerance for changes in the solution vector
        f_reltol = 1e-12, # A tight relative tolerance for changes in the objective function value
        g_tol = 1e-10,    # A tighter absolute tolerance for the gradient
    )
    result = optimize(negative_cost, 0.0, π)
    # result = optimize(negative_cost, [0.0, π], Brent())
    # result = optimize(negative_cost, [0.0, π], LBFGS())
    θ = result.minimizer
    # f_min = result.minimum

    if Optim.iteration_limit_reached(result)
        @show Optim.abs_tol(result), Optim.rel_tol(result)
        @warn " minimization failed"
    end

    # Make sure bounds are respected
    θ < π || throw(ErrorException)
    θ > 0 || throw(ErrorException)

    if cost(θ) < cost(0)
        @warn " optimal θ worse than zero" θ cost(θ)  cost(0) cost(θ) - cost(0) "resetting"
        θ = 0
    end
    stepsize = 1e-5
    # idx = argmax([cost(i*π) for i in 0:stepsize:1-stepsize])
    # θ = (idx-1) * stepsize * π
    if cost(θ+stepsize) > cost(θ) || cost(θ-stepsize) > cost(θ)
        @show cost(θ+stepsize) , cost(θ) , cost(θ-stepsize), θ
        @show cost(θ+stepsize) - cost(θ)
        @show cost(θ-stepsize) - cost(θ)
        throw(ErrorException) 
    end
    
    verbose < 1 || @show θ, sqrt(cost(θ))
    return θ, cost
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
                            n_top=1000) where N
    curr_key = Pauli{N}(0,0,0)
    curr_val = 0
    out = PauliSum(N)
    sizehint!(out, min(1000, length(A) * length(B) ÷ 4))
    Btop = find_top_k_offdiag(B,n_top)
    # @show length(Btop) 
    @inbounds for (p1,c1) in find_top_k(A,n_top)

        @inbounds for (p2,c2) in Btop

            if PauliOperators.commute(p1,p2) == false
                prod = 2*c1*c2*(p1*p2)
                curr = get(out, PauliBasis(prod), 0.0) + PauliOperators.coeff(prod)
                
                out[PauliBasis(prod)] = curr 
            end
        end
    end
    return out
    # if length(out) == 0
    #     return PauliSum(N) 
    # end
    # coeff, G = findmax(v -> abs(v), out) 
    # return PauliSum(out[G]*G)
    # return PauliSum(coeff*G)
    # return PauliSum(curr_key*curr_val)
end

