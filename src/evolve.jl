using PauliOperators
using LinearAlgebra



"""
    evolve(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real)

Evolve the `O` by `G` 
O(θ) = exp(i θ/2 G) O exp(-i θ/2 G)

if [G,O] == 0
    O(θ) = O 
else
    O(θ) = O cos(θ) - i 2 sin(θ) G*O
"""
function evolve(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real) where {N,T}
    _cos = cos(θ)
    _sin = 1im*sin(θ)
    cos_branch = deepcopy(O) 
    sin_branch = PauliSum(N)
    for (p,c) in O
        if PauliOperators.commute(p,G) == false
            cos_branch[p] *= _cos
            sum!(sin_branch, c*_sin*G*p)
        end
    end
    sum!(cos_branch, sin_branch)
    return cos_branch 
end


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