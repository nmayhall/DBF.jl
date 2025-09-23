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


function dbf_diag(Oin::PauliSum; 
    max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3)
    O = deepcopy(Oin)
    generators = Vector{PauliBasis}([])
    angles = Vector{Float64}([])
    norm_old = norm(diag(O))

    verbose < 1 || @printf(" %6s %12s %12s G\n", "Iter", "θ", "Norm")
    for iter in 1:max_iter
        com = O*diag(O) - diag(O)*O
        if length(com) == 0
            return
        end
        coeff, G = findmax(v -> abs(v), com) 
        θi, costi = DBF.optimize_theta_diagonalization(O,G,stepsize=.00001, verbose=0)
        O = evolve(O,G,θi)
        norm_new = norm(diag(O))
        verbose < 1 || @printf(" %6i %12.8f %12.8f %s", iter, θi, norm_new, string(G))
        verbose < 1 || @printf("\n")
        push!(generators, G)
        push!(angles, θi)

        if norm_new - norm_old < conv_thresh
            verbose < 1 || @printf(" Converged.")
            break
        end
        
        norm_old = norm_new
        if iter == max_iter
            verbose < 1 || @printf(" Not Converged.")
        end
    end
    return O, generators, angles
end