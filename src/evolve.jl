using PauliOperators
using LinearAlgebra

function LinearAlgebra.norm(p::PauliSum{N,T}) where {N,T}
    out = T(0)
    for (p,c) in p 
        out += abs2(c) 
    end
    return sqrt(out)
end

function largest_diag(ps::PauliSum{N,T}) where {N,T}
    argmax(kv -> abs(last(kv)), filter(p->p.first.x == 0, ps))
end
    
function largest(ps::PauliSum{N,T}) where {N,T}
    max_val, max_key = findmax(v -> abs(v), ps)

    return PauliSum{N,T}(max_key => ps[max_key])
end
    
function LinearAlgebra.diag(ps::PauliSum{N,T}) where {N,T}
    filter(p->p.first.x == 0, ps)
end

function offdiag(ps::PauliSum{N,T}) where {N,T}
    filter(p->p.first.x != 0, ps)
end


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

"""
    cost_function_diagonal(O)

Return norm(diag(O))
"""
function cost_function_diagonal(O)
    return norm(diag(O))
end
