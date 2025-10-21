using PauliOperators
using LinearAlgebra



"""
    evolve(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real)

Evolve the `O` by `G` 
O(θ) = exp(i θ/2 G) O exp(-i θ/2 G)

if [G,O] == 0
    O(θ) = O 
else
    O(θ) = O cos(θ) - i sin(θ) G*O
"""
function evolve(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real) where {N,T}
    _cos = cos(θ)
    _sin = 1im*sin(θ)
    cos_branch = deepcopy(O) 
    sin_branch = PauliSum(N)
    for (p,c) in O
        if PauliOperators.commute(p,G) == false
            cos_branch[p] *= _cos
            # replace sum! with more efficient version
            # sum!(sin_branch, c*_sin*G*p)
            tmp = c*_sin*G*p
            curr = get(sin_branch, PauliBasis(tmp), 0.0) + PauliOperators.coeff(tmp)
            sin_branch[PauliBasis(tmp)] = curr 
        end
    end
    sum!(cos_branch, sin_branch)
    return cos_branch 
end

function evolve!(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real) where {N,T}
    _cos = cos(θ)
    _sin = 1im*sin(θ)
    sin_branch = PauliSum(N)
    for (p,c) in O
        if PauliOperators.commute(p,G) == false
            O[p] *= _cos
            # replace sum! with more efficient version
            # sum!(sin_branch, c*_sin*G*p)
            tmp = c*_sin*G*p
            curr = get(sin_branch, PauliBasis(tmp), 0.0) + PauliOperators.coeff(tmp)
            sin_branch[PauliBasis(tmp)] = curr 
        end
    end
    sum!(O, sin_branch)
    return O 
end


function evolve(O0::PauliSum{N,T}, g::Vector{PauliBasis{N}}, θ::Vector{<:Real}; 
                thresh=1e-3,
                ψ=Ket{N}(0)) where {N,T}
    expvals = Vector{T}([])
    err = 0
    Ot = deepcopy(O0)
    for (gi,θi) in zip(g,θ)
            
        Ot = DBF.evolve(Ot, gi, θi)
        
        e1 = expectation_value(Ot,ψ)
        DBF.coeff_clip!(Ot, thresh=thresh)
        e2 = expectation_value(Ot,ψ)

        err += e2 - e1
    end
    return Ot, err
end    


"""
    dissipate!(O::PauliSum, lmax::Int, γ::Real)

Apply dissipator to `O`, damping at a rate `γ` operators with
weight greater than `lmax`
"""
function dissipate!(O::PauliSum, lmax::Int, γ::Real)
    for (p,c) in O
        if weight(p) > lmax
            O[p] = exp(-γ*(weight(p)-lmax))*c
        end
    end
end


# function split_into_threads(O::PauliSum{N,T}) where {N,T}
#     nt = Threads.nthreads() 
#     psvec = Vector{PauliSum{N,T}}([])
# end


function split_dict(dict::Dict, n::Int)
    pairs = collect(dict)  # Get vector of key => value pairs
    chunk_size = cld(length(pairs), n)  # Ceiling division
    
    # Create n dictionaries
    dicts = [Dict(pairs[i:min(i+chunk_size-1, end)]) 
             for i in 1:chunk_size:length(pairs)]
    
    return dicts
end

function evolve_thread(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real) where {N,T}
    nt = Threads.nthreads()
    Ovec = split_dict(O, nt)
    @threads for i in 1:length(Ovec)
        evolve!(Ovec[i], G, θ)
    end

    return combine_dicts_with(+, Ovec)
end

# For overlapping keys, specify how to combine values
function combine_dicts_with(f, dicts::Vector{Dict{K,V}}) where {K,V}
    total_size = sum(length, dicts)
    result = Dict{K,V}()
    sizehint!(result, total_size)
    
    for d in dicts
        mergewith!(f, result, d)  # f(old_val, new_val) combines values
    end
    
    return result
end

# # Example: sum values for duplicate keys
# combined = combine_dicts_with(+, dicts)