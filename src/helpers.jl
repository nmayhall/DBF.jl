function inner_product(O1::PauliSum{N,T}, O2::PauliSum{N,T}) where {N,T}
    out = T(0)
    if length(O1) < length(O2)
        for (p1,c1) in O1
            if haskey(O2,p1)
                out += c1'*O2[p1]
            end
        end
    else
        for (p2,c2) in O2
            if haskey(O1,p2)
                out += c2*O1[p2]'
            end
        end
    end
    return out
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

function LinearAlgebra.norm(p::PauliSum{N,T}) where {N,T}
    out = T(0)
    for (p,c) in p 
        out += abs2(c) 
    end
    return sqrt(real(out))
end

function weight(p::PauliBasis) 
    return count_ones(p.x | p.z)
end

function coeff_clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter(p->abs(p.second) > thresh, ps)
end

function weight_clip!(ps::PauliSum{N}, max_weight::Int) where {N}
    filter!(p->weight(p.first) <= max_weight, ps)
end

function reduce_by_1body(p::PauliBasis{N}, ψ) where N
    out = PauliSum(N)
    # for i in 1:N
    n_terms = length(PauliOperators.get_on_bits(p.z|p.x)) 
    for i in PauliOperators.get_on_bits(p.z|p.x) 
        mask = 1 << (i - 1) 
        tmp1 = PauliBasis{N}(p.z & ~mask, p.x & ~mask)
        tmp2 = PauliBasis{N}(p.z & mask, p.x & mask)
        tmp3 = tmp1*tmp2
        if isapprox(coeff(tmp3), 1) == false || PauliBasis(tmp3) != p
            throw(ErrorException)
        end 
        # println(string(tmp1), "*", string(tmp2), "=", string(tmp1*tmp2))
        out += tmp1 * (expectation_value(tmp2, ψ) / n_terms)
        out += tmp1 * (expectation_value(tmp2, ψ) / n_terms)
        # display(PauliBasis{N}(p.z & ~mask, p.x & ~mask))
    end
    out = out * (1/norm(out))
    # for (p,c) in out
    #     println(string(p), " ", weight(p))
    # end
    return out
end

function meanfield_reduce!(O::PauliSum{N},s, weightclip) where N
    tmp = PauliSum(N)
    for (p,c) in O
        if weight(p) > weightclip 
            tmp += reduce_by_1body(p,s)
            O[p] = 0
        end
    end 
    O += tmp
end

function max_of_commutator(A::PauliSum{N},B::PauliSum{N};clip=1e-8) where N
    curr_key = Pauli{N}(0,0,0)
    curr_val = 0
    out = PauliSum(N)
    sizehint!(out, min(1000, length(A) * length(B) ÷ 4))
    for (p1,c1) in A
        abs(c1) > clip || continue

        for (p2,c2) in B
            abs(c2) > clip || continue
            if PauliOperators.commute(p1,p2) == false
                prod = 2*c1*c2*(p1*p2)
                # out += 2*c1*c2*(p1*p2)
                curr = get(out, PauliBasis(prod), 0.0) + PauliOperators.coeff(prod)
                
                out[PauliBasis(prod)] = curr 
                # if abs(prod) > abs(curr_val)
                #     curr_val = prod
                #     curr_key = 2*p1*p2
                # end
            end
        end
    end
    coeff, G = findmax(v -> abs(v), out) 
    return PauliSum(coeff*G)
    # return PauliSum(curr_key*curr_val)
end