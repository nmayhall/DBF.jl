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
