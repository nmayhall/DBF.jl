module DBF

include("hamiltonians.jl")
include("evolve.jl")

function overlap(O1::PauliSum{N,T}, O2::PauliSum{N,T}) where {N,T}
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

export evolve
export overlap

end
