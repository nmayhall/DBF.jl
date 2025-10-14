using DBF

function Base.Vector(k::OrderedDict{Ket{N},T}) where {N,T}
    vec = zeros(T,Int128(2)^N)
    for (k,coeff) in k
        vec[PauliOperators.index(k)] = T(coeff) 
    end
    return vec 
end

function Base.Matrix(k::OrderedDict{Ket{N},T}) where {N,T}
    vec = zeros(T,Int128(2)^N,1)
    for (k,coeff) in k
        vec[PauliOperators.index(k),1] = T(coeff) 
    end
    return vec 
end


function matvec(O::XZPauliSum, v::OrderedDict{Ket{N}, T}) where {N,T}
    s = OrderedDict{Ket{N}, T}()
    sizehint!(s, length(O)) 

    for (vi, ci) in v
        for (x, zs) in O
            b = Ket{N}(vi.v ⊻ x)
            
            val = get(s, b, T(0))
            for (z, c) in zs
                p = PauliBasis{N}(z,x)
                ph,b = p*vi
                val += ph * c * ci
            end
            s[b] = val
        end
    end
    return s
end


function subspace_matvec(O::XZPauliSum, v::OrderedDict{Ket{N}, T}) where {N,T}
    s = deepcopy(v)
    return subspace_matvec!(s, O, v) 
end

# function subspace_matvec(O::XZPauliSum, v::OrderedDict{Ket{N}, T}) where {N,T}
function subspace_matvec!(s::OrderedDict{Ket{N}, T}, O::XZPauliSum, v::OrderedDict{Ket{N}, T}) where {N,T}
    s = deepcopy(v)
    for (sk,sc) in s 
        s[sk] = T(0)
    end
    PHASE_SIGNS = [1, 1im, -1, -1im]

    for (vi, ci) in v
        for (x, zs) in O
            b = Ket{N}(vi.v ⊻ x)
            
            haskey(s,b) || continue

            val = get(s, b, T(0))
            for (z, c) in zs
                p = PauliBasis{N}(z, x)
                
                phase = symplectic_phase(p) + 2 * (count_ones(z & b.v) % 2)
                sign = PHASE_SIGNS[phase%4+1]
                val += sign * c * ci
            end
            s[b] = val
        end
    end
    return s
end
    
    