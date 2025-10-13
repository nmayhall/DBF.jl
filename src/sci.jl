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
    
    for (sk,sc) in s 
        s[sk] = T(0)
    end

    for (vi, ci) in v
        for (x, zs) in O
            b = Ket{N}(vi.v ⊻ x)
            if haskey(s,b)
                # display(b)
                val = get(s, b, T(0))
                for (z, c) in zs
                    p = PauliBasis{N}(z,x)
                    ph,b = p*vi
                    val += ph * c * ci
                end
                s[b] = val
            end
        end
    end
    return s
end