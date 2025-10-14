using DBF
using Base.Threads

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
    # return subspace_matvec!(s, O, v) 
    return subspace_matvec_thread!(s, O, v) 
end

function subspace_matvec!(s::OrderedDict{Ket{N}, T}, O::XZPauliSum, v::OrderedDict{Ket{N}, T}) where {N,T}
    s = deepcopy(v)
    for (sk,sc) in s 
        s[sk] = T(0)
    end

    for (vi, ci) in v
        for (x, zs) in O
            b = Ket{N}(vi.v ⊻ x)
            
            haskey(s,b) || continue

            val = get(s, b, T(0))
            for (z, c) in zs
                p = PauliBasis{N}(z, x)
                ph, b = p*vi
                val += ph * c * ci
            end
            s[b] = val
        end
    end
    return s
end
    
    
function subspace_matvec_thread!(s::OrderedDict{Ket{N}, T}, O::XZPauliSum, v::OrderedDict{Ket{N}, T}) where {N,T}
    s = deepcopy(v)
    for (sk,sc) in s 
        s[sk] = T(0)
    end

    function collect_sigma_block!(s)
        #
        #|i>si = \sum_kj hj Pj |k> vk
        #
        @threads for i in collect(keys(s))
            si = s[i]
            for (x, zs) in O
                k = Ket{N}(x ⊻ i.v)

                haskey(v, k) || continue

                vk = get(v, k, T(0))
                for (z, c) in zs
                    p = PauliBasis{N}(z, x)
                    ph, _ = p * k
                    si += ph * c * vk 
                end
            end
            s[i] = si 
        end
    end
    collect_sigma_block!(s)
    return s
end
    
    