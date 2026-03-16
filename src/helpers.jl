"""
    coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}

Non-mutating version of coeff_clip! — returns a filtered copy.
"""
function coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter(p->abs(p.second) > thresh, ps)
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


function find_top_k_offdiag(dict, k=10)
    """Optimized for when k << length(dict)"""
    
    # Pre-allocate arrays
    top_keys = Vector{keytype(dict)}(undef, k)
    top_vals = Vector{valtype(dict)}(undef, k) 
    top_abs = Vector{Float64}(undef, k)
    
    n_found = 0
    min_val = 0.0
    min_idx = 1
    
    @inbounds for (key, val) in dict
        key.x != 0 || continue
        abs_val = abs(val)
        
        if n_found < k
            # Still filling up
            n_found += 1
            top_keys[n_found] = key
            top_vals[n_found] = val  
            top_abs[n_found] = abs_val
            
            # Update minimum
            if abs_val < min_val || n_found == 1
                min_val = abs_val
                min_idx = n_found
            end
            
        elseif abs_val > min_val
            # Replace minimum
            top_keys[min_idx] = key
            top_vals[min_idx] = val
            top_abs[min_idx] = abs_val
            
            # Find new minimum
            min_val = top_abs[1]
            min_idx = 1
            for i in 2:k
                if top_abs[i] < min_val
                    min_val = top_abs[i]
                    min_idx = i
                end
            end
        end
    end
    
    # Sort the results
    p = sortperm(view(top_abs, 1:n_found), rev=true)
    return [top_keys[p[i]] => top_vals[p[i]] for i in 1:n_found]
end



# Weight distribution analysis functions are now in PauliOperators:
# get_weight_counts, get_weight_probs, get_majorana_weight_counts, get_majorana_weight_probs
# Note: DBF's get_mweight_counts/get_mweight_probs → PauliOperators' get_majorana_weight_counts/get_majorana_weight_probs

function add_single_excitations(k::Ket{N}) where N
    s = KetSum(N)
    s[k] = 1
    for i in 1:N
        for j in 1:N
            i != j || continue
            c,b = Pauli(N, X=[i, j]) * k
            # count_ones(k.v) == count_ones(b.v) || continue 
            coeff = get(s, b, 0)
            s[b] = coeff + c
        end
    end
    # for (k,c) in s 
    #     @show count_ones(k.v)
    # end
    return s
end

# Base.Matrix(O::PauliSum{N,T}, S::Vector{Ket{N}}) is now in PauliOperators analysis.jl

function Base.Matrix(O::XZPauliSum{T}, basis::Vector{Ket{N}}) where {N,T}
    n = length(basis)
    
    def = Dict{Int128, Float64}()

    M = zeros(ComplexF64,n,n)
    for (i, keti) in enumerate(basis)
        # M[i,i] = expectation_value(O,keti)

        for (j, ketj) in enumerate(basis)
            j >= i || continue
            x = keti.v ⊻ ketj.v
            ox = get(O, x, def)
            for (z,c) in ox
            # for (z,c) in o[x]
                p = PauliBasis{N}(z,x)
                phase,_ = p*ketj
                M[i,j] += phase*c

                j > i || continue
                phase,_  = p*keti
                M[j,i] += phase*c
            end
        end
    end
    return M
end

function Base.Matrix(k::KetSum{N,T}, S::Vector{Ket{N}}) where {N,T}
    nS = length(S)
    v = zeros(T,nS,1)
    length(k) == length(S) || throw(DimensionMismatch)
    for (i,keti) in enumerate(S)
        v[i,1] = k[keti]
    end
    return v
end

# Base.Vector(K::KetSum{N,T}, S::Vector{Ket{N}}) is now in PauliOperators analysis.jl




# Base.sum!(O::KetSum, k::KetSum) is now in PauliOperators addition.jl

# Base.:*(PauliSum, Ket) is now in PauliOperators multiplication.jl

# expectation_value(O::PauliSum, v::KetSum) is now in PauliOperators expectation_value.jl

function PauliOperators.expectation_value(O::XZPauliSum, v::KetSum{N,T}) where {N,T}
    ev = 0
    for (x,zs) in O
        for (z,c) in zs 
            p = PauliBasis{N}(z,x)
            for (k1,c1) in v
                ev += expectation_value(p,k1)*c*c1'*c1
                for (k2,c2) in v
                    k2 != k1 || continue
                    ev += matrix_element(k2', p, k1)*c*c2'*c1
                end
            end
        end
    end
    return ev
end
function PauliOperators.expectation_value(O::XZPauliSum, v::Ket{N}) where {N}
    ev = 0
    haskey(O,0) || return 0.0
    for (z,c) in O[0]
        p = PauliBasis{N}(z,Int128(0))
        ev += expectation_value(p,v)*c
    end
    return ev
end

"""
    pack_x_z(H::PauliSum{N,T}) where {N,T}

Convert PauliSum into a Dict{Int128,Vector{Tuple{Int128,Float64}}}
This allows us to access Pauli's by first specifying `x`, 
then 'z'. 
"""
function pack_x_z(H::PauliSum{N,T}) where {N,T}
    # Build [X][Z] container
    h = Dict{Int128,Vector{Tuple{Int128,T}}}()
    for (p,c) in H
        dx = get(h, p.x, Vector{Tuple{Int128,T}}())
        push!(dx, (p.z,c))
        h[p.x] = dx
    end
    return h
end


# KetSum +/- KetSum are now in PauliOperators addition.jl