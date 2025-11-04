"""
 Compute the Majorana weight of a Pauli string.
"""
function majorana_weight(Pb::Union{PauliBasis{N}, Pauli{N}}) where N
    w = 0
    control = true
    # tmp = Pb.z & ~Pb.x  # Bitwise AND with bitwise NOT
    Ibits = ~(Pb.z|Pb.x)
    Zbits = Pb.z & ~Pb.x

    for i in reverse(1:N)  # Iterate from N down to 1
        xbit = (Pb.x >> (i - 1)) & 1 != 0
        Zbit = (Zbits >> (i - 1)) & 1 != 0
        Ibit = (Ibits >> (i - 1)) & 1 != 0
        #println("i=$i, xbit=$xbit, Zbit=$Zbit, Ibit=$Ibit, control=$control, w=$w")
        if Zbit && control || Ibit && !control
            w += 2
        elseif xbit
            control = !control
            w += 1
        end
    end
    return w
end

"""
 Compute the Pauli weight of a Pauli string.
"""
function pauli_weight(Pb::Union{PauliBasis{N}, Pauli{N}}) where N
    w = 0
    for i in 1:N
        xbit = (Pb.x >> (i - 1)) & 1
        zbit = (Pb.z >> (i - 1)) & 1

        if xbit != 0 || zbit != 0
            w += 1
        end
    end
    return w
end

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

function inner_product(k1::KetSum{N,T}, k2::KetSum{N,T}) where {N,T}
    out = T(0)
    if length(k1) < length(k2)
        for (p1,c1) in k1
            if haskey(k2,p1)
                out += c1'*k2[p1]
            end
        end
    else
        for (p2,c2) in k2
            if haskey(k1,p2)
                out += c2*k1[p2]'
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

function LinearAlgebra.norm(p::KetSum{N,T}) where {N,T}
    out = T(0)
    for (p,c) in p 
        out += abs2(c) 
    end
    return sqrt(real(out))
end

function weight(p::PauliBasis) 
    return count_ones(p.x | p.z)
end

function coeff_clip!(ps::KetSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter(p->abs(p.second) > thresh, ps)
end

function weight_clip!(ps::PauliSum{N}, max_weight::Int) where {N}
    return filter!(p->weight(p.first) <= max_weight, ps)
end

function majorana_weight_clip!(ps::PauliSum{N}, max_weight::Int) where {N}
    return filter!(p->majorana_weight(p.first) <= max_weight, ps)
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


function find_top_k(dict, k=10)
    """Optimized for when k << length(dict)"""
    
    # Pre-allocate arrays
    top_keys = Vector{keytype(dict)}(undef, k)
    top_vals = Vector{valtype(dict)}(undef, k) 
    top_abs = Vector{Float64}(undef, k)
    
    n_found = 0
    min_val = 0.0
    min_idx = 1
    
    @inbounds for (key, val) in dict
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



function get_weight_counts(O::PauliSum{N}) where N
    counts = zeros(Int, N)
    for (p,c) in O
        counts[weight(p)] += 1
    end
    return counts
end


function get_weight_probs(O::PauliSum{N}) where N
    probs = zeros(N)
    for (p,c) in O
        probs[weight(p)] += abs2(c) 
    end
    return probs 
end

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

"""
    Base.Matrix(p::PauliSum{N,T}, Vector{Ket{N}}) where {N,T}

Build Matrix representation of `p` in the space dfined by `S`
"""
# function Base.Matrix(p::PauliSum{N,T}, S::Vector{Ket{N}}) where {N,T}
#     nS = length(S)
#     M = zeros(T,nS,nS)
#     for i in 1:nS
#         M[i,i] = expectation_value(p,S[i])
#         for j in i+1:nS
#             M[i,j] = matrix_element(S[i]',p,S[j])
#             M[j,i] = matrix_element(S[j]',p,S[i])
#         end
#     end
#     return M
# end
function Base.Matrix(O::PauliSum{N,T}, S::Vector{Ket{N}}) where {N,T}
    nS = length(S)
    
    o = pack_x_z(O)

    def = Vector{Tuple{Int128, Float64}}()

    M = zeros(T,nS,nS)
    for i in 1:nS
        M[i,i] = expectation_value(O,S[i])
        for j in i+1:nS
            x = S[i].v ⊻ S[j].v
            ox = get(o, x, def)
            for (z,c) in ox
                p = PauliBasis{N}(z,x)
                phase, k = p*S[j]
                M[i,j] += phase*c

                phase, k = p*S[i]
                M[j,i] += phase*c
            end
        end
    end
    return M
end

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

function Base.Vector(k::KetSum{N,T}, S::Vector{Ket{N}}) where {N,T}
    nS = length(S)
    v = zeros(T,nS)
    length(k) == length(S) || throw(DimensionMismatch)
    for (i,keti) in enumerate(S)
        v[i] = k[keti]
    end
    return v
end




function Base.sum!(O::KetSum{N,T}, k::KetSum{N}) where {N,T}
    out = KetSum(N)
    for (p,c) in O
        c2,k2 = p*k
        tmp = get(out, k2, 0.0)
        out[k2] = tmp + c2*c
    end
    return out 
end

function Base.:*(O::PauliSum{N,T}, k::Ket{N}) where {N,T}
    out = KetSum(N)
    for (p,c) in O
        c2,k2 = p*k
        tmp = get(out, k2, 0.0)
        out[k2] = tmp + c2*c
    end
    return out 
end

function PauliOperators.expectation_value(O::PauliSum, v::KetSum)
    ev = 0
    for (p,c) in O
        for (k1,c1) in v
            ev += expectation_value(p,k1)*c*c1'*c1
            for (k2,c2) in v
                k2 != k1 || continue
                ev += matrix_element(k2', p, k1)*c*c2'*c1
            end
        end
    end
    return ev
end

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


function Base.:-(ps1::KetSum, ps2::KetSum)
    out = deepcopy(ps2)
    map!(x->-x, values(out))
    mergewith!(+, out, ps1)
    return out 
end


function Base.:+(ps1::KetSum, ps2::KetSum)
    out = deepcopy(ps2)
    mergewith!(+, out, ps1)
    return out 
end