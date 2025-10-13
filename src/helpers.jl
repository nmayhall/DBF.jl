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
    
    # Build [X][Z] container
    o = Dict{Int128,Dict{Int128,Float64}}()
    for (p,c) in O
        dx = get(o, p.x, Dict{Int128,Float64}())
        dxz = get(dx, p.z, 0.0)
        dx[p.z] = dxz + c
        o[p.x] = dx
    end

    def = Dict{Int128, Float64}()

    M = zeros(T,nS,nS)
    for i in 1:nS
        M[i,i] = expectation_value(O,S[i])
        for j in i+1:nS
            x = S[i].v ⊻ S[j].v
            ox = get(o, x, def)
            for (z,c) in ox
            # for (z,c) in o[x]
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


function Base.:*(O::PauliSum{N,T}, k::Ket{N}) where {N,T}
    out = KetSum(N)
    for (p,c) in O
        c2,k2 = p*k
        tmp = get(out, k2, 0.0)
        out[k2] = tmp + c2*c
    end
    return out 
end

"""
    pt2(H::PauliSum{N,T}, k::Ket{N}) where {N,T}

e2 = |<k|Ho|x>|^2 / (e0 - <x|Hd|x>)
"""
function pt2(H::PauliSum{N,T}, ψ::Ket{N}) where {N,T}
    Hd = diag(H)
    e2 = T(0)
    e0 = expectation_value(Hd,ψ)

    # Build [X][Z] container
    h = Dict{Int128,Dict{Int128,Float64}}()
    for (p,c) in H
        dx = get(h, p.x, Dict{Int128,Float64}())
        dxz = get(dx, p.z, 0.0)
        dx[p.z] = dxz + c
        h[p.x] = dx
    end
    def = Dict{Int128, Float64}()

    @show length(h), length(H)
    for (x,dx) in h

        # make sure p isn't diagonal
        x != 0 || continue       
        
        σHψ = 0
        σ = Ket{N}(0)
        
        hx = get(h, x, def)
        for (z,c) in hx
            pzx = PauliBasis{N}(z,x)
            czx, σ = pzx * ψ
            σHψ += czx * c 
        end
        e2 +=  abs2(σHψ) / (e0 - expectation_value(Hd, σ))
        # c2,k2 = p*k
        
        # k2 != k || error(" k==k2")
        # e2 += (c*c2)'*(c*c2) / (e0 - expectation_value(Hd, k2))
        # e2 += 1 / (e0 - expectation_value(Hd, k2))
        # e2 += 1 / (e0)
        # @show (c*c2)'*(c*c2) / (e0 - expectation_value(Hd, k2))
    end
    return e0, e2
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
