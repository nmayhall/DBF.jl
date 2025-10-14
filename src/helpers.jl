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

"""
HS norm in Pauli basis: sum_P |c_P|^2.
"""
function hs_norm2(ps::PauliSum)
    s = 0.0
    @inbounds for (_, c) in ps
        s += abs2(c)
    end
    return s
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

function weight(p::PauliBasis) 
    return count_ones(p.x | p.z)
end

function coeff_clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter(p->abs(p.second) > thresh, ps)
end

"""
    Clip based on Pauli weight.
    Performs the pruning by removing all terms with weight > max_weight.
"""
function weight_clip!(ps::PauliSum{N}, max_weight::Int) where {N}
    filter!(p->weight(p.first) <= max_weight, ps)
end

function majorana_weight_clip!(ps::PauliSum{N}, max_weight::Int) where {N}
    filter!(p->majorana_weight(p.first) <= max_weight, ps)
end

"""
    Combined weight and coefficient clipping.
    w_type = 0 : Pauli weight
    w_type = 1 : Majorana weight
"""
function clip_thresh_weight!(ps::PauliSum{N}; thresh=1e-16, lc = 0, w_type = 0) where {N}
    if w_type == 0 
        filter!(p->(weight(p.first) <= lc) && (abs(p.second) > thresh) , ps)
    else
        filter!(p->(majorana_weight(p.first) <= lc) && (abs(p.second) > thresh) , ps)
    end     
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


