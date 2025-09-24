
function p_space(O::PauliSum{N,T}, split_idx) where {N,T}
    out = PauliSum(N)
    M = split_idx
    for (p,c) in O
        if p.x < 2^M || p.x % 2^M == 0
            out[p] = c
        end
    end
    return out 
end

function q_space(O::PauliSum{N,T}, split_idx) where {N,T}
    out = PauliSum(N)
    M = split_idx
    for (p,c) in O
        if p.x >= 2^M && p.x % 2^M != 0
            out[p] = c
        end
        println(string(p), " ", p.x % 2^M == 1)
    end
    return out 
end

function optimize_theta_downfold(O, G, split_idx; stepsize=.001, verbose=1)
    Od = diag(O)
    GOGd = diag(G*O*G)
    comd = diag(G*O-O*G)
    A = inner_product(Od,Od) 
    B = inner_product(GOGd,GOGd) 
    C = inner_product(comd,comd) 
    D = inner_product(Od,GOGd) 
    E = inner_product(Od,comd) 
    F = inner_product(GOGd,comd) 

    verbose < 2 || @show A, B, C, D, E

    function cost(θ)
        # C(x) = ||diag(U(x)' O U(x))||
        #      = ||diag((cos(x/2)+isin(x/2)G) O (cos(x/2)-isin(x/2)G)) ||
        #      = ||cos(x/2)^2diag(O) + sin(x/2)^2 diag(GOG) + isin(x/2)cos(x/2)diag([G,O]))||
        #
        #      = tr(cos(x/2)^2diag(O)' + sin(x/2)^2 diag(GOG)' - isin(x/2)cos(x/2)diag([G,O])')
        #          *cos(x/2)^2diag(O) + sin(x/2)^2 diag(GOG) + isin(x/2)cos(x/2)diag([G,O])) )
        #
        #      = cos(x/2)^4 (Od|Od) + sin(x/2)^4 (GOGd|GOGd) + sin(x/2)^2cos(x/2)^2 ([G,O]d|[G,O]d)
        #      + 2 cos(x/2)^2 sin(x/2)^2 (Od|GOGd)
        #      + i cos(x/2)^3 sin(x/2)^1 (Od|[G,O]d) 
        #      - i cos(x/2)^3 sin(x/2)^1 ([G,O]d|Od)    
        #      + i cos(x/2)^1 sin(x/2)^3 (GOGd|[G,O]d) 
        #      - i cos(x/2)^1 sin(x/2)^3 ([G,O]d|GOGd)   
        
        #      = cos(x/2)^4 (Od|Od) + sin(x/2)^4 (GOGd|GOGd) + sin(x/2)^2cos(x/2)^2 ([G,O]d|[G,O]d)
        #      + 2 cos(x/2)^2 sin(x/2)^2 (Od|GOGd) 
        #      + 2i cos(x/2)^3 sin(x/2)^1 (Od|comd) 
        #      + 2i cos(x/2)^1 sin(x/2)^3 (GOGd|comd) 
        
        #      = cos(x/2)^4 A + sin(x/2)^4 B + sin(x/2)^2cos(x/2)^2 C
        #      + 2 cos(x/2)^2 sin(x/2)^2  D 
        #      + 2i cos(x/2)^3 sin(x/2)^1 E 
        #      + 2i cos(x/2)^1 sin(x/2)^3 F
        
        #       where:
        #           A = (Od|Od)
        #           B = (GOGd|GOGd)
        #           C = ([G,O]d|[G,O]d)
        #           D = (Od|GOGd)
        #           E = (Od|[G,O]d)
        #           F = (GOGd|[G,O]d)
        #      
        #       Eventually we can simplify:
        #       since (Od|Od) == (GOGd|GOGd)
        #      = (cos^4 + sin^4) (Od|Od)  
        #       + sin^2 * cos^2 ((comd|comd) + 2 (Od|GOGd) )
        #       + 2i cos^3sin (Od|comd)
        #       + 2i cos sin^3 (GOGd|comd)
        #      
        #       since (GOGd|comd) == -(Od|comd)
        #      = (cos^4 + sin^4) (Od|Od)  
        #       + sin^2 * cos^2 ((comd|comd) + 2 (Od|GOGd) )
        #       + 2i (cos^3sin - cos sin^3) (Od|comd)
        #      
        #      
        c = cos(θ/2)
        s = sin(θ/2)
        return real(c^4 * A + s^4 * B + s^2*c^2*(C+2D) + 2im *c^3*s*E + 2im *c*s^3*F)
    end
    
    idx = argmax([real(cost(i*π)) for i in 0:stepsize:1-stepsize])
    θ = (idx-1) * stepsize * π
    
    verbose < 1 || @show θ, sqrt(cost(θ))
    return θ, cost
end
