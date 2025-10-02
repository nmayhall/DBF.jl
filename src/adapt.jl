using PauliOperators

"""
    dbf_eval(Oin::PauliSum{N,T}, ψ::Ket{N}; 
    max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
    evolve_coeff_thresh=1e-12) where {N,T}

TBW
"""
function adapt(Oin::PauliSum{N,T}, pool::Vector{PauliBasis{N}}, ψ::Ket{N}; 
            max_iter=10, thresh=1e-4, verbose=1, conv_thresh=1e-3,
            evolve_coeff_thresh=1e-12,
            evolve_weight_thresh=20,
            evolve_grad_thresh=1e-8,
            extra_diag=nothing) where {N,T}
    O = deepcopy(Oin)
    generators = Vector{PauliBasis}([])
    angles = Vector{Float64}([])
    norm_old = norm(offdiag(O))
            
    ecurr = expectation_value(O, ψ) 

    G_old = Pauli(N)
    
    accumulated_error = 0

    grad_vec = zeros(length(pool))

    verbose < 1 || @printf(" %6s %12s %12s", "Iter", "|H|", "<ψ|H|ψ>")
    verbose < 1 || @printf(" %12s", "||<[H,Gi]>||")
    verbose < 1 || @printf(" %12s", "# Rotations")
    verbose < 1 || @printf(" %12s", "len(H)")
    verbose < 1 || @printf(" %12s", "total_error")
    verbose < 1 || @printf(" %12s", "variance")
    verbose < 1 || @printf("\n")
    for iter in 1:max_iter
        

        # Compute gradient vector
        for (pi,p) in enumerate(pool)
            # dyad = (ψ * ψ') * p'
            # grad_vec[pi] = 2*imag(expectation_value(O,dyad))
            c, σ = p*ψ
            grad_vec[pi] = 2*imag(matrix_element(σ', O, ψ)*c)
        end
        
        Gidx = argmax(abs.(grad_vec))
        G = pool[Gidx]
        # G = argmax(k -> abs(f(k)), pool)

        norm_new = norm(grad_vec)
        # G = argmax(k -> abs(com[k]), keys(com))

        # if G == G_old
        #     println(" Trapped? ", string(G), " ", coeff)
        #     θi, costi = DBF.optimize_theta_expval(O, G, ψ, verbose=1)
        #     # θi, costi = DBF.optimize_theta_expval(O, G, ψ, stepsize=.000001, verbose=1)
        #     step = .1
        #     for i in 0:.01:1
        #         θ = i*step*2π
        #         @printf(" θ=%12.8f cost=%12.8f\n", θ, costi(θ))
        #     end
        #     break
        # end
       
        sorted_idx = reverse(sortperm(abs.(grad_vec)))

        verbose < 2 || @printf("     %8s %12s %12s", "pool idx", "||O||", "<ψ|H|ψ>")
        verbose < 2 || @printf(" %12s %12s %s", "len(O)", "θi", string(G))
        verbose < 2 || @printf("\n")
        n_rots = 0
        for gi in sorted_idx

            #
            # make sure gradient is non-negligible
            abs(grad_vec[gi]) > evolve_grad_thresh || continue

            G = pool[gi]
            θi, costi = DBF.optimize_theta_expval(O, G, ψ, verbose=0)
           
            #
            # make sure energy lowering is large enough to warrent evolving
            abs(costi(0) - costi(θi)) > evolve_grad_thresh || continue
           

            O = evolve(O,G,θi)

            e1 = expectation_value(O,ψ)
            #
            # Truncate operator
            coeff_clip!(O, thresh=evolve_coeff_thresh)
            weight_clip!(O, evolve_weight_thresh)
            e2 = expectation_value(O,ψ)

            accumulated_error += e2 - e1
            # if norm_new - costi(θi) > 1e-12
            #     @show norm_new - costi(θi)
            #     throw(ErrorException)
            # end
            # norm_new = costi(θi)/O_norm
            ecurr = expectation_value(O, ψ) 
            verbose < 2 || @printf("     %8i %12.8f %12.8f", gi, norm(O), ecurr)
            verbose < 2 || @printf(" %12i %12.8f %s", length(O), θi, string(G))
            verbose < 2 || @printf("\n")
            push!(generators, G)
            push!(angles, θi)
            n_rots += 1
            flush(stdout)
        end
        var_curr = variance(O,ψ)
        verbose < 1 || @printf("*%6i %12.8f %12.8f %12.8f", iter, norm(O), ecurr, norm_new)
        verbose < 1 || @printf(" %12i", n_rots)
        verbose < 1 || @printf(" %12i", length(O))
        verbose < 1 || @printf(" %12.8f", real(accumulated_error))
        verbose < 1 || @printf(" %12.8f", real(var_curr))
        verbose < 1 || @printf("\n")
        
        # if norm_new - norm_old < conv_thresh
        if norm_new < conv_thresh
            verbose < 1 || @printf(" Converged.\n")
            break
        end
       
        # if norm_new > norm_old
        #     println(" Norm increased?")
        #     throw(ErrorException)
        # end
        if iter == max_iter
            verbose < 1 || @printf(" Not Converged.\n")
        end
      
        if n_rots == 0
            @warn """ No search directions found. 
                    Tighten `evolve_grad_thresh` or expand pool"""
            break
        end
        norm_old = norm_new
        # G_old = G
    end
    return O, generators, angles
end

function variance(O::PauliSum{N}, ψ::Ket{N}) where N
    σ = KetSum(N)
    for (p,ci) in O
        cj, ki = p*ψ
        # σ[ki] += cj*ci
        curr = get(σ, ki, 0.0) + cj*ci
        σ[ki] = curr 
    end 
    
    # @show norm(Vector(σ)), norm(Matrix(O)*Vector(ψ))
    e2 = 0
    for (k,v) in σ 
        e2 += v'*v
    end

    e1 = expectation_value(O,ψ)

    return e2 - e1^2
end



function generate_pool_1_weight(N)
    pool = Vector{PauliBasis{N}}([])
    for i in 1:N
        push!(pool,PauliBasis(Pauli(N,X=[i])))
        push!(pool,PauliBasis(Pauli(N,Y=[i])))
        push!(pool,PauliBasis(Pauli(N,Z=[i])))
    end
    return pool
end

function generate_pool_2_weight(N)
    pool = Vector{PauliBasis{N}}([])
    for i in 1:N
        for j in i+1:N
            # push!(pool,PauliBasis(Pauli(N,X=[i,j])))
            push!(pool,PauliBasis(Pauli(N,Y=[i],X=[j])))
        end
    end
    return pool
end

function generate_pool_3_weight(N)
    pool = Vector{PauliBasis{N}}([])
    for i in 1:N
        for j in i+1:N
            for k in j+1:N
                push!(pool,PauliBasis(Pauli(N,Y=[i], X=[j,k])))
            end
        end
    end
    return pool
end

function generate_pool_4_weight(N)
    pool = Vector{PauliBasis{N}}([])
    for i in 1:N
        for j in i+1:N
            for k in j+1:N
                for l in k+1:N
                    push!(pool,PauliBasis(Pauli(N,Y=[i], X=[j,k,l])))
                    push!(pool,PauliBasis(Pauli(N,X=[i], Y=[j,k,l])))
                end
            end
        end
    end
    return pool
end


function generate_pool_5_weight(N)
    pool = Vector{PauliBasis{N}}([])
    for i in 1:N
        for j in i+1:N
            for k in j+1:N
                for l in k+1:N
                    for m in l+1:N
                        push!(pool,PauliBasis(Pauli(N,Y=[i], X=[j,k,l,m])))
                    end
                end
            end
        end
    end
    return pool
end


function generate_pool_6_weight(N)
    pool = Vector{PauliBasis{N}}([])
    for i in 1:N
        for j in i+1:N
            for k in j+1:N
                for l in k+1:N
                    for m in l+1:N
                        for n in m+1:N
                            push!(pool,PauliBasis(Pauli(N,Y=[i], X=[j,k,l,m,n])))
                        end
                    end
                end
            end
        end
    end
    return pool
end

# function generate_pool_3_weight(N)
#     pool = Vector{PauliBasis{N}}([])
#     for i in 1:N
#         for j in i+1:N
#             for j in i+1:N
#                 push!(pool,PauliBasis(Pauli(N,X=[i,j,k])))
#                 push!(pool,PauliBasis(Pauli(N,Y=[i,j,k])))
#                 push!(pool,PauliBasis(Pauli(N,Z=[i,j,k])))
#                 push!(pool,PauliBasis(Pauli(N,X=[i],Y=[j])))
#                 push!(pool,PauliBasis(Pauli(N,X=[i],Z=[j])))
#                 push!(pool,PauliBasis(Pauli(N,Y=[i],Z=[j])))
#                 push!(pool,PauliBasis(Pauli(N,Y=[i],X=[j])))
#                 push!(pool,PauliBasis(Pauli(N,Z=[i],X=[j])))
#                 push!(pool,PauliBasis(Pauli(N,Z=[i],Y=[j])))
#             end
#         end
#     end
#     return pool
# end


# function matrix_element(b::Bra{N}, p::PauliBasis{N}, k::Ket{N}) where N
#     # <b| ZZZ...*XXX...|k> (1im)^sp
#     sgn = count_ones(p.z & b.v)  # sgn <j| = <j| z 
#     val = k.v ⊻ b.v == p.x # <j|x|i>
#     if val
#         return (-1)^sgn * 1im^symplectic_phase(p)
#     else
#         return 0
#     end 
#     # sgn1 = 1
#     # phs1 = 1
#     # if sgn % 2 != 0
#     #     sgn1 = -1
#     # end
#     # sp = symplectic_phase(p)
#     # if sp == 1
#     #     phs1 = 1im
#     # elseif sp == 2
#     #     phs1 = -1
#     # elseif sp == 3
#     #     phs1 = -1im
#     # end

#     # return sgn1 * phs1 * val * coeff(p) * coeff(d)
#     # # return (-1)^sgn * val * coeff(p) * coeff(d) * 1im^symplectic_phase(p)
# end

function pool_test1(O::PauliSum{N}) where N
    
    pool = PauliSum(N)
    for i in 1:N
        pool += Pauli(N,X=[i])
        pool += Pauli(N,Y=[i])
        pool += Pauli(N,Z=[i])
    end


    A = diag(O)
    B = offdiag(O)
    pool = A*B-B*A 
    # for i in 1:N
    #     for j in i+1:N
    #         for k in j+1:N
    #             pool += Pauli(N,Y=[i], X=[j,k])
    #         end
    #     end
    # end

    # for i in 1:N
    #     for j in i+1:N
    #         for k in j+1:N
    #             for l in k+1:N
    #                 pool += Pauli(N,Y=[i], X=[j,k,l])
    #             end
    #         end
    #     end
    # end

    # pool = O*pool + pool*O
    # weight_clip!(pool,5)
    coeff_clip!(pool)

    return [first(x) for x in sort(collect(pool), by = x -> abs(last(x)))]
end

function entropy(H)

    # S = -sum_i |c_i|^2 log(|c_i|^2)
    # S(x) = 
end