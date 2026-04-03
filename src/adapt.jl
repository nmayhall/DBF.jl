using PauliOperators

"""
    adapt(Oin::PauliSum{N,T}, pool::Vector{PauliBasis{N}}, ψ::Ket{N};
        max_iter=10, verbose=1, conv_thresh=1e-3,
        active_window=nothing,
        operator_truncation=CoeffTruncation(1e-6),
        compute_var_error=true,
        maxiter_lbfgs=100, g_tol=1e-8) where {N,T}

ADAPT-VQE optimization. At each macro-iteration:
1. Compute the energy gradient for every operator in `pool`
2. Select the operator with the largest gradient magnitude
3. Append it to the growing generator sequence
4. Re-optimize the last `active_window` rotation angles simultaneously via LBFGS

When the ansatz grows beyond `active_window` operators, the earliest operators are
"frozen": their rotations are absorbed into a truncated Heisenberg-evolved Hamiltonian
`H_frozen`, with truncation error tracked via an accumulated correction. Only the last
`active_window` operators remain variationally active.

If `active_window=nothing` (default), all operators are always optimized (no freezing).

Returns a `NamedTuple` with fields:
- `generators`: full sequence of Pauli generators
- `angles`: corresponding rotation angles
- `energies`: energy after each ADAPT iteration
- `H_frozen`: the truncated, frozen-layer Hamiltonian (equals `Oin` if no freezing occurred)
- `accumulated_error`: total energy truncation error from freezing
- `accumulated_var_error`: total variance truncation error from freezing (if `compute_var_error=true`)
"""
function adapt(Oin::PauliSum{N,T}, pool::Vector{PauliBasis{N}}, ψ::Ket{N};
            max_iter=10, verbose=1, conv_thresh=1e-3,
            active_window::Union{Nothing,Int}=nothing,
            operator_truncation::TruncationStrategy=CoeffTruncation(1e-6),
            compute_var_error::Bool=true,
            maxiter_lbfgs::Int=100, g_tol::Float64=1e-8) where {N,T}

    generators = Vector{PauliBasis{N}}([])
    angles = Vector{Float64}([])
    energies = Vector{Float64}([])

    # H_frozen is the Heisenberg-evolved H through the frozen (non-active) generators
    H_frozen = deepcopy(Oin)
    n_frozen = 0  # number of generators absorbed into H_frozen

    # Track truncation error from freezing
    corr = compute_var_error ? EnergyVarianceCorrection(ψ) : EnergyCorrection(ψ)

    ecurr = real(expectation_value(H_frozen, ψ))
    push!(energies, ecurr)

    verbose < 1 || @printf(" %6s %14s %12s %12s %8s %6s %12s %8s",
        "Iter", "Energy", "Variance", "max|grad|", "len(U)", "active", "LBFGS_iters", "len(H)")
    if compute_var_error
        verbose < 1 || @printf(" %12s %12s", "E_err", "V_err")
    end
    verbose < 1 || @printf("\n")

    for iter in 1:max_iter

        # Determine the active window
        n_total = length(generators)
        n_active = active_window === nothing ? n_total : min(active_window, n_total)
        active_gens = @view generators[n_frozen+1:end]
        active_angs = @view angles[n_frozen+1:end]

        # Build H_eff = H_frozen evolved through the active generators (no truncation)
        H_eff = PauliOperators.evolve(H_frozen, collect(active_gens), collect(active_angs))

        # Gradient screening in the Heisenberg picture:
        #   ∂E/∂θ|_{θ=0} = 2 * Im⟨ψ|H_eff G|ψ⟩
        Hxz_eff = pack_x_z(H_eff)
        σ_eff = matvec(Hxz_eff, ψ)

        grad_vec = zeros(length(pool))
        for (pi, p) in enumerate(pool)
            c, σ = p * ψ
            grad_vec[pi] = 2 * imag(get(σ_eff, σ, ComplexF64(0)) * c)
        end

        max_grad = maximum(abs.(grad_vec))

        if max_grad < conv_thresh
            verbose < 1 || @printf(" Converged: max|grad| = %.2e < %.2e\n", max_grad, conv_thresh)
            break
        end

        # Select the operator with the largest gradient
        Gidx = argmax(abs.(grad_vec))
        G_new = pool[Gidx]
        push!(generators, G_new)
        push!(angles, 0.0)

        # Freeze operators if we exceed the active window
        if active_window !== nothing
            while (length(generators) - n_frozen) > active_window
                # Freeze the oldest active generator into H_frozen
                gi = generators[n_frozen + 1]
                θi = angles[n_frozen + 1]
                PauliOperators.evolve!(H_frozen, gi, θi)
                truncate!(H_frozen, operator_truncation, corr)
                n_frozen += 1
            end
        end

        # Re-optimize active angles
        active_gens = generators[n_frozen+1:end]
        active_angs = Float64.(angles[n_frozen+1:end])

        res = optimize_rotation_sequence(H_frozen, active_gens, ψ,
            initial_angles=active_angs,
            maxiter=maxiter_lbfgs,
            g_tol=g_tol,
            verbose=max(0, verbose - 1))

        angles[n_frozen+1:end] .= res.angles
        ecurr = res.energy + real(corr.accumulated_energy)
        push!(energies, ecurr)

        # Compute variance on the fully evolved Hamiltonian
        H_opt = PauliOperators.evolve(H_frozen, generators[n_frozen+1:end], Float64.(angles[n_frozen+1:end]))
        var_curr = real(variance(H_opt, ψ))

        n_active_now = length(generators) - n_frozen
        verbose < 1 || @printf(" %6i %14.8f %12.3e %12.3e %8i %6i %12i %8i",
            iter, ecurr, var_curr, max_grad, length(generators), n_active_now,
            Optim.iterations(res.result), length(H_frozen))
        if compute_var_error
            verbose < 1 || @printf(" %12.2e %12.2e",
                real(corr.accumulated_energy),
                compute_var_error ? real(corr.accumulated_variance) : 0.0)
        end
        verbose < 1 || @printf("\n")

        if iter == max_iter
            verbose < 1 || @printf(" Not Converged.\n")
        end
    end

    return (generators=generators, angles=angles, energies=energies,
            H_frozen=H_frozen,
            accumulated_error=real(corr.accumulated_energy),
            accumulated_var_error=compute_var_error ? real(corr.accumulated_variance) : 0.0)
end

# variance(O::PauliSum, ψ::Ket) is now in PauliOperators statistics.jl

function skewness(O::PauliSum{N,T}, ψ::Ket{N}) where {N,T}
    Oxz = pack_x_z(O)
    σ = matvec(Oxz, ψ)
    Oσ = deepcopy(σ)
    for (x,zs) in Oσ
        Oσ[x] = T(0)
    end
    # Oσ = subspace_matvec(Oxz, σ)
    subspace_matvec_thread!(Oσ, Oxz, σ)

    m3 = inner_product(σ,Oσ)
    m2 = inner_product(σ,σ)
    m1 = expectation_value(O,ψ)

    k1 = m1
    k2 = m2 - m1^2
    k3 = m3 - 3*m2*m1 + 2*m1^3

    # @show m1
    # @show m2
    # @show m3
    return k1, k2, k3 
end


function generate_commutator_pool(O::PauliSum{N}) where N
    S = PauliSum(N)
    for i in 1:N
        S += PauliBasis(Pauli(N,Z=[i]))
        for j in i+1:N
            S += PauliBasis(Pauli(N,Z=[i,j]))
        end
    end
    gen = O*S-S*O
    coeff_clip!(gen, 1e-16)

    pool = Vector{PauliBasis{N}}([])
    for (p,c) in gen 
        push!(pool,p)
    end
    return pool
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

# = = = = ADDITIONAL POOLS OCT 2025 = = = =
"""
  The following functions implement standard pools (GSD for fermions)
  Part of them has been taken or based from Kyle Sherbert's ADAPT.jl package:
  https://github.com/kmsherbertvt/ADAPT.jl/blob/main/src/base/pools.jl

  Qubit excitation pool is based on: https://www.nature.com/articles/s42005-021-00730-0

  # N is the number of qubits
    # i,j are the qubits indices as defined Yordanov et. al. 2021
    # Note that Yordanov's unitaries are defined as `exp(iθG)` rather than `exp(-iθG)`,
    # so variational parameters will be off by a sign.
    # Returns 
    # -PauliOperators.PauliSum{N} : The qubit excitation operator as a PauliSum
    # Note that all pauli terms in any single qubit excitation operator commute, 
    #so we can return a single PauliSum
"""
function qubitexcitation(n::Int, i::Int, k::Int)
    return 0.5 .* [Pauli(n, X=[i], Y=[k]),
                   -Pauli(n, X=[k], Y=[i])]
end

function qubitexcitation(n::Int, i::Int, j::Int, k::Int, l::Int)
        return (1/8) .* [Pauli(n; X=[i,k,l], Y=[j]),
                         Pauli(n; X=[j,k,l], Y=[i]),
                         Pauli(n; X=[l], Y=[i,j,k]),
                         Pauli(n; X=[k], Y=[i,j,l]),
                         -Pauli(n; X=[i,j,l], Y=[k]),
                         -Pauli(n; X=[i,j,k], Y=[l]),
                         -Pauli(n; X=[j], Y=[i,k,l]),
                         -Pauli(n; X=[i], Y=[j,k,l])]
    end

"""                
    qubitexcitationpool(n_system::Int)
          
    The number of singles excitations = (n 2), and the doubles = 3*(n 4).
            
   # Parameters
    - `n_system`: Number of qubits in the system

    # Returns
    - `pool`: the qubit-excitation-based pool as defined in Communications Physics 4, 1 (2021).
    - `target_and_source`: Dict mapping each pool operator to the target and source orbitals involved in the excitation. 
"""               
function qubitexcitationpool(n_system::Int)
    N = n_system    
    pool = Vector{PauliBasis{N}}([])
    
    for i in 1:n_system
        for j in i+1:n_system
            # singles excitations
            singles = qubitexcitation(n_system, i, j)
            for op in singles
#                println("Operator in Single excitation")
                push!(pool, PauliBasis(op))
            end
#            target_and_source[singles] = [[i,j]]

            # doubles excitations
            for k in j+1:n_system
                for l in k+1:n_system
                    target_pair = [i,j]; source_pair = [k,l]
                    doubles = qubitexcitation(n_system, target_pair[1], target_pair[2], source_pair[1], source_pair[2])
                    for op in doubles
                        push!(pool, PauliBasis(op))
                    end
#                    target_and_source[doubles] = [target_pair,source_pair]

                    target_pair = [i,k]; source_pair = [j,l]
                    doubles = qubitexcitation(n_system, target_pair[1], target_pair[2], source_pair[1], source_pair[2])
                    for op in doubles
                        push!(pool, PauliBasis(op))
                    end
#                    target_and_source[doubles] = [target_pair,source_pair]
               
                    target_pair = [j,k]; source_pair = [i,l]     
                    doubles = qubitexcitation(n_system, target_pair[1], target_pair[2], source_pair[1], source_pair[2])
                    for op in doubles
#                        println("Operator in Double excitation")
                        push!(pool, PauliBasis(op))
                    end
#                    target_and_source[doubles] = [target_pair,source_pair]

                    end
                end
            end
        end
        return pool
    end


#pool = qubitexcitationpool(4)
#for p in pool
#    display(p)
#    println("Weight: ", weight(p))
#end
#println("Total operators in pool: ", length(pool))
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
    coeff_clip!(pool, 1e-16)

    return [first(x) for x in sort(collect(pool), by = x -> abs(last(x)))]
end

function entropy(O)

    # S = -sum_i |c_i|^2 log(|c_i|^2)
    # S(x) = 
    s = 0
    n = norm(O)
    for (_,c) in O
        p = abs2(c)/n^2
        s -= p*log(p)
    end
    return s
end
