using PauliOperators
using LinearAlgebra

# Single-step evolve, evolve!, and KetSum evolve are now in PauliOperators
# TODO: move evolve! for KetSum upstream to PauliOperators.jl (see agent.md)

"""
    evolve!(K::KetSum{N, ComplexF64}, G::PauliBasis{N}, θ::Real) where {N}

In-place Schrödinger-picture evolution: K → exp(-iθ/2 G) K

Modifies `K` in place. Required element type is `ComplexF64`.
"""
function evolve!(K::KetSum{N, ComplexF64}, G::PauliBasis{N}, θ::Real) where {N}
    _cos = cos(θ/2)
    _sin = -1im*sin(θ/2)
    GK = KetSum(N, T=ComplexF64)
    for (k, c) in K
        K[k] *= _cos
        ci, ki = G * k
        tmp = get(GK, ki, 0)
        GK[ki] = tmp + _sin * c * ci
    end
    for (k, c) in GK
        tmp = get(K, k, 0)
        K[k] = c + tmp
    end
    return K
end

"""
    evolve(O0::PauliSum{N,T}, g::Vector{PauliBasis{N}}, θ::Vector{<:Real};
           truncation=CompositeTruncation(CoeffTruncation(1e-3), WeightTruncation(N)),
           verbose=1, compute_var_err=false, print_n_steps=10, ψ=Ket{N}(0))

Evolve `O0` through a sequence of Pauli rotations, truncating after each step.

The `truncation` kwarg accepts any `TruncationStrategy` from PauliOperators.
Returns `(O, energies, variances, accumulated_error, accumulated_var_error)`.
"""
function evolve(O0::PauliSum{N,T}, g::Vector{PauliBasis{N}}, θ::Vector{<:Real};
                truncation::TruncationStrategy=CompositeTruncation(CoeffTruncation(1e-3), WeightTruncation(N)),
                verbose=1,
                compute_var_err = false,
                print_n_steps = 10,
                ψ=Ket{N}(0)) where {N,T}

    ng = length(g)
    ng == length(θ) || throw(DimensionMismatch)

    verbose < 1 || @printf(" Number of rotations: %i\n", ng)
    energies = zeros(T,ng)
    variances = zeros(T,ng)
    accumulated_error = zeros(T,ng)
    accumulated_var_error = zeros(T,ng)

    # Choose correction accumulator based on whether we track variance error
    corr = compute_var_err ? EnergyVarianceCorrection(ψ) : EnergyCorrection(ψ)

    Ot = deepcopy(O0)
    idx = 1
    for (gi, θi) in zip(g, θ)

        Ot = PauliOperators.evolve(Ot, gi, θi)

        truncate!(Ot, truncation, corr)

        e2 = expectation_value(Ot, ψ)
        v2 = compute_var_err ? variance(Ot, ψ) : zero(T)

        energies[idx] = e2
        variances[idx] = v2
        accumulated_error[idx] = corr.accumulated_energy
        accumulated_var_error[idx] = compute_var_err ? corr.accumulated_variance : zero(T)

        if idx%(length(g)÷print_n_steps) == 0
            @printf(" %4i E = %12.8f Var = %12.8f err = %12.8f verr = %12.8f\n",
                    idx, e2, v2, corr.accumulated_energy,
                    compute_var_err ? corr.accumulated_variance : zero(T))
        end
        idx += 1
    end
    return Ot, energies, variances, accumulated_error, accumulated_var_error
end


"""
    dissipate!(O::PauliSum, lmax::Int, γ::Real)

Apply dissipator to `O`, damping at a rate `γ` operators with
weight greater than `lmax`
"""
function dissipate!(O::PauliSum, lmax::Int, γ::Real)
    for (p,c) in O
        if weight(p) > lmax
            O[p] = exp(-γ*(weight(p)-lmax))*c
        end
    end
end

# Gate functions (cnot, hadamard, X/Y/Z/S/T_gate, *_to_paulis) are now in PauliOperators

function get_1d_neel_state_sequence(N)
    g = Vector{PauliBasis{N}}([])
    a = Vector{Float64}([])
    for i in 1:N
        if i%2 == 0
            push!(g, PauliBasis(Pauli(N, X=[i])))
            push!(a, π)
        end
    end
    return g, a 
end

function get_rvb_sequence(N)
    g = Vector{PauliBasis{N}}([])
    θ = Vector{Float64}([])
    for i in 0:N-1
        if i%2==0
            gi, θi = X_gate_to_paulis(N, i+1)
            g = vcat(g, reverse(gi))
            θ = vcat(θ, reverse(θi))
        end
    end
    for i in 0:N-1
        if i % 2 == 1
            gi, θi = hadamard_to_paulis(N, i+1)
            g = vcat(g, reverse(gi))
            θ = vcat(θ, reverse(θi))
            gi, θi = Z_gate_to_paulis(N, i+1)
            g = vcat(g, reverse(gi))
            θ = vcat(θ, reverse(θi))
        end
    end
    for i in 0:N-1
        if i % 2 == 1
            gi, θi = cnot_to_paulis(N, i+1, (i+1)%N + 1)
            g = vcat(g, reverse(gi))
            θ = vcat(θ, reverse(θi))
        end
    end
    # push!(g, PauliBasis(Pauli(N, Y=[1], X=[i for i in 2:N])))
    # push!(θ, -π/2)
    return reverse(g), reverse(θ) 
end

function get_1d_cluster_state_sequence(N)
    g = Vector{PauliBasis{N}}([])
    θ = Vector{Float64}([])
    for i in 0:N-1
        if i%2==0
            gi, θi = X_gate_to_paulis(N, i+1)
            g = vcat(g, reverse(gi))
            θ = vcat(θ, reverse(θi))
        end
    end
    for i in 0:N-1
        gi, θi = hadamard_to_paulis(N, i+1)
        g = vcat(g, reverse(gi))
        θ = vcat(θ, reverse(θi))
    end
    for i in 0:N-2
        
        gi, θi = hadamard_to_paulis(N, i+2)
        g = vcat(g, reverse(gi))
        θ = vcat(θ, reverse(θi))
        
        gi, θi = cnot_to_paulis(N, i+1, i+2)
        g = vcat(g, reverse(gi))
        θ = vcat(θ, reverse(θi))
        
        gi, θi = hadamard_to_paulis(N, i+2)
        g = vcat(g, reverse(gi))
        θ = vcat(θ, reverse(θi))
        
    end
    return reverse(g), reverse(θ) 
end