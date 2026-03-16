using PauliOperators
using LinearAlgebra

# Single-step evolve, evolve!, and KetSum evolve are now in PauliOperators

function evolve(O0::PauliSum{N,T}, g::Vector{PauliBasis{N}}, θ::Vector{<:Real};
                thresh=1e-3,
                max_weight=N,
                verbose=1,
                compute_var_err = false,
                print_n_steps = 10,
                ψ=Ket{N}(0)) where {N,T}

    ng = length(g)
    ng == length(θ) || throw(DimensionMismatch)

    verbose < 1 || @printf(" Number of rotations: %i\n", ng)
    energies = zeros(T,ng)
    variances = zeros(T,ng)
    accumated_error = zeros(T,ng)
    accumated_var_error = zeros(T,ng)
    v1 = 0
    v2 = 0

    err = 0
    verr = 0
    Ot = deepcopy(O0)
    idx = 1
    for (gi, θi) in zip(g, θ)

        Ot = PauliOperators.evolve(Ot, gi, θi)
        e1 = expectation_value(Ot, ψ)
        if compute_var_err
            v1 = variance(Ot, ψ)
        end 
        coeff_clip!(Ot, thresh)
        weight_clip!(Ot, max_weight)
        e2 = expectation_value(Ot, ψ)
        if compute_var_err
            v2 = variance(Ot, ψ)
        end 

        err += e2 - e1
        verr += v2 - v1
        energies[idx] = e2
        variances[idx] = v2
        accumated_error[idx] = err
        accumated_var_error[idx] = verr

        if idx%(length(g)÷print_n_steps) == 0
            @printf(" %4i E = %12.8f Var = %12.8f err = %12.8f verr = %12.8f\n", idx, e2, v2, err, verr)
        end
        idx += 1
    end
    return Ot, energies, variances, accumated_error, accumated_var_error
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