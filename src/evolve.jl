using PauliOperators
using LinearAlgebra



"""
    evolve(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real)

Evolve the `O` by `G` 
O(θ) = exp(i θ/2 G) O exp(-i θ/2 G)

if [G,O] == 0
    O(θ) = O 
else
    O(θ) = O cos(θ) - i sin(θ) G*O
"""
function evolve(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real) where {N,T}
    _cos = cos(θ)
    _sin = 1im*sin(θ)
    cos_branch = deepcopy(O) 
    sin_branch = PauliSum(N)
    for (p,c) in O
        if PauliOperators.commute(p,G) == false
            cos_branch[p] *= _cos
            # replace sum! with more efficient version
            # sum!(sin_branch, c*_sin*G*p)
            tmp = c*_sin*G*p
            curr = get(sin_branch, PauliBasis(tmp), 0.0) + PauliOperators.coeff(tmp)
            sin_branch[PauliBasis(tmp)] = curr 
        end
    end
    sum!(cos_branch, sin_branch)
    return cos_branch 
end


function evolve!(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real) where {N,T}
    _cos = cos(θ)
    _sin = 1im*sin(θ)
    sin_branch = PauliSum(N)
    for (p,c) in O
        if PauliOperators.commute(p,G) == false
            # replace sum! with more efficient version
            # sum!(sin_branch, c*_sin*G*p)
            tmp = c*_sin*G*p
            curr = get(sin_branch, PauliBasis(tmp), 0.0) + PauliOperators.coeff(tmp)
            sin_branch[PauliBasis(tmp)] = curr 
            O[p] *= _cos
        end
    end
    sum!(O, sin_branch)
    return O 
end


function evolve(K::KetSum{N, T}, G::PauliBasis{N}, θ::Real) where {N,T}
    _cos = cos(θ/2)
    _sin = 1im*sin(θ/2)
    K2 = KetSum(N, T=ComplexF64)
    GK = KetSum(N, T=ComplexF64)
    for (k,c) in K
        K2[k] = c*_cos
        ci,ki = G*k
        
        tmp = get(GK,ki,0)
        GK[ki] = tmp + _sin*c*ci
    end
    for (k,c) in GK 
        tmp = get(K2,k,0)
        K2[k] = c + tmp
    end
    return K2 
end


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

        Ot = DBF.evolve(Ot, gi, θi)
        e1 = expectation_value(Ot, ψ)
        if compute_var_err
            v1 = variance(Ot, ψ)
        end 
        DBF.coeff_clip!(Ot, thresh=thresh)
        DBF.weight_clip!(Ot, max_weight)
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


function cnot(p::PauliSum{N}, c::Int, t::Int) where N 
    c <= N || throw(DimensionMismatch)
    t <= N || throw(DimensionMismatch)
    Zc = PauliBasis(Pauli(N,Z=[c]))
    Xt = PauliBasis(Pauli(N,X=[t]))
    ZXct = PauliBasis(Pauli(N,Z=[c],X=[t]))
    out = deepcopy(p)
    out = evolve(out, ZXct, π/2)
    out = evolve(out, Xt, -π/2)
    out = evolve(out, Zc, -π/2)
end

function cnot(p::KetSum{N}, c::Int, t::Int) where N 
    c <= N || throw(DimensionMismatch)
    t <= N || throw(DimensionMismatch)
    Zc = PauliBasis(Pauli(N,Z=[c]))
    Xt = PauliBasis(Pauli(N,X=[t]))
    ZXct = PauliBasis(Pauli(N,Z=[c],X=[t]))
    out = deepcopy(p)
    out = evolve(out, Zc, -π/2)
    out = evolve(out, Xt, -π/2)
    out = evolve(out, ZXct, π/2)
    return exp(1im*π/4)*out
end

function cnot_to_paulis(N, c::Int, t::Int) 
    c <= N || throw(DimensionMismatch)
    t <= N || throw(DimensionMismatch)
    Zc = PauliBasis(Pauli(N,Z=[c]))
    Xt = PauliBasis(Pauli(N,X=[t]))
    ZXct = PauliBasis(Pauli(N,Z=[c],X=[t]))
    g = Vector{PauliBasis{N}}([])
    θ = Vector{Float64}([])
    push!(g,ZXct)
    push!(g,Xt)
    push!(g,Zc)
    push!(θ, π/2)
    push!(θ, -π/2)
    push!(θ, -π/2)
    return g, θ 
end

function hadamard_to_paulis(N, q::Int) 
    q <= N || throw(DimensionMismatch)
    Z = PauliBasis(Pauli(N,Z=[q]))
    X = PauliBasis(Pauli(N,X=[q]))
    g = Vector{PauliBasis{N}}([])
    θ = Vector{Float64}([])
    push!(g,Z)
    push!(g,X)
    push!(g,Z)
    push!(θ, π/2)
    push!(θ, π/2)
    push!(θ, π/2)
    return g, θ 
end

function hadamard(p::Union{PauliSum{N}, KetSum{N}}, q::Int) where N
    out = deepcopy(p)
    Z = PauliBasis(Pauli(N,Z=[q]))
    X = PauliBasis(Pauli(N,X=[q]))
    out = evolve(out, Z, π/2)
    out = evolve(out, X, π/2)
    out = evolve(out, Z, π/2)
    return -1im*out
end

function X_gate_to_paulis(N, q) 
    return Vector{PauliBasis{N}}([PauliBasis(Pauli(N, X=[q]))]), Vector{Float64}([π])
end

function Z_gate_to_paulis(N, q) 
    return Vector{PauliBasis{N}}([PauliBasis(Pauli(N, Z=[q]))]), Vector{Float64}([π])
end

function S_gate(p::Union{PauliSum{N}, KetSum{N}}, q) where N
    return evolve(p, PauliBasis(Pauli(N, Z=[q])), π/2)
end

function T_gate(p::Union{PauliSum{N}, KetSum{N}}, q) where N
    return evolve(p, PauliBasis(Pauli(N, Z=[q])), π/4)
end

function X_gate(p::Union{PauliSum{N}, KetSum{N}}, q) where N
    return -1im*evolve(p, PauliBasis(Pauli(N, X=[q])), π)
end
function Y_gate(p::Union{PauliSum{N}, KetSum{N}}, q) where N
    return -1im*evolve(p, PauliBasis(Pauli(N, Y=[q])), π)
end
function Z_gate(p::Union{PauliSum{N}, KetSum{N}}, q) where N
    return -1im*evolve(p, PauliBasis(Pauli(N, Z=[q])), π)
end

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