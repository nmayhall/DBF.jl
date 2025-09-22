using LinearAlgebra
using Printf
using Random

function test_grad(O::PauliSum{N,T}, G::PauliBasis{N}, θ::Real) where {N,T}
    Umat = exp(-1im * θ / 2 * Matrix(G))
    U = cos(θ/2)*PauliBasis{N}(0,0) - 1im*sin(θ/2)*G
    @show norm(Matrix(U) - Umat)
    Omat = Matrix(O)
end

function evolve(O::PauliSum{N,T}, G::PauliBasis{N}, θ::Real, ψ::Ket{N}) where {N,T}
    U = cos(θ/2)*PauliBasis{N}(0,0) - 1im*sin(θ/2)*G
    O2 = U' * O * U
    return expectation_value(O2,ψ)
end

function evolve(A::Number, B::Number, C::Number, θ::Real)
    return A * cos(θ/2)^2 - 2 * C * cos(θ/2)*sin(θ/2) + B * sin(θ/2)^2
end

function get_opt_theta(A,B,C)

    K = (B-A)/C
    θ1 = -2*atan( K/2 + sqrt(K^2 + 4)/2)
    θ2 = -2*atan( K/2 - sqrt(K^2 + 4)/2)
    θ1 = real(θ1)
    θ2 = real(θ2)
    if C ≈ 0
        θ1 = 0
        θ2 = π/2
    end
    E1 = evolve(A,B,C,real(θ1))
    E2 = evolve(A,B,C,real(θ2))
    E1 = real(E1)
    E2 = real(E2)
    if E2<E1
        return real(θ2)
    elseif E1<E2
        return real(θ1)
    else
        return 0
    end
end

function run()
    # Random.seed!(15)
    N = 5
    H = rand(PauliSum{N}, n_paulis = 50)
    H += H'
    G = rand(PauliBasis{N})

    @show norm(Matrix(G*H-H*G))
    
    ψ = Ket{N}(0)
    A = expectation_value(H, ψ)
    B = expectation_value(G*H*G, ψ)
    C = imag(expectation_value(G*H, ψ))

    @show A
    @show B
    @show C
    @show get_opt_theta(A,B,C)

    for i in 0:.1:2
    
        θ = i*π
        e1 = evolve(H,G,θ,ψ)
        e2 = evolve(A,B,C,θ)
        @printf(" %12.8f : %12.8f %12.8f\n", θ, real(e1), real(e2))
    end
    println()
    θmin = get_opt_theta(A,B,C)
    e1 = evolve(H,G,θmin,ψ)
    e2 = evolve(A,B,C,θmin)
    @printf(" E(%f) : %12.8f %12.8f\n", θmin, real(e1), real(e2))
    

end

run()