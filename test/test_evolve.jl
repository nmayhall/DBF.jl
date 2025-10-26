using Test
using Random
using Printf
using PauliOperators
using LinearAlgebra
using DBF

@testset "test_evolve" begin
# function test2()
    Random.seed!(2)
    for N in 1:6
        O = rand(PauliSum{N}, n_paulis=50)
        @test norm(diag(Matrix(O))) ≈ norm(Matrix(diag(O))) 

        G = rand(PauliBasis{N})

        @show norm(O)
        @show norm(evolve(O,G,rand()))
        @test norm(O) ≈ norm(evolve(O,G,rand()))

        θ = rand()
        U = exp(Matrix(-1im * θ/2 * Matrix(G)))
        @test norm(Matrix(evolve(O,G,θ)) -  U'*Matrix(O)*U) < 1e-13
        # @test isapprox(Matrix(evolve(O,G,θ)),  U'*Matrix(O)*U, atol=1e-14)
        
        
        A = rand(PauliSum{N}, n_paulis=50)
        B = rand(PauliSum{N}, n_paulis=50)
        @test inner_product(A, B) ≈ tr(A'*B)/2^N
        @test norm(A)^2 ≈ inner_product(A,A) 
    end
end

function test()
    Random.seed!(2)

    N = 6 
    p = rand(PauliBasis{N})
    # p = PauliBasis("IIYZXZ")
    s = DyadSum(N)
    for i in 1:1
        ki = rand(Ket{N})
        s += rand() * (ki*ki')
    end
    s = s * (1/tr(s))
    display(s)
    
    p1 = DBF.reduce_by_1body(p,s)

    display(p1)
    @show expectation_value(p,s)
    @show expectation_value(p1,s)

    H = DBF.heisenberg_1D(N, 1,1,1)

    # generators = [rand(PauliBasis{N}) for i in 1:20]
    # angles = [rand()*2π for i in 1:20]
    generators = []
    angles = []
    for (p,c) in H
        push!(generators, p)
        push!(angles, real(c))
    end

    O = PauliSum(Pauli(N,Z=[1]))
    O2 = deepcopy(O)
    O0 = deepcopy(O)
    dt = .1
    diff = []
    for trot in 1:100
        for i in 1:length(generators)
            gi = generators[i]
            ai = angles[i]*dt
            O2 = evolve(O2, gi, ai)
            O = evolve(O, gi, ai)
            DBF.weight_clip!(O, 5)
            # DBF.meanfield_reduce!(O,s,2)
            DBF.coeff_clip!(O, thresh=1e-3)
            e = expectation_value(O,s)
            e2 = expectation_value(O2,s)
            push!(diff, e-e2)
            # @printf(" %3i %12.8f %12.8f %12.8f %12i\n", length(generators)*(trot-1)+i, e, e2, n, nterms)
        end 
        # e = expectation_value(O,s)
        # e2 = expectation_value(O2,s)
        e = expectation_value(O*O0, s)
        e2 = expectation_value(O2*O0, s)
        n = norm(O)
        nterms = length(O)
        @printf(" %3i %12.8f %12.8f %12.8f %12i\n", trot, e, e2, n, nterms)
    end
    @show norm(diff) 
end

# test()


# @testset "test_cnot" begin
function test2()
    N = 3
    Random.seed!(1)
    H = rand(PauliSum{N}, n_paulis=1)
    ψ = KetSum(N)
    ψ[Ket([0,0,0])] = 1
    # ψ[Ket([0,0,0])] = 1
    ψ = ψ * (1/norm(ψ))
    println("H:") 
    display(H)
    println("ψ:") 
    display(coeff_clip!(ψ, thresh=1e-12))
    println("H1*H2*ψ:")
    ψ = hadamard(ψ, 1)
    ψ = hadamard(ψ, 2) 
    # ψ = coeff_clip!(hadamard(ψ, 3), thresh=1e-12) 
    display(coeff_clip!(ψ, thresh=1e-12))
    println("CNOT(1,2)*H1*H2*ψ:")
    ψ = cnot(ψ, 1, 2)
    # display(coeff_clip!(ψ, thresh=1e-12))
    ψ = cnot(ψ, 1, 2)
    ψ = hadamard(ψ, 2)
    ψ = hadamard(ψ, 1)
    println()
    display(coeff_clip!(ψ, thresh=1e-12))

    # e1 = expectation_value(H,ψ)
    # H = DBF.cnot(H, 2, 1)
    # ψ = DBF.cnot(ψ, 2, 1)
    # e2 = expectation_value(H,ψ)
    # coeff_clip!(H, thresh=1e-14)
    # coeff_clip!(ψ, thresh=1e-14)
    # println()
    # display(H)
    # println()
    # display(ψ)
    # @show e1 ≈ e2
    # @test e1 ≈ e2
    
    H = rand(PauliSum{N}, n_paulis=10)
    H += H'
    ψ = rand(KetSum{N}, n_terms=10)
    ψ = ψ * (1/norm(ψ))

    e1 = expectation_value(H, ψ)
    e2a = expectation_value(H, cnot(hadamard(hadamard(ψ, 1), 2), 1, 2))
    e2b = expectation_value(hadamard(hadamard(cnot(H, 1, 2), 1), 2), ψ)
    @show e1
    @show e2a
    @show e2b
    @test e2a ≈ e2b
end

test2()