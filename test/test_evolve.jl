using Test
using Random
using Printf
using PauliOperators
using LinearAlgebra
using DBF

# @testset "theta_opt" begin
function test2()
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
        @test isapprox(Matrix(evolve(O,G,θ)),  U'*Matrix(O)*U, atol=1e-14)
        
        
        A = rand(PauliSum{N}, n_paulis=50)
        B = rand(PauliSum{N}, n_paulis=50)
        @test inner_product(A, B) ≈ tr(A'*B)/2^N
        @test norm(A)^2 ≈ inner_product(A,A) 
    end
end

function test()
    Random.seed!(2)

    N = 10 
    p = PauliBasis("IIYZXZ")
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
  
    diff = []
    for trot in 1:10
        for i in 1:length(generators)
            gi = generators[i]
            ai = angles[i]
            O2 = evolve(O2, gi, ai)
            O = evolve(O, gi, ai)
            DBF.weight_clip!(O, 2)
            # DBF.meanfield_reduce!(O,s,2)
            DBF.coeff_clip!(O, thresh=1e-4)
            e = expectation_value(O,s)
            e2 = expectation_value(O2,s)
            push!(diff, e-e2)
            n = norm(O)
            nterms = length(O)
            @printf(" %3i %12.8f %12.8f %12.8f %12i\n", length(generators)*(trot-1)+i, e, e2, n, nterms)
        end 
    end
    @show norm(diff) 
end

test()