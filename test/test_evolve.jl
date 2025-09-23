using Test
using Random

@testset "theta_opt" begin
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
        @test overlap(A, B) ≈ tr(A'*B)/2^N
        @test norm(A)^2 ≈ overlap(A,A) 
    end
end