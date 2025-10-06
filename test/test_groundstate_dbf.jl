using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using Test

# function test()
@testset "test_groundstate_dbf" begin
    N = 3 
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, 1, 2, 3, z=.1)
    DBF.coeff_clip!(H)

    kidx = argmin([real(expectation_value(H,Ket{N}(ψi))) for ψi in 1:2^N])
    ψ = Ket{N}(kidx)
    display(ψ)
    display(expectation_value(H,ψ))
    
    println(" Original H:")
    display(H)
    e1 = minimum(real(eigvals(Matrix(H))))
    e2 = real(expectation_value(H,ψ))
    evals1 = eigvals(Matrix(H))
    evals2 = eigvals(Matrix(diag(H)))
    
    @show DBF.variance(H,ψ)

    H, gi, θi = DBF.dbf_groundstate(H, ψ, n_body=2,
                    max_iter=20, conv_thresh=1e-3, 
                    evolve_coeff_thresh=1e-6,
                    evolve_grad_thresh=1e-10,
                    search_n_top=100)
   
    e3 = real(expectation_value(H,ψ))
    @printf(" E0 = %12.8f <H> = %12.8f <U'HU> = %12.8f \n", e1, e2, e3)
    println(" New H:")
    display(H)
    evals3 = eigvals(Matrix(H))
    evals4 = eigvals(Matrix(diag(H)))
    @printf(" %3s %12s %12s %12s %12s\n", "Idx", "H", "diag(H)", "U'HU", "diag(U'HU)")
    for i in 1:2^N
        @printf(" %3i %12.8f %12.8f %12.8f %12.8f\n", i, evals1[i], evals2[i], evals3[i], evals4[i])
        @test isapprox(evals1[i], evals3[i], atol=1e-8)
    end
    @test isapprox(evals1[1], evals4[1], atol=1e-8)
    @test abs(DBF.variance(H,ψ)) < 1e-6
end

# test()