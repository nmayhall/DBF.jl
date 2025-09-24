using DBF
using PauliOperators
using Test
using Printf
using Random
using LinearAlgebra

function test()
# @testset "test_blockdiag_dbf" begin
    N = 6
    M = 3 
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, 1, 2, 3, z=.1)
    DBF.coeff_clip!(H)

    println(" Original H:")
    display(H)


    println(" Projected PHP:")
    Hp = DBF.p_space(H,M)
    display(Hp)

    println(" Projected QHQ:")
    Hq = DBF.q_space(H,M)
    display(Hq)

    @test norm(H - Hp - Hq) ≈ 0

    Hmat = Matrix(H)
    e,v = eigen(Hmat)
    println(" Eigenvalues of H")
    for i in e 
        @printf(" %12.8f\n",i)
    end

    @printf(" <0|H|0> = %12.8f\n", (v'*Hmat*v)[1])
    # v = reshape(v[:,1], ntuple(i->2, N))
    v = reshape(v[:,1], (2^(N-M), 2^M))
    @show size(v)
    Using,Ssing,Vsing = svd(v)
    println(" Singular Values of V")
    for i in Ssing
        @printf(" %12.8f\n",i)
    end
    return

    evals1 = eigvals(Matrix(H))
    evals2 = eigvals(Matrix(diag(H)))
    @printf(" ||H||  = %12.8f\n", norm(H))
    @printf(" ||Hd|| = %12.8f\n", norm(diag(H)))
    @printf(" ||Ho|| = %12.8f\n", norm(DBF.offdiag(H)))
    H, gi, θi = dbf_diag(H, max_iter=1000, conv_thresh=1e-7, evolve_coeff_thresh=1e-4)
    
    println(" New H:")
    display(H)
    evals3 = eigvals(Matrix(H))
    evals4 = eigvals(Matrix(diag(H)))
    @printf(" %3s %12s %12s %12s %12s\n", "Idx", "H", "diag(H)", "U'HU", "diag(U'HU)")
    for i in 1:2^N
        @printf(" %3i %12.8f %12.8f %12.8f %12.8f\n", i, evals1[i], evals2[i], evals3[i], evals4[i])
        @test isapprox(evals1[i], evals3[i], atol=1e-5)
        @test isapprox(evals3[i], evals4[i], atol=1e-5)
    end
end

test()