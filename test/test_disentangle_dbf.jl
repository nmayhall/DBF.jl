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
    v = reshape(v[:,1], (2^M, 2^(N-M)))
    @show size(v)
    Using,Ssing,Vsing = svd(v)
    println(" Singular Values of V")
    for i in Ssing
        @printf(" %12.8f\n",i)
    end
    @printf(" Entropy: %12.8f\n", sum([-p^2*log(p^2) for p in Ssing]))

    evals1 = eigvals(Matrix(H))
    evals2 = eigvals(Matrix(DBF.p_space(H,M)))
    @printf(" ||H||  = %12.8f\n", norm(H))
    @printf(" ||Hp|| = %12.8f\n", norm(DBF.p_space(H,M)))
    @printf(" ||Hq|| = %12.8f\n", norm(DBF.q_space(H,M)))
    H, gi, θi = dbf_disentangle(H, M, max_iter=1000, conv_thresh=1e-7, evolve_coeff_thresh=1e-6)
   
    Hmat = Matrix(H)
    e,v = eigen(Hmat)
    # println(" Eigenvalues of H")
    # for i in e 
    #     @printf(" %12.8f\n",i)
    # end
    @printf(" <0|H|0> = %12.8f\n", (v'*Hmat*v)[1])
    # v = reshape(v[:,1], ntuple(i->2, N))
    v = reshape(v[:,1], (2^M, 2^(N-M)))
    @show size(v)
    Using,Ssing,Vsing = svd(v)
    println(" Singular Values of V")
    for i in Ssing
        @printf(" %12.8f\n",i)
    end
    @printf(" Entropy: %12.8f\n", sum([-p^2*log(p^2) for p in Ssing]))
end

test()