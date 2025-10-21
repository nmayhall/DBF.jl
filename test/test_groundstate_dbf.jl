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
    H0 = deepcopy(H)

    kidx = argmin([real(expectation_value(H,Ket{N}(ψi))) for ψi in 1:2^N])
    ψ = Ket{N}(kidx)
    display(ψ)
    display(expectation_value(H,ψ))
    
    # println(" Original H:")
    # display(H)
    e1 = minimum(real(eigvals(Matrix(H))))
    e2 = real(expectation_value(H,ψ))
    evals1 = eigvals(Matrix(H))
    evals2 = eigvals(Matrix(diag(H)))
    
    @show DBF.variance(H,ψ)

    # H, gi, θi = DBF.dbf_groundstate(H, ψ, n_body=2,
    #                 max_iter=20, conv_thresh=1e-3, 
    #                 evolve_coeff_thresh=1e-6,
    #                 grad_coeff_thresh=1e-10,
    #                 search_n_top=100)
    H, gi, θi = DBF.dbf_groundstate(H, ψ, n_body=2,
                    max_iter=20, conv_thresh=1e-3, 
                    evolve_coeff_thresh=1e-6,
                    grad_coeff_thresh=1e-10,
                    energy_lowering_thresh=1e-10)
   
    e3 = real(expectation_value(H,ψ))
    @printf(" E0 = %12.8f <H> = %12.8f <U'HU> = %12.8f \n", e1, e2, e3)
    # println(" New H:")
    # display(H)
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
@testset "pt2" begin 
    N = 6 
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, -1, -2, -3, x=.3)
    DBF.coeff_clip!(H)
    kidx = argmin([real(expectation_value(H,Ket{N}(ψi))) for ψi in 1:2^N])
    ψ = Ket{N}(kidx)
    H0 = deepcopy(H)
    @show norm(H)^2, norm(diag(H))^2, norm(offdiag(H))^2
    @show e_exact = minimum(real(eigvals(Matrix(H))))
    @show e_ref = expectation_value(H,ψ) 
    # H, gi, θi = DBF.dbf_diag(H0,
    #                 max_iter=20, conv_thresh=1e-3, 
    #                 evolve_coeff_thresh=1e-6,
    #                 search_n_top=1000)
    H, gi, θi = DBF.dbf_groundstate(H, ψ, n_body=1,
                    max_iter=20, conv_thresh=1e-3, 
                    evolve_coeff_thresh=1e-6,
                    grad_coeff_thresh=1e-3,
                    energy_lowering_thresh=1e-3)
    
    # H = deepcopy(H0)
    # Compute PT2 explicitly with matrix
    basis_dict = H*ψ
    basis = Vector{Ket{N}}([ψ])
    for (k,c) in basis_dict
        k != ψ || continue 
        push!(basis, k)
    end
    
    Hmat = Matrix(H,basis)
    display(eigvals(Hmat)) 
    display(basis[1])
    e2ref = 0
    for i in 2:length(basis)
        basis[i] != ψ || error(" ψ == i")
        e2ref += Hmat[1,i]*Hmat[i,1] / (Hmat[1,1] - Hmat[i,i])
    end
    @show e2ref

    println("\n Compute PT2 correction")
    e0, e2 = DBF.pt2(H, ψ)
    @printf(" E0 = %12.8f E2 = %12.8f EPT2 = %12.8f \n", e0, e2, e0+e2)
    @test e2 ≈ e2ref
    @show norm(H)^2, norm(diag(H))^2, norm(offdiag(H))^2
    
    # CEPA?
    basis_dict = H*ψ
    DBF.coeff_clip!(basis_dict, thresh=1e-2)
    @show length(basis_dict)
    
    basis = Vector{Ket{N}}([ψ])
    for (k,c) in basis_dict
        k != ψ || continue 
        push!(basis, k)
    end
    
    Hmat = Matrix(H,basis)
    display(eigvals(Hmat)) 
    A = diagm([Hmat[1,1] for i in 2:length(basis)]) - Hmat[2:end,2:end]
    # ecepa = size(inv(A)*Hmat[2:end,1])
    ecepa = Hmat[1,2:end]'*inv(A)*Hmat[2:end,1]
    @show real(e0), real(ecepa), real(e0) + real(ecepa)

    # Hmat2 = DBF.Matrix2(H,basis)
    # @test norm(Hmat-Hmat2) < 1e-13
    # @time Hmat = Matrix(H,basis)
    # @time Hmat2 = DBF.Matrix2(H,basis)
end

# test()