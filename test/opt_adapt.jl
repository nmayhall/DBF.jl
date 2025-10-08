using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using Test

function run()
    # N = 49 
    N = 6 
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, -1, -1, -1, x=.1)
    # H = DBF.heisenberg_2D(2, 2, -1, -1, -1, z=.1)
    # H = DBF.heisenberg_2D(7, 7, -0, -0, -1, x=.1)
    # H = DBF.heisenberg_2D(9, 1, -0, -0, -1, x=.1)
    # H = DBF.heisenberg_1D(N, -1, -2, -3, x=.1)
    DBF.coeff_clip!(H)

    
    ψ = Ket([i%2 for i in 1:N])
    # kidx = argmin([real(expectation_value(H,Ket{N}(ψi))) for ψi in 1:2^N])
    # ψ = Ket{N}(kidx)
    display(ψ)
    e0 = expectation_value(H,ψ)
    
    Hmat = Matrix(H)
    evals = eigvals(Hmat)
    @show minimum(evals)
    @show expectation_value(H,ψ)
    @show norm(H)
   
    @printf(" E0 = %12.8f\n", e0)

    pool = DBF.generate_pool_1_weight(N)
    pool = vcat(pool, DBF.generate_pool_2_weight(N))
    pool = vcat(pool, DBF.generate_pool_3_weight(N))
    pool = vcat(pool, DBF.generate_pool_4_weight(N))
    pool = vcat(pool, DBF.generate_pool_5_weight(N))
    pool = vcat(pool, DBF.generate_pool_6_weight(N))

    pool = DBF.generate_commutator_pool(H)
    # search_n_top = 100
    # # pool = DBF.pool_test1(H)
    # pool = DBF.max_of_commutator2(H, H, n_top=search_n_top)
    # DBF.coeff_clip!(pool)
    # pool = [first(x) for x in sort(collect(pool), by = x -> abs(last(x)))]
  
    # for i in 1:4
    # pool = H*diag(H)-diag(H)*H
    # DBF.coeff_clip!(pool)
    # pool = [first(x) for x in sort(collect(pool), by = x -> abs(last(x)))]

    @printf(" Size of pool: %12i\n", length(pool))
    
    @show DBF.variance(H,ψ)
    @show norm(H)


    # H, gi, θi = adapt(H, pool, ψ, 
    #                 max_iter=120, conv_thresh=1e-3, 
    #                 evolve_weight_thresh=8,
    #                 evolve_coeff_thresh=1e-5)
    
    for i in 1:10
        pool = DBF.generate_commutator_pool(H)
        H, gi, θi = adapt(H, pool, ψ, 
                        max_iter=2, conv_thresh=1e-3, 
                        evolve_weight_thresh=8,
                        evolve_coeff_thresh=1e-6,
                        grad_coeff_thresh=1e-4)
    end
    # end
    # println("")
    # println(" Now update pool")
    # pool = DBF.max_of_commutator2(H, H, n_top=search_n_top)
    # DBF.coeff_clip!(pool)
    # pool = [first(x) for x in sort(collect(pool), by = x -> abs(last(x)))]
    
    # @printf(" Size of pool: %12i\n", length(pool))
    
    # @show DBF.variance(H,ψ)
    # @show norm(H)


    # H, gi, θi = adapt(H, pool, ψ, 
    #                 max_iter=120, conv_thresh=1e-3, 
    #                 evolve_weight_thresh=7,
    #                 evolve_coeff_thresh=1e-6)
    
    
    e1 = expectation_value(H,ψ)
    
    @printf(" E = %12.8f\n", e1)
    
    @show DBF.variance(H,ψ)

end


run()