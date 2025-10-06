using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using Test

function run()
    N = 49 
    Random.seed!(2)
    # H = DBF.heisenberg_1D(N, -1, -2, -3, x=.1)
    # H = DBF.heisenberg_2D(2, 2, -1, -1, -1, z=.1)
    H = DBF.heisenberg_2D(7, 7, -0, -0, -1, x=.1)
    DBF.coeff_clip!(H)

    println(" Original H:")
    # display(H)
    
    ψ = Ket([i%2 for i in 1:N])
    # kidx = argmin([real(expectation_value(H,Ket{N}(ψi))) for ψi in 1:2^N])
    # ψ = Ket{N}(kidx)
    display(ψ)
    e0 = expectation_value(H,ψ)
    
    # @show e1 = minimum(real(eigvals(Matrix(H))))
   
    @printf(" E0 = %12.8f\n", e0)

    display(norm(H))
    display(norm(diag(H)))
    H, gi, θi = DBF.dbf_groundstate(H, ψ, n_body=1, 
                                max_iter=120, conv_thresh=1e-3, 
                                evolve_coeff_thresh=1e-5,
                                evolve_weight_thresh=49,
                                evolve_grad_thresh=1e-8,
                                search_n_top=1000)
    # H, gi, θi = DBF.dbf_groundstate_old(H, ψ, 
    #                             max_iter=120, conv_thresh=1e-3, 
    #                             evolve_coeff_thresh=1e-4,
    #                             evolve_weight_thresh=3,
    #                             search_n_top=100)
    
    println(" New H:")
    display(norm(H))
    display(norm(diag(H)))
    
    e1 = expectation_value(H,ψ)
    
    @printf(" E1 = %12.8f\n", e1)

end


run()