using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using Test

function run()
    N = 50 
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, -1, -1, -1, z=.1)
    # H = DBF.heisenberg_2D(2, 2, -1, -1, -1, z=.1)
    # H = DBF.heisenberg_2D(7, 7, -1, -1, -1, z=.1)
    DBF.coeff_clip!(H)

    println(" Original H:")
    # display(H)
    
    ψ = Ket([i%2 for i in 1:N])
    # kidx = argmin([real(expectation_value(H,Ket{N}(ψi))) for ψi in 1:2^N])
    # ψ = Ket{N}(kidx)
    display(ψ)
    e0 = expectation_value(H,ψ)
   
    @printf(" E0 = %12.8f\n", e0)

    pool1 = DBF.generate_pool_1_weight(N)
    pool2 = DBF.generate_pool_2_weight(N)
    pool = vcat(pool1, pool2)

    H, gi, θi = adapt(H, pool, ψ, 
                    max_iter=120, conv_thresh=1e-3, 
                    evolve_weight_thresh=6,
                    evolve_coeff_thresh=1e-4)
    
    
    e1 = expectation_value(H,ψ)
    
    @printf(" E1 = %12.8f\n", e1)

end


run()