using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using Test

function PauliOperators.expectation_value(O::PauliSum, v::KetSum)
    ev = 0
    for (p,c) in O
        for (k1,c1) in v
            ev += expectation_value(p,k1)*c*c1'*c1
            for (k2,c2) in v
                k2 != k1 || continue
                ev += matrix_element(k2', p, k1)*c*c2'*c1
            end
        end
    end
    return ev
end


function run()
    N = 10 
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, -1, -2, -3, x=.1)
    # H = DBF.heisenberg_2D(2, 2, -1, -1, -1, z=.1)
    # H = DBF.heisenberg_2D(7, 7, -0, -0, -1, x=.1)
    DBF.coeff_clip!(H)

    println(" Original H:")
    # display(H)
    
    ψ = Ket([i%2 for i in 1:N])
    # ψ += Ket([i%2 for i in 0:N-1])
    # PauliOperators.scale!(ψ, 1/norm(ψ))

    # kidx = argmin([real(expectation_value(H,Ket{N}(ψi))) for ψi in 1:2^N])
    # ψ = Ket{N}(kidx)
    display(ψ)
    e0 = expectation_value(H,ψ)
    
    @show e1 = minimum(real(eigvals(Matrix(H))))
   
    @printf(" E0 = %12.8f\n", e0)

    @show norm(H)
    
    H, gi, θi = DBF.dbf_groundstate(H, ψ, n_body=1, 
                                max_iter=120, conv_thresh=1e-3, 
                                evolve_coeff_thresh=1e-3,
                                evolve_weight_thresh=5,
                                grad_coeff_thresh=1e-3,
                                # grad_weight_thresh=4,
                                search_n_top=2000)
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