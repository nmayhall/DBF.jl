using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using Test

"""
 Here we provide a test example for application of the Adapt-DBF to a fermionic system, specifically the Hubbard model.
"""
 
function run()
    # Parameters for Hubbard model
    Lx = 1
    Ly = 4
    Nsites = Lx * Ly
    N = 2 * Nsites   # 2 spin states per site
    t = 1.0
    U = 2.0
    #H = DBF.fermi_hubbard_2D(Lx, Ly, t, U)
    H = DBF.hubbard_model_1D(Nsites, t, U)

    println(" Original H:")
    display(H)
    
    Hmat = Matrix(H)
    evals = eigvals(Hmat)
   
    @show minimum(evals)
   

    #ψ = Ket([i%2 for i in 1:N])
    #kidx = argmin([real(expectation_value(H,Ket{N}(ψi))) for ψi in 1:2^N])
    #ψ = Ket{N}(kidx)
    
    ψ = Ket{N}(3)    # half filled state
    display(ψ)
    e0 = expectation_value(H,ψ)
   
    @printf(" E0 = %12.8f\n", e0)

    pool = DBF.qubitexcitationpool(N)

    @show DBF.variance(H,ψ)

    H, gi, θi = adapt(H, pool, ψ, 
                    max_iter=120, conv_thresh=1e-3, 
                    evolve_weight_thresh=6,
                    evolve_coeff_thresh=1e-4)
    
    
    e1 = expectation_value(H,ψ)
    
    @printf(" E = %12.8f\n", e1)
    
    @show DBF.variance(H,ψ)

end


run()