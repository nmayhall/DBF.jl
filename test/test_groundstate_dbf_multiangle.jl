using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using Test

function test()
# @testset "test_groundstate_dbf" begin
    N = 32 
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, 1, 2, 3, z=.1)
    DBF.coeff_clip!(H)
    H0 = deepcopy(H)

    kidx = argmin([real(expectation_value(H,Ket{N}(ψi))) for ψi in 1:2^N])
    ψ = Ket{N}(kidx)
    display(ψ)
    display(expectation_value(H,ψ))
    
    # e1 = minimum(real(eigvals(Matrix(H))))
    # @printf(" Exact: %12.8f\n", e1)
    # e2 = real(expectation_value(H,ψ))
    # evals1 = eigvals(Matrix(H))
    # evals2 = eigvals(Matrix(diag(H)))
    
    @show DBF.variance(H,ψ)

    res = DBF.dbf_groundstate(H, ψ,
    # res = DBF.dbf_groundstate_multiangle(H, ψ,
                    verbose=1,
                    max_rots_per_grad=10,
                    max_iter=10, conv_thresh=1e-3, 
                    evolve_coeff_thresh=1e-3,
                    grad_coeff_thresh=1e-10,
                    energy_lowering_thresh=1e-10)
  
    H = res["hamiltonian"]
    gi = res["generators"]
    θi = res["angles"]

    @show norm(θi)
end
test()