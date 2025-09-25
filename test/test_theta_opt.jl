using PauliOperators
using DBF
using LinearAlgebra
using Printf
using Random
using Test

@testset "theta_opt" begin
    Random.seed!(2)
    N = 7
    O = rand(PauliSum{N}, n_paulis=100)
    O += O'
    G = rand(PauliBasis{N})
    ψ = Ket{N}(0)
    split_idx = 3
    # display(O)
    # display(G)

    step = .0001

    θ_diag, cost_diag = DBF.optimize_theta_diagonalization(O,G; stepsize=step, verbose=0)
    @show θ_diag, cost_diag(θ_diag)

    θ_eval, cost_eval = DBF.optimize_theta_expval(O, G, ψ; stepsize=step, verbose=0)
    @show θ_eval, cost_eval(θ_eval)

    θ_dtgl, cost_dtgl = DBF.optimize_theta_disentangle(O, G, split_idx; stepsize=step, verbose=0)
    @show θ_dtgl, cost_dtgl(θ_dtgl)



    for i in 0:.1:1
        θ = i*2π
        cost1_ref = norm(diag(evolve(O,G,θ)))^2
        cost2_ref = expectation_value(evolve(O,G,θ), ψ)
        cost3_ref = norm(DBF.p_space(evolve(O,G,θ), split_idx))^2

        cost1_test = cost_diag(θ)
        cost2_test = cost_eval(θ)
        cost3_test = cost_dtgl(θ)

        @printf(" %5.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n", θ, real(cost1_ref), cost1_test, real(cost2_ref), cost2_test, real(cost3_ref), cost3_test)

        # Make sure the give the correct values
        @test abs(cost1_ref - cost1_test) < 1e-12
        @test abs(cost2_ref - cost2_test) < 1e-12
        @test abs(cost3_ref - cost3_test) < 1e-12

        # Make sure we have correct periodicity
        @test abs(cost_diag(θ) - cost_diag(θ + π)) < 1e-12 
        @test abs(cost_eval(θ) - cost_eval(θ + 2π)) < 1e-12 
        @test abs(cost_dtgl(θ) - cost_dtgl(θ + π)) < 1e-12 
    end


    # Make sure the optimal values are correctly found
    @test cost_diag(θ_diag) - cost_diag(θ_diag + step) >= 0
    @test cost_diag(θ_diag) - cost_diag(θ_diag - step) >= 0
    @test cost_eval(θ_eval) - cost_eval(θ_eval + step) <= 0
    @test cost_eval(θ_eval) - cost_eval(θ_eval - step) <= 0
    @test cost_dtgl(θ_dtgl) - cost_dtgl(θ_dtgl + step) >= 0
    @test cost_dtgl(θ_dtgl) - cost_dtgl(θ_dtgl - step) >= 0
    
end

# test1()