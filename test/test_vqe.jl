using Test
using Random
using Printf
using PauliOperators
using LinearAlgebra
using DBF

@testset "optimize_angles" begin
    Random.seed!(31)
    N = 6
    H = rand(PauliSum{N}, n_paulis=60); H = H + H'
    ψ = Ket{N}(Int128(0))
    gens = [rand(PauliBasis{N}) for _ in 1:4]

    # the analytic gradient behind optimize_angles must match finite differences
    θ = 0.4 .* randn(4)
    _, g = PauliOperators.expectation_value_gradient(H, gens, θ, ψ)
    h = 1e-5
    fd = [(real(expectation_value(H, gens, (θ .+ h .* (1:4 .== k)), ψ)) -
           real(expectation_value(H, gens, (θ .- h .* (1:4 .== k)), ψ))) / (2h) for k in 1:4]
    @test maximum(abs.(g .- fd)) < 1e-7

    r = optimize_angles(H, gens, ψ; initial_angles=zeros(4), verbose=0)
    # the reported minimum must be the cost actually attained at those angles
    @test abs(r.energy - real(expectation_value(H, gens, r.angles, ψ))) < 1e-10
    # and it must not be worse than the starting point
    @test r.energy <= real(expectation_value(H, gens, zeros(4), ψ)) + 1e-10

    # empty sequence is the bare expectation value
    r0 = optimize_angles(H, PauliBasis{N}[], ψ; verbose=0)
    @test isempty(r0.angles)
    @test abs(r0.energy - real(expectation_value(H, ψ))) < 1e-12

    @test_throws DimensionMismatch optimize_angles(H, gens, ψ; initial_angles=zeros(3), verbose=0)
end

@testset "dbf_vqe" begin
    N = 8
    H = DBF.heisenberg_1D(N, 0.8, 0.8, 1.0, x=0.4, z=0.5)
    ψ = best_reference(H, verbose=0)
    e_hf  = real(expectation_value(H, ψ))
    e_fci = minimum(real(eigvals(Hermitian(Matrix(H)))))
    tr = CoeffTruncation(1e-10)

    out = dbf_vqe(H, ψ; max_iter=6, n_rots=5, verbose=0,
                  operator_truncation=tr, gradient_truncation=tr)

    @test length(out["generators"]) == length(out["angles"])
    for k in ("energies","variances","accumulated_error","accumulated_var_error",
              "norms","norm_error")
        @test length(out[k]) == length(out["energies"])
    end
    @test out["energies"][1] ≈ e_hf
    @test out["energies"][end] < e_hf
    @test out["energies"][end] > e_fci - 1e-8            # variational
    @test abs(real(expectation_value(out["hamiltonian"], ψ)) - out["energies"][end]) < 1e-10

    @test_throws ArgumentError dbf_vqe(H, ψ; max_iter=1, initialization=:bogus, verbose=0)
end

@testset "dbf_vqe reduces to dbf_groundstate" begin
    # With :forwardsweep the angles are exactly those dbf_groundstate picks --
    # each optimized alone against the running operator, in the order evolve()
    # applies them -- so opt_maxiter=0 (no further optimization) must reproduce
    # dbf_groundstate identically. This pins generator selection, tie-breaking,
    # ordering convention and the truncation path all at once; any drift
    # between the two shows up here.
    N = 10
    H = DBF.heisenberg_1D(N, 0.8, 0.8, 1.0, x=0.4, z=0.5)
    ψ = best_reference(H, verbose=0)
    tr = CoeffTruncation(1e-10)
    NIT, NR, ELT = 6, 20, 1e-6

    g = dbf_groundstate(H, ψ; max_iter=NIT, max_rots_per_grad=NR, n_body=1, verbose=0,
                        compute_var_error=false, operator_truncation=tr,
                        gradient_truncation=tr, energy_lowering_thresh=ELT)
    v = dbf_vqe(H, ψ; max_iter=NIT, n_rots=NR, n_body=1, verbose=0,
                initialization=:forwardsweep, opt_maxiter=0,
                operator_truncation=tr, gradient_truncation=tr,
                energy_lowering_thresh=ELT)

    ge = real.(g["energies_per_grad"])
    @test length(v["energies"]) == length(ge)
    @test maximum(abs.(ge .- v["energies"])) < 1e-12
    @test v["generators"] == g["generators"]
    @test maximum(abs.(real.(g["angles"]) .- v["angles"])) < 1e-12

    # and one LBFGS step must move off that solution, lowering the energy
    v1 = dbf_vqe(H, ψ; max_iter=NIT, n_rots=NR, n_body=1, verbose=0,
                 initialization=:forwardsweep, opt_maxiter=1,
                 operator_truncation=tr, gradient_truncation=tr,
                 energy_lowering_thresh=ELT)
    @test v1["energies"][end] < v["energies"][end]
end
