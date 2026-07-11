using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using Test

@testset "spv unit equivalence" begin
    for (N, args, kwargs) in [(3, (1, 2, 3), (z=.1,)), (6, (-1, -2, -3), (x=.3,))]
        Random.seed!(2)
        H = DBF.heisenberg_1D(N, args...; kwargs...)
        coeff_clip!(H, 1e-16)
        kidx = argmin([real(expectation_value(H, Ket{N}(ψi))) for ψi in 1:2^N])
        ψ = Ket{N}(kidx)
        Hspv = SparsePauliVector(H)

        # matvec: single-pass SPV kernel vs packed XZPauliSum path
        σ_ps = DBF.matvec(H, ψ)
        σ_spv = DBF.matvec(Hspv, ψ)
        @test length(σ_ps) == length(σ_spv)
        for (k, v) in σ_ps
            @test isapprox(get(σ_spv, k, zero(v)), v, atol=1e-13)
        end

        # pt2
        e0_ps, e2_ps = DBF.pt2(H, ψ)
        e0_spv, e2_spv = DBF.pt2(Hspv, ψ)
        @test e0_ps ≈ e0_spv
        @test e2_ps ≈ e2_spv

        # commutator_clipped
        P = DBF.create_0_projector(N, 1)
        G_ps = DBF.commutator_clipped(P, H)
        G_spv = PauliSum(DBF.commutator_clipped(SparsePauliVector(P), Hspv))
        @test length(G_ps) == length(G_spv)
        for (p, c) in G_ps
            @test isapprox(get(G_spv, p, zero(c)), c, atol=1e-12)
        end
    end
end

@testset "test_groundstate_spv" begin
    N = 3
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, 1, 2, 3, z=.1)
    # Break translation symmetry: the uniform ring has exactly degenerate
    # gradient magnitudes, so tied generators get rotated in backend-dependent
    # order (stable sortperm keeps Dict vs sorted insertion order) and the two
    # trajectories legitimately diverge. Site-dependent fields remove the ties.
    for i in 1:N
        H += 0.05 * i * Pauli(N, Z=[i])
        H += 0.03 * i * Pauli(N, X=[i])
    end
    coeff_clip!(H, 1e-16)

    kidx = argmin([real(expectation_value(H, Ket{N}(ψi))) for ψi in 1:2^N])
    ψ = Ket{N}(kidx)

    evals1 = eigvals(Matrix(H))

    checkfile = joinpath(mktempdir(), "spv_chk")
    flow_kwargs = (conv_thresh=1e-3,
                   operator_truncation=CoeffTruncation(1e-6),
                   gradient_truncation=CoeffTruncation(1e-6),
                   energy_lowering_thresh=1e-6)

    # --- Strict step-for-step equivalence over a fixed short horizon.
    # Near convergence the surviving gradient magnitudes hover at the filter
    # threshold, so fp-level backend drift inevitably flips keep/drop decisions
    # there; before that tail the trajectories must match exactly.
    res_ps = DBF.dbf_groundstate(H, ψ; max_iter=3, flow_kwargs...)
    res_spv = DBF.dbf_groundstate(SparsePauliVector(H), ψ; max_iter=3, flow_kwargs...)

    @test res_spv["hamiltonian"] isa SparsePauliVector

    # Energies are stationary at the optimized θ, so they match tightly;
    # variances/norms inherit the flat-minimum θ jitter linearly
    for (key, rtol) in [("energies", 1e-8), ("norms", 1e-8),
                        ("variances", 1e-6), ("accumulated_error", 1e-6)]
        @test length(res_ps[key]) == length(res_spv[key])
        @test isapprox(res_ps[key], res_spv[key], rtol=rtol, atol=1e-9)
    end
    @test res_ps["generators"] == res_spv["generators"]
    # Angles are 2π-periodic (θ≈0 and θ≈2π are the same rotation) and near-zero
    # minima are flat, so compare wrapped with a loose absolute tolerance
    @test length(res_ps["angles"]) == length(res_spv["angles"])
    for (a, b) in zip(res_ps["angles"], res_spv["angles"])
        @test abs(mod(a - b + π, 2π) - π) < 1e-6
    end

    # Coefficients inherit the flat-minimum angle jitter linearly (energies are
    # stationary in θ, coefficients are not), so the tolerance follows the
    # angle tolerance
    Hf_ps = res_ps["hamiltonian"]
    Hf_spv = PauliSum(res_spv["hamiltonian"])
    @test length(Hf_ps) == length(Hf_spv)
    for (p, c) in Hf_ps
        @test isapprox(get(Hf_spv, p, zero(c)), c, atol=1e-5)
    end

    # --- Full-flow convergence: same physics from both backends
    res_ps = DBF.dbf_groundstate(H, ψ; max_iter=30, flow_kwargs...)
    res_spv = DBF.dbf_groundstate(SparsePauliVector(H), ψ; max_iter=30, flow_kwargs...,
                                  checkfile=checkfile)

    @test isfile("$(checkfile).jld2")
    @test isapprox(res_ps["energies"][end], res_spv["energies"][end], atol=1e-5)

    # Eigenvalues preserved (up to truncation error) by the SPV flow
    evals3 = eigvals(Matrix(PauliSum(res_spv["hamiltonian"])))
    for i in 1:2^N
        @test isapprox(evals1[i], evals3[i], atol=1e-5)
    end
    @test abs(variance(res_spv["hamiltonian"], ψ)) < 1e-4
    @test abs(variance(res_ps["hamiltonian"], ψ)) < 1e-4
end
