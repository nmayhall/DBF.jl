using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using Test
using JLD2

# A run split in two and resumed from a snapshot must reproduce the
# uninterrupted run exactly -- not approximately. The flow is deterministic, so
# any difference means the restored state is incomplete: a missing accumulator
# would show up as drifting truncation error, and a mis-ordered operator would
# change which generator the gradient picks.
@testset "resume snapshots" begin
    N = 6
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, 1, 2, 3; z=0.1)
    coeff_clip!(H, 1e-16)
    kidx = argmin([real(expectation_value(H, Ket{N}(ψi))) for ψi in 1:2^N])
    ψ = Ket{N}(kidx)

    kw = (n_body=1, conv_thresh=1e-12,
          operator_truncation=CoeffTruncation(1e-4),
          gradient_truncation=CoeffTruncation(1e-6),
          max_rots_per_grad=5, compute_var_error=true, verbose=0)

    rf = tempname() * ".jld2"
    try
        full = dbf_groundstate(SparsePauliVector(H), ψ; max_iter=12, kw...)

        # stop at 6, then continue to 12 through the snapshot
        dbf_groundstate(SparsePauliVector(H), ψ;
                        max_iter=6, resume_file=rf, resume_stride=3, kw...)
        @test isfile(rf)
        split = dbf_groundstate(SparsePauliVector(H), ψ;
                                max_iter=12, resume_file=rf, resume_stride=3, kw...)

        @test split["generators"] == full["generators"]
        @test split["angles"] == full["angles"]
        @test length(split["energies"]) == length(full["energies"])
        @test real.(split["energies"]) == real.(full["energies"])
        @test real.(split["variance_per_grad"]) == real.(full["variance_per_grad"])
        # accumulated truncation error is the state most easily lost on restore
        @test split["accumulated_error"][end] == full["accumulated_error"][end]
        @test split["accumulated_var_error"][end] == full["accumulated_var_error"][end]

        # a completed run leaves a snapshot; re-running must not redo any work
        again = dbf_groundstate(SparsePauliVector(H), ψ;
                                max_iter=12, resume_file=rf, resume_stride=3, kw...)
        @test again["generators"] == full["generators"]
    finally
        rm(rf; force=true)
        rm(rf * ".tmp"; force=true)
    end
end

# The SparsePauliVector round trip must preserve both the terms and their
# canonical ordering -- only the live slices are written, so a capacity or
# ordering slip would silently corrupt the operator.
@testset "resume operator round trip" begin
    N = 6
    H = DBF.heisenberg_1D(N, 1, 2, 3; z=0.1)
    spv = SparsePauliVector(H)
    restored = DBF._restore_operator(DBF._snapshot_operator(spv))
    @test restored.n == spv.n
    @test restored.z[1:restored.n] == spv.z[1:spv.n]
    @test restored.x[1:restored.n] == spv.x[1:spv.n]
    @test restored.c[1:restored.n] == spv.c[1:spv.n]

    ψ = Ket{N}(1)
    @test expectation_value(restored, ψ) == expectation_value(spv, ψ)
    @test norm(restored) == norm(spv)
end
