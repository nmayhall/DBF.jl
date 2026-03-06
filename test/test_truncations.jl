using DBF
using PauliOperators
using Random
using Test

@testset "Truncation Strategies" begin
    N = 4

    # Build a small PauliSum with known coefficients
    H = PauliSum(N)
    H[PauliBasis(Pauli(N, X=[1]))] = 0.5
    H[PauliBasis(Pauli(N, Z=[1]))] = 1.0
    H[PauliBasis(Pauli(N, X=[1], Z=[2]))] = 0.01    # weight-2
    H[PauliBasis(Pauli(N, X=[1], Z=[2,3]))] = 0.001  # weight-3
    H[PauliBasis(Pauli(N, Z=[1,2,3,4]))] = 0.0001    # weight-4

    @testset "NoTruncation" begin
        O = deepcopy(H)
        truncate!(O, NoTruncation())
        @test length(O) == length(H)
    end

    @testset "CoeffTruncation" begin
        O = deepcopy(H)
        truncate!(O, CoeffTruncation(0.005))
        # Should remove terms with |c| < 0.005: the 0.001 and 0.0001 terms
        @test length(O) == 3
        @test haskey(O, PauliBasis(Pauli(N, X=[1])))
        @test haskey(O, PauliBasis(Pauli(N, Z=[1])))
        @test haskey(O, PauliBasis(Pauli(N, X=[1], Z=[2])))
    end

    @testset "WeightTruncation" begin
        O = deepcopy(H)
        truncate!(O, WeightTruncation(2))
        # Should keep weight-1 and weight-2 terms, remove weight-3 and weight-4
        @test length(O) == 3
        @test !haskey(O, PauliBasis(Pauli(N, X=[1], Z=[2,3])))
        @test !haskey(O, PauliBasis(Pauli(N, Z=[1,2,3,4])))
    end

    @testset "MajoranaWeightTruncation" begin
        O = deepcopy(H)
        n_before = length(O)
        truncate!(O, MajoranaWeightTruncation(100))  # high threshold, should keep all
        @test length(O) == n_before
    end

    @testset "CompositeTruncation" begin
        O = deepcopy(H)
        strat = CompositeTruncation(CoeffTruncation(0.0005), WeightTruncation(2))
        truncate!(O, strat)
        # CoeffTruncation(0.0005) removes the 0.0001 term, WeightTruncation(2) removes weight-3
        @test length(O) == 3

        # Verify varargs constructor
        strat2 = CompositeTruncation(CoeffTruncation(0.0005), WeightTruncation(2))
        O2 = deepcopy(H)
        truncate!(O2, strat2)
        @test length(O2) == length(O)
    end

    @testset "StochasticTruncation" begin
        Random.seed!(42)
        O = deepcopy(H)
        strat = StochasticTruncation(3, Random.MersenneTwister(42))
        n_before = norm(O)
        truncate!(O, strat)
        @test length(O) == 3
        # Norm should be rescaled to match original
        @test norm(O) ≈ n_before atol=1e-10
    end
end

@testset "Correction Accumulators" begin
    N = 3
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, 1, 2, 3, z=0.1)
    DBF.coeff_clip!(H)

    kidx = argmin([real(expectation_value(H, Ket{N}(ψi))) for ψi in 1:2^N])
    ψ = Ket{N}(kidx)

    @testset "NoCorrection" begin
        O = deepcopy(H)
        O_ref = deepcopy(H)
        truncate!(O, CoeffTruncation(0.1))
        DBF.coeff_clip!(O_ref, thresh=0.1)
        # Should produce identical results to raw coeff_clip!
        for (k, v) in O
            @test O_ref[k] ≈ v
        end
        @test length(O) == length(O_ref)
    end

    @testset "EnergyCorrection" begin
        O = deepcopy(H)
        e_before = real(expectation_value(O, ψ))
        corr = EnergyCorrection(ψ)
        truncate!(O, CoeffTruncation(0.1), corr)
        e_after = real(expectation_value(O, ψ))

        # accumulated_energy should equal e_after - e_before
        @test corr.accumulated_energy ≈ (e_after - e_before)

        # Apply a second truncation and verify accumulation
        e_before2 = e_after
        truncate!(O, CoeffTruncation(0.5), corr)
        e_after2 = real(expectation_value(O, ψ))
        @test corr.accumulated_energy ≈ (e_after - e_before) + (e_after2 - e_before2)
    end

    @testset "EnergyVarianceCorrection" begin
        O = deepcopy(H)
        e_before = real(expectation_value(O, ψ))
        v_before = real(DBF.variance(O, ψ))
        corr = EnergyVarianceCorrection(ψ)
        truncate!(O, CoeffTruncation(0.1), corr)
        e_after = real(expectation_value(O, ψ))
        v_after = real(DBF.variance(O, ψ))

        @test corr.accumulated_energy ≈ (e_after - e_before)
        @test corr.accumulated_variance ≈ (v_after - v_before)
    end
end
