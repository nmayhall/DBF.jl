#
# Performance comparison: dbf_groundstate with PauliSum vs SparsePauliVector.
# Driver script, not part of the CI suite. The TimerOutput printed by each run
# gives the per-section breakdown (commutator / matvec / evolve / clip / ...).
#
using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra

function run(; N=50, max_iter=20, run_paulisum=false)
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, -1, -1, -1, x=.1)
    coeff_clip!(H, 1e-16)

    # Transform H to make |000...> the most stable bitstring
    for i in 1:N
        if i % 2 == 0
            H = Pauli(N, X=[i]) * H * Pauli(N, X=[i])
        end
    end

    ψ = Ket([0 for i in 1:N])
    @printf(" N = %i   E0 = %12.8f   len(H) = %i\n", N, real(expectation_value(H, ψ)), length(H))

    flow_kwargs = (verbose=1, max_iter=max_iter, conv_thresh=1e-3,
                   operator_truncation=CoeffTruncation(1e-5),
                   gradient_truncation=CoeffTruncation(1e-4),
                #    energy_lowering_thresh=1e-5,
                   compute_var_error=false)

    res_ps = nothing
    t_ps = 0.0
    if run_paulisum
        println("\n======== PauliSum backend ========")
        t_ps = @elapsed res_ps = DBF.dbf_groundstate(H, ψ; flow_kwargs...)
    end

    println("\n======== SparsePauliVector backend ========")
    t_spv = @elapsed res_spv = DBF.dbf_groundstate(SparsePauliVector(H), ψ; flow_kwargs...)

    if run_paulisum
        e_ps = res_ps["energies"][end]
        e_spv = res_spv["energies"][end]
        @printf("\n Final energy   PauliSum: %16.10f   SPV: %16.10f   diff: %8.1e\n",
                e_ps, e_spv, abs(e_ps - e_spv))
        @printf(" Wall time      PauliSum: %10.2fs   SPV: %10.2fs   speedup: %.1fx\n",
                t_ps, t_spv, t_ps / t_spv)
    else
        @printf("\n Final energy   SPV: %16.10f\n", res_spv["energies"][end])
        @printf(" Wall time      SPV: %10.2fs   (run(run_paulisum=true) for the comparison)\n", t_spv)
    end
    return res_ps, res_spv
end

run(run_paulisum=true)
