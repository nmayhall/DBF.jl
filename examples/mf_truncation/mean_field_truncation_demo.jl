# Demo: PauliOperators.MeanFieldTruncation inside DBF's dbf_groundstate.
#
# At the same weight budget k, compare:
#   - WeightTruncation(k)      : drop every term of weight > k
#   - MeanFieldTruncation(k, ψ): fold those terms back onto weight <= k, so
#                                <ψ|O|ψ> is preserved at each truncation step
# against the exact ground-state energy (eigmin of the dense Hamiltonian).
#
# Run:  julia --project=. mean_field_truncation_demo.jl

using PauliOperators
using DBF
using LinearAlgebra
using Printf

function build_hamiltonian(N)
    H = DBF.heisenberg_1D(N, 1, 1, 1)          # nearest-neighbour Heisenberg
    for i in 1:N
        H += 0.30 * Pauli(N, Z=[i])            # a field to break degeneracy
        H += 0.15 * Pauli(N, X=[i])
    end
    coeff_clip!(H, 1e-16)
    return H
end

function run_case(Oin, ψ, label, operator_truncation; k_iter=60)
    res = DBF.dbf_groundstate(Oin, ψ;
            max_iter=k_iter, verbose=0, conv_thresh=1e-5,
            operator_truncation=operator_truncation,
            gradient_truncation=CoeffTruncation(1e-6))
    ae = get(res, "accumulated_error", NaN)
    acc_err = ae isa AbstractVector ? (isempty(ae) ? NaN : real(ae[end])) : real(ae)
    return (label = label,
            energy = real(res["energies"][end]),
            variance = real(res["variances"][end]),
            nterms = length(res["hamiltonian"]),
            acc_err = acc_err)
end

function main()
    N = 8
    k = 3                                        # weight budget

    H = build_hamiltonian(N)
    Eexact = eigmin(Hermitian(Matrix(H)))        # dense reference (N=8 -> 256x256)

    # reference computational-basis state: lowest-diagonal-energy determinant
    kidx = argmin([real(expectation_value(H, Ket{N}(i))) for i in 1:2^N])
    ψ = Ket{N}(kidx)

    @printf("N = %d qubits,  weight budget k = %d\n", N, k)
    @printf("exact ground-state energy      = %14.8f\n", Eexact)
    @printf("reference determinant <ψ|H|ψ>  = %14.8f\n\n", real(expectation_value(H, ψ)))

    # PauliSum (Dict) engine vs SparsePauliVector (flat) engine, same strategies.
    ps() = deepcopy(H)
    spv() = SparsePauliVector(H)
    cases = [
        run_case(ps(),  ψ, "PauliSum  WeightTruncation($k)",       WeightTruncation(k)),
        run_case(ps(),  ψ, "PauliSum  MeanFieldTruncation($k)",    MeanFieldTruncation(k, ψ)),
        run_case(spv(), ψ, "SPV       WeightTruncation($k)",       WeightTruncation(k)),
        run_case(spv(), ψ, "SPV       MeanFieldTruncation($k)",    MeanFieldTruncation(k, ψ)),
    ]

    @printf("%-34s %14s %12s %10s %12s\n", "engine / strategy", "energy", "err vs exact", "nterms", "acc_err")
    for c in cases
        @printf("%-34s %14.8f %12.2e %10d %12.2e\n",
                c.label, real(c.energy), abs(real(c.energy) - Eexact), c.nterms, real(c.acc_err))
    end
end

main()
