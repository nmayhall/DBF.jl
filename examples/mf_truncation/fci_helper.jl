# Matrix-free FCI ground state for a PauliSum Hamiltonian on n qubits.
# Dense statevector (length 2^n) Lanczos via KrylovKit; each Pauli term maps every
# basis index i -> i⊻x with a Z-phase, matching PauliOperators' PauliBasis*Ket.
# Feasible to ~n=20 (2^20 ComplexF64 ≈ 16 MB per Krylov vector). Returns
# (E0, overlap0) where overlap0 = |⟨0…0|GS⟩|².
using KrylovKit, LinearAlgebra, Random

function _pauli_terms(H::PauliSum{N,T}) where {N,T}
    zs = Int[]; xs = Int[]; phs = Int[]; cs = ComplexF64[]
    for (p, c) in H
        push!(zs, Int(p.z)); push!(xs, Int(p.x))
        push!(phs, symplectic_phase(p)); push!(cs, ComplexF64(c))
    end
    return zs, xs, phs, cs
end

function fci_ground(H::PauliSum{N,T}; nqubits::Int=N, maxdim::Int=20) where {N,T}
    nqubits > maxdim && return (NaN, NaN)
    dim = 1 << nqubits
    zs, xs, phs, cs = _pauli_terms(H)
    ptbl = ComplexF64[1, im, -1, -im]
    nt = length(cs)
    function matvec(x::Vector{ComplexF64})
        y = zeros(ComplexF64, dim)
        @inbounds for t in 1:nt
            z = zs[t]; xx = xs[t]; base = phs[t]; c = cs[t]
            for i in 0:(dim-1)
                j = i ⊻ xx
                sgn = count_ones(z & j) & 1
                y[j+1] += c * ptbl[((base + 2*sgn) & 3) + 1] * x[i+1]
            end
        end
        return y
    end
    # Random start so Lanczos finds the GLOBAL ground state regardless of which
    # symmetry sector |0…0⟩ lives in (|0…0⟩ can be orthogonal to the GS, e.g. odd L).
    rng = MersenneTwister(1234)
    x0 = randn(rng, ComplexF64, dim)
    vals, vecs, info = eigsolve(matvec, x0, 1, :SR; ishermitian=true)
    gs = vecs[1]
    return real(vals[1]), abs2(gs[1])
end
