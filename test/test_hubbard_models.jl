using PauliOperators
using DBF
using LinearAlgebra
using Printf

############## TESTING PARAMETERS #################
# Test Hubbard 2D
println("Major-Row Ordering: Hubbard 2D Hamiltonian")
Lx, Ly = 2, 2
t, U = 1.00, 1.0
H = DBF.fermi_hubbard_2D(Lx, Ly, t, U)   # your function (returns PauliSum)
N_total = 2 * Lx * Ly


################ FUNCTIONS #######################################################
# --- Build number operator (sum over modes) ---------------------------------
"""
    number_operator(N_total)

Return the PauliSum representing the total particle-number operator N̂ = sum_k c_k^† c_k,
where each mode k is mapped via `JWmapping(N_total, i=k, j=k)`.
"""
function number_operator(N_total::Integer)
    Nop = PauliSum(N_total)   # Dict{PauliBasis{N_total}, ComplexF64} per your alias
    for k in 1:N_total
        # JWmapping(N_total, i=k, j=k) should return a PauliSum (or Pauli-like object)
        term = DBF.JWmapping(N_total, i=k, j=k)
        # Make sure we add as PauliSum-style; `term` might already be a PauliSum or Pauli
        Nop += term
    end
    return Nop
end


# --- Build per-site (spin-summed) number operator (for Hubbard lattice) -----
"""
    site_number_operators(Lx, Ly)

Return a Vector of PauliSums `site_n[s]` (length = Lx*Ly) where each entry is the
number operator for that physical *site* (n_up + n_dn). Uses your up/dn indexing convention:
  up(j) = 2*j - 1
  dn(j) = 2*j
"""
function site_number_operators(Lx::Int, Ly::Int)
    Nsites = Lx * Ly
    N_total = 2 * Nsites
    up(j) = 2*j - 1
    dn(j) = 2*j
    site_ns = Vector{PauliSum{typeof(N_total), ComplexF64}}(undef, Nsites)
    for i in 1:Nsites
        a_up = up(i)
        a_dn = dn(i)
        # each JWmapping(i,i) is n_mode for that spin-orbital
        n_up = DBF.JWmapping(N_total, i=a_up, j=a_up)
        n_dn = DBF.JWmapping(N_total, i=a_dn, j=a_dn)
        site_ns[i] = n_up + n_dn
    end
    return site_ns
end


# --- Commutation test (symbolic/PauliSum-level when possible; fallback to matrices) ---
"""
    commutes_with(H, O; tol=1e-12, small_dim_limit=16)

Return (commutes::Bool, norm_est) where `commutes` is true if ‖[H,O]‖ < tol.
This attempts to compute the commutator at the PauliSum level (H*O - O*H) if `*` is defined
for PauliSum. If PauliSum multiplication is not available in your environment, it falls back
to dense-matrix conversion for small total qubit counts (2^N_total dimension).
"""
function commutes_with(H, O; tol=1e-12, small_dim_limit=16)
    try
        # attempt algebraic commutator using your PauliSum operations
        C = H * O - O * H
        # drop tiny coefficients if you have DBF.coeff_clip!
        if typeof(C) <: PauliSum
            DBF.coeff_clip!(C, thresh=tol)   # if DBF available; otherwise ignore
            # if container empty => commutes
            isempty(C) && return (true, 0.0)
            # otherwise compute an l1 norm of coefficients
            s = zero(Float64)
            for (_, coeff) in C
                s += abs(coeff)
            end
            return (s < tol, s)
        end
        # else fall through to matrix fallback
    catch err
        # if algebraic ops not available, we'll do matrix fallback
        # (don't rethrow; the fallback is safer)
    end

    # Matrix fallback: convert both to dense and compute operator norm of the commutator
    # Expect `H` and `O` to be PauliSums with a known N; try to infer N_total
    N_total = first(keys(H)).x |> (z->begin  # try to find a PauliBasis and infer N bits
        # fallback if that fails
        try
            # this is fragile if dict keys aren't PauliBasis{N}
            kv = first(keys(H))
            # we need N; rely on your PauliBasis constructor or external information:
            # instead, user should pass O/H built from same N; we can deduce N_total
            return nothing
        catch
            return nothing
        end
    end)

    M_H = Matrix(H)   # defined below
    M_O = Matrix(O)

    Cmat = M_H * M_O - M_O * M_H
    normC = opnorm(Cmat)  # 2-norm
    return (normC < tol, normC)
end


# --- Convert PauliSum -> dense matrix (ONLY for small N_total) ----------------
"""
    paulisum_to_matrix(Ps::PauliSum{N, T})

Convert a PauliSum into a dense complex matrix by summing coefficients * pauli_matrix(pb).
Warning: dimension = 2^N (exponential). Use only for small systems (N_total <= ~16).
"""
function paulisum_to_matrix(ps::PauliSum)
    # infer N from one key (PauliBasis{N}) - we assume Dict key type encodes N
    anykey = first(keys(ps))
    # extract the param N from the key type name:
    # we assume `anykey` is PauliBasis{N} with fields z,x.
    # Use the pauli_matrix helper you already have that accepts PauliBasis{N}.
    M = nothing
    for (pb, coeff) in ps
        Mterm = Matrix(pb) * (coeff)    # pauli_matrix(pb) returns basis matrix
        if M === nothing
            M = copy(Mterm)
        else
            M .+= Mterm
        end
    end
    return M === nothing ? zeros(ComplexF64, 0, 0) : M
end


# --- Get particle-number eigenvalues (small systems only) -------------------
"""
    number_spectrum(Nop::PauliSum)

Return sorted unique eigenvalues of the number operator as read from its dense matrix.
Works only for small N_total (due to dense diagonalization).
"""
function number_spectrum(Nop::PauliSum)
    M = paulisum_to_matrix(Nop)
    eigs = eigen(Hermitian(M)).values
    # eigenvalues of the number operator should be integers (0..N_total) up to numerical error
    vals = unique(round.(Int, round.(eigs; digits=8)))  # round to nearest integer
    sort(vals)
end

# -------------------------
# Particle-number utilities
# -------------------------

# Count bits (occupation) of a UInt128 index
@inline function popcount_u128(x::UInt128)
    return count_ones(x)
end

"""
    enumerate_basis_by_particle(N_total; max_dim_check=22)

Return a Vector `groups` of length `N_total+1` where `groups[n+1]` is a Vector{UInt128}
of computational-basis indices (bitstrings) that have particle-number `n`.

WARNING: this enumerates all 2^N_total basis states. For N_total > ~22 this becomes large.
`max_dim_check` stops you from accidentally enumerating huge spaces (default 22).
"""
function enumerate_basis_by_particle(N_total::Int; max_dim_check::Int=22)
    if N_total < 0
        throw(ArgumentError("N_total must be nonnegative"))
    end
    if N_total > max_dim_check
        throw(ArgumentError("Refuse to enumerate: 2^$N_total > 2^$max_dim_check (set max_dim_check explicitly if you know what you are doing)"))
    end

    dim = UInt128(1) << N_total
    groups = [Vector{UInt128}() for _ in 0:N_total]

    for idx::UInt128 in UInt128(0):(dim - UInt128(1))
        n = popcount_u128(idx)
        push!(groups[n+1], idx)
    end
    return groups
end

"""
    bitstring_repr(idx::UInt128, N_total::Int)

Return a human-readable bitstring like "01011" with leftmost = qubit 1.
"""
function bitstring_repr(idx::UInt128, N_total::Int)
    chars = Vector{Char}(undef, N_total)
    for q in 1:N_total
        mask = UInt128(1) << (q-1)
        chars[q] = ((idx & mask) != 0) ? '1' : '0'
    end
    return join(chars)#reverse(chars))   # reverse if you want qubit 1 on leftmost
end

"""
    projector_for_particle(n::Int, N_total::Int; max_dense=16)

Return the dense matrix `P_n` (2^N_total × 2^N_total) that projects onto states
with total particle number `n`. Only allowed when `N_total ≤ max_dense` (default 16)
to avoid huge allocations.
"""
function projector_for_particle(n::Int, N_total::Int; max_dense::Int=16)
    if n < 0 || n > N_total
        throw(ArgumentError("n must be between 0 and N_total"))
    end
    if N_total > max_dense
        throw(ArgumentError("Dense projector refused: N_total=$N_total > max_dense=$max_dense"))
    end
    dim = Int(1) << N_total
    P = zeros(ComplexF64, dim, dim)
    groups = enumerate_basis_by_particle(N_total; max_dim_check=max_dense)
    for idx in groups[n+1]
        j = Int(idx) + 1  # Julia 1-based index
        P[j,j] = 1.0 + 0im
    end
    return P
end

"""
    project_H_to_sector_dense(H::PauliSum, n::Int; max_dense=16)

Return the dense matrix P_n * H * P_n. Uses paulisum_to_matrix(H) internally.
Only allowed when N_total ≤ max_dense.
"""
function project_H_to_sector_dense(H::PauliSum, n::Int; max_dense::Int=16)
    # infer N_total from one PauliBasis key (assumes keys are PauliBasis{N})
    anykey = first(keys(H))
    # we assume pauli_matrix expects a PauliBasis{N} and will produce 2^N dim matrices.
    # infer N_total by computing the number of qubits from anykey's type param:
    # Fallback: attempt to compute N_total from the size of pauli_matrix(anykey)
    Mtemp = Matrix(anykey)
    dim = size(Mtemp, 1)
    N_total = round(Int, log2(dim))
    if N_total > max_dense
        throw(ArgumentError("Dense sector projection refused: N_total=$N_total > max_dense=$max_dense"))
    end

    M_H = paulisum_to_matrix(H)   # your helper; returns dense matrix
    P = projector_for_particle(n, N_total; max_dense=max_dense)
    return P * M_H * P
end

"""
    expectation_particle_number_from_state(psi::AbstractVector{<:Number}, N_total::Int)

Compute ⟨psi| N̂ |psi⟩ where N̂ = sum_k n_k, without building operators.
`psi` is assumed normalized and length = 2^N_total.
This computes the expectation by summing |psi_i|^2 * occupation(i).
"""
function expectation_particle_number_from_state(psi::AbstractVector{<:Number}, N_total::Int)
    dim = length(psi)
    if dim != (1 << N_total)
        throw(ArgumentError("psi length $(dim) does not match 2^N_total=$(1<<N_total)"))
    end
    ex = 0.0
    for j in 1:dim
        p = abs2(psi[j])
        idx = UInt128(j-1)               # basis index
        ex += p * popcount_u128(idx)
    end
    return ex
end

# --------------------------------------------------------------------------
# Analyze ground state properties
# --------------------------------------------------------------------------

# popcount for UInt128
@inline function popcount_u128(x::UInt128)
    return count_ones(x)
end

# string repr: leftmost = qubit 1 (same convention as earlier)
function bitstring_repr(idx::UInt128, N_total::Int)
    chars = Vector{Char}(undef, N_total)
    for q in 1:N_total
        mask = UInt128(1) << (q-1)
        chars[q] = ((idx & mask) != 0) ? '1' : '0'
    end
    return join(reverse(chars))
end

################################################################################
println("\n--- Analyzing Hubbard 2D Hamiltonian ---")
println("Lx=$(Lx) Ly=$(Ly) t=$(t) U=$(U)  N_total=$(N_total)")
println("Hamiltonian has ", length(H), " Pauli terms.")

# Test number operator functions
Nop = number_operator(N_total)

(commutes, normC) = commutes_with(H, Nop; tol=1e-10)
println("Commutes? ", commutes, "  ||[H,N]|| ≈ ", normC)

# For small systems, inspect the particle-number sectors:
spec = number_spectrum(Nop)
println("Particle-number eigenvalues present: ", spec)

groups = enumerate_basis_by_particle(N_total)   # groups[5] are all states with 4 electrons
println("Number of states with 4 electrons: ", length(groups[5]))

# pretty one basis element
idx = groups[5][1]
println("example basis (idx=$(idx)) -> ", bitstring_repr(idx, N_total))

# build dense projector for n=4 (only if N_total small, default max_dense=16)
P4 = projector_for_particle(4, N_total)

# project Hamiltonian into the 4-electron sector (dense)
H4 = project_H_to_sector_dense(H, 4)
println("Projected H to n=4 sector has size: ", size(H4))
evals4 = eigen(Hermitian(H4)).values
println("Eigenvalues in n=4 sector:")
for i in evals4
    @printf("%12.8f\n", real(i))
end

# --- main extraction given eigvecs and known N_total --------------------------
println("\nAnalyzing full Hamiltonian eigensystem...")
# Compute eigenvectors:
Hmat = Matrix(H)
eigvecs = eigen(Hermitian(Hmat))
println("Eigenvectors (columns):")
display(eigvecs.vectors)

vals = eigvecs.values
vecs = eigvecs.vectors   # columns are eigenvectors

# ground energy and indices (handle degeneracy with tolerance)
tol = 1e-10
Egs = minimum(vals)
gs_inds = findall(x -> isapprox(x, Egs; atol=tol, rtol=0), vals)

println("Ground energy = ", Egs, "  (degeneracy = ", length(gs_inds), ")")
println("Ground eigenvector index/indices: ", gs_inds)

dim = size(vecs, 1)
# precompute popcounts for each computational basis index 0..dim-1
popcounts = Array{Int}(undef, dim)
for j in 1:dim
    popcounts[j] = popcount_u128(UInt128(j-1))
end

# For each ground eigenvector, compute expectation, variance, and top kets
for (kcount, gi) in enumerate(gs_inds)
    psi = vecs[:, gi]                 # ground eigenvector (normalized)
    probs = abs2.(psi)                # probabilities in computational basis

    # expectation and variance of particle number
    n_expect = sum(probs .* popcounts)
    n2_expect = sum(probs .* (popcounts .^ 2))
    n_var = n2_expect - n_expect^2

    # dominant computational basis state
    top_j = argmax(probs)
    top_prob = probs[top_j]
    top_idx = UInt128(top_j - 1)
    top_bitstr = bitstring_repr(top_idx, N_total)

    println("\nGround vector #$(kcount) (column $gi):")
    println("  ⟨N̂⟩ = ", n_expect, "  Var(N̂) = ", n_var)
    if n_var < 1e-8
        println("  Ground state is (numerically) a number eigenstate with n = ", Int(round(n_expect)))
    else
        println("  Ground state is a superposition across particle sectors (nonzero variance).")
    end
    println("  Most-weighted basis ket: |", top_bitstr, "⟩  with probability ", top_prob)

    # show top few basis amplitudes (sorted)
    K = min(8, dim)  # how many to show
    idxs = partialsortperm(probs, rev=true, 1:K)  # indices of top K probabilities
    println("  Top basis kets (index -> bitstring : amplitude (abs) , probability):")
    for j in idxs
        bidx = UInt128(j-1)
        println("    $(j-1) -> |", bitstring_repr(bidx, N_total), "⟩ : ", abs(psi[j]), ", ", probs[j])
    end
end
