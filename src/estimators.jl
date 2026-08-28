using SparseArrays
using Base.Threads
using KrylovKit
using Random

"""
    subspace_sparse(O::XZPauliSum{T}, basis::Vector{Ket{N}}) where {N,T}

Matrix of `O` projected onto `span(basis)`, as a `SparseMatrixCSC`.

`subspace_matvec` rediscovers the subspace connectivity on every application:
each call costs `O(dim * n_x)` hash probes, and a Krylov solve spends 20-30
applications on the *same* connectivity. Building the block once and handing a
matrix to Krylov is ~20x faster at the FOIS sizes reached mid-flow (measured:
72s -> 3.8s at dim=10807 with a 506k-term H).

Storage is the limit rather than time -- the block is far from sparse (~40%
fill at dim 1e4), so past dim ~2e4 use a rank-truncated basis instead of a
threshold-selected one.
"""
function subspace_sparse(O::XZPauliSum{T}, basis::Vector{Ket{N}}) where {N,T}
    dim = length(basis)
    idx = Dict{Ket{N},Int}(k => i for (i, k) in enumerate(basis))

    # threadid() can exceed nthreads() when an interactive threadpool exists,
    # so size the per-thread buffers by maxthreadid()
    nt = Threads.maxthreadid()
    Is = [Int[] for _ in 1:nt]
    Js = [Int[] for _ in 1:nt]
    Vs = [T[]   for _ in 1:nt]

    xs = collect(O)
    @threads for i in 1:dim
        t  = threadid()
        ki = basis[i]
        for (x, zs) in xs
            kj = Ket{N}(ki.v ⊻ x)
            j = get(idx, kj, 0)
            j == 0 && continue
            val = zero(T)
            for (z, c) in zs
                ph, _ = PauliBasis{N}(z, x) * kj
                val += ph * c
            end
            iszero(val) && continue
            push!(Is[t], i); push!(Js[t], j); push!(Vs[t], val)
        end
    end
    return sparse(vcat(Is...), vcat(Js...), vcat(Vs...), dim, dim)
end


# ---------------------------------------------------------------------------
#  Matrix-free subspace operator
# ---------------------------------------------------------------------------

"""
    XTable

Open-addressing `Int128 -> Int32` table. `Dict{Ket{N},Int}` lookups are the hot
spot of every subspace kernel; this removes the generic hash and the Dict
indirection from the inner loop.
"""
struct XTable
    keys::Vector{Int128}
    vals::Vector{Int32}
    mask::UInt64
end

@inline function _xhash(x::Int128, mask::UInt64)
    u = reinterpret(UInt128, x)
    v = UInt64(u & typemax(UInt64)) ⊻ UInt64((u >> 64) & typemax(UInt64))
    ((v * 0x9E3779B97F4A7C15) >> 24) & mask
end

function XTable(xs::Vector{Int128})
    cap = 1
    while cap < 2*max(1, length(xs)); cap <<= 1; end
    t = XTable(fill(Int128(-1), cap), zeros(Int32, cap), UInt64(cap - 1))
    for (i, x) in enumerate(xs)
        sl = _xhash(x, t.mask)
        while t.keys[sl+1] != Int128(-1); sl = (sl + 1) & t.mask; end
        t.keys[sl+1] = x
        t.vals[sl+1] = Int32(i)
    end
    return t
end

@inline function Base.get(t::XTable, x::Int128)
    sl = _xhash(x, t.mask)
    @inbounds while true
        k = t.keys[sl+1]
        k == x && return t.vals[sl+1]
        k == Int128(-1) && return Int32(0)
        sl = (sl + 1) & t.mask
    end
end

# ⟨i|P|j⟩ for the x-group `xc`, where `target` is the ket P|j⟩ lands on. This is
# `PauliBasis(z,xc) * ket` with the PauliBasis construction, the generic multiply
# and the phase-table indirection collapsed to two popcounts.
@inline function _phase_sum(zs::Vector{Tuple{Int128,T}}, xc::Int128, target::Int128) where T
    val = zero(ComplexF64)
    @inbounds for (z, c) in zs
        # symplectic_phase(p) = (4 - count_ones(p.z & p.x) % 4) % 4 -- note the
        # NEGATIVE sign. Getting this wrong conjugates the block, which is
        # invisible whenever the subspace matrix happens to be real (JW
        # molecular and Heisenberg cases both are).
        ph = (4 - (count_ones(z & xc) & 3) + 2*count_ones(z & target)) & 3
        val += ph == 0 ? ComplexF64(c) :
               ph == 1 ? im*ComplexF64(c) :
               ph == 2 ? -ComplexF64(c) : -im*ComplexF64(c)
    end
    return val
end

"""
    SubspaceOp(O::XZPauliSum, basis, ψ; memory_budget, mode)

The subspace block of `O` as an applicable operator, in one of two
representations chosen automatically from a projected memory cost:

* `:sparse` — the explicit `SparseMatrixCSC`. Fastest per application once
  built, but O(nnz) memory, which is unbounded as the flow grows `H`.
* `:free` — matrix-free, O(dim) memory. Two structural facts make this ~24x
  faster than the `subspace_matvec` it replaces (measured, bit-identical
  results):

  1. The FOIS basis is *indexed by* `H`'s x-strings: element `a` is
     `ψ ⊻ x_a`, so `⟨i_a|H|i_b⟩ ≠ 0` iff `x_a ⊻ x_b ∈ X`. The scan is therefore
     `dim²`, not `dim × n_x` — and after clipping dim < n_x always, with the
     gap *widening* as the flow grows `H` (60k vs 242k at iteration 0 of HBC;
     181k vs 800k by iteration 10).
  2. The Pauli with x-string `x_a ⊻ x_b` acting on `i_b` lands on `ψ ⊻ x_a`,
     i.e. the row's own ket — constant across the inner loop, so the phase
     hoists out of it.

  The block is Hermitian, so only the upper triangle is traversed.
"""
struct SubspaceOp{T}
    mode::Symbol
    M::SparseMatrixCSC{ComplexF64,Int}
    xb::Vector{Int128}
    tab::XTable
    groups::Vector{Vector{Tuple{Int128,T}}}
    psiv::Int128
    dim::Int
    nnz_est::Int
end

# Sampled row density; the pair scan makes this cheap (nsample x dim).
function _estimate_nnz(xb::Vector{Int128}, tab::XTable, dim::Int; nsample=256)
    dim == 0 && return 0
    step = max(1, dim ÷ min(nsample, dim))
    hits = 0; rows = 0
    @inbounds for a in 1:step:dim
        rows += 1
        xa = xb[a]
        for b in 1:dim
            get(tab, xa ⊻ xb[b]) != 0 && (hits += 1)
        end
    end
    return round(Int, hits * (dim / rows))
end

function SubspaceOp(O::XZPauliSum{T}, basis::Vector{Ket{N}}, ψ::Ket{N};
                    memory_budget = 8 * 2^30,
                    mode = nothing,
                    verbose = 0) where {N,T}
    Xs     = collect(keys(O))
    tab    = XTable(Xs)
    groups = Vector{Vector{Tuple{Int128,T}}}([O[x] for x in Xs])
    xb     = Int128[k.v ⊻ ψ.v for k in basis]
    dim    = length(basis)

    nnz_est = _estimate_nnz(xb, tab, dim)
    # ~88 B/nnz: CSC (24) plus the COO staging the builder passes through (32x2)
    bytes = nnz_est * 88
    m = mode !== nothing ? mode : (bytes <= memory_budget ? :sparse : :free)
    verbose < 1 || @printf(" SubspaceOp: dim=%d  nnz~%d  projected %.2f GiB  -> :%s\n",
                           dim, nnz_est, bytes/2^30, m)

    M = m === :sparse ? SparseMatrixCSC{ComplexF64,Int}(subspace_sparse(O, basis)) :
                        spzeros(ComplexF64, 0, 0)
    return SubspaceOp{T}(m, M, xb, tab, groups, ψ.v, dim, nnz_est)
end

function (op::SubspaceOp{T})(v::AbstractVector) where T
    op.mode === :sparse && return op.M * v
    dim = op.dim
    nch = max(1, Threads.nthreads())
    # explicit chunking rather than threadid() indexing: under :dynamic
    # scheduling a task can migrate threads, which would corrupt the buffers
    bufs = [zeros(ComplexF64, dim) for _ in 1:nch]
    xb, tab, groups, psiv = op.xb, op.tab, op.groups, op.psiv
    gdiag = get(tab, Int128(0))
    @threads for c in 1:nch
        buf = bufs[c]
        @inbounds for a in c:nch:dim          # interleaved: row cost falls with a
            xa = xb[a]; ia = psiv ⊻ xa; va = v[a]
            if gdiag != 0
                buf[a] += _phase_sum(groups[gdiag], Int128(0), ia) * va
            end
            for b in (a+1):dim
                xc = xa ⊻ xb[b]
                g = get(tab, xc); g == 0 && continue
                val = _phase_sum(groups[g], xc, ia)
                buf[a] += val * v[b]
                buf[b] += conj(val) * va      # Hermitian: upper triangle only
            end
        end
    end
    s = bufs[1]
    for t in 2:nch; s .+= bufs[t]; end
    return s
end

"""
    qblock(op, e0)

`x -> (e₀ - H_QQ)x` over `Q = basis[2:end]`, without materialising the block.
Valid in either representation: the reference sits at index 1, so embedding with
a zero there and restricting the result is exactly the `Q` restriction.
"""
function qblock(op::SubspaceOp, e0::Real)
    return function (x::AbstractVector)
        full = zeros(ComplexF64, op.dim)
        @views full[2:end] .= x
        y = op(full)
        return e0 .* x .- @view(y[2:end])
    end
end

"""
    fois_space(O::XZPauliSum{T}, ψ::Ket{N}; thresh=1e-4) where {N,T}

`H|ψ⟩` clipped at `thresh`, as `(v, basis)` with `basis[1] === ψ` and the
first-order interacting space in `basis[2:end]`.
"""
function fois_space(O::XZPauliSum{T}, ψ::Ket{N}; thresh=1e-4, max_dim=0) where {N,T}
    v = matvec(O, ψ)
    thresh > 0 && coeff_clip!(v, thresh)
    haskey(v, ψ) || (v[ψ] = zero(valtype(v)))
    basis = Vector{Ket{N}}([ψ])
    for k in keys(v)
        k == ψ || push!(basis, k)
    end
    # A fixed `thresh` does not fix the cost: the FOIS grows with len(H), so an
    # estimate that is cheap at iteration 0 can be unaffordable 10 iterations
    # later (HBC: dim 60k -> 181k, build work 1.4e10 -> 1.5e11). Capping the
    # dimension instead holds the cost flat along the flow -- keep the largest
    # |⟨q|H|ψ⟩|, which is what a tighter threshold would have selected anyway.
    if max_dim > 0 && length(basis) > max_dim + 1
        q = @view basis[2:end]
        keep = partialsortperm([abs(v[k]) for k in q], 1:max_dim, rev=true)
        basis = vcat([ψ], Vector{Ket{N}}(q[keep]))
    end
    return v, basis
end

"""
    subspace_estimates(H, ψ; thresh=1e-4, ...)

CEPA and CMX(2) ground-state estimates over the reference `ψ`, sharing a single
subspace build.

Both are **size extensive**, which is why these two are the pair worth tracking
along a flow (verified on non-interacting `osum` fragments: additive to machine
precision, while FOIS-CI and PDS drift by 1e-2 at three fragments).

* `CEPA`   `E = e₀ + b†(e₀ - H_QQ)⁻¹b` over `Q = FOIS \\ {ψ}`, the one-shot
  Löwdin partitioning. Extensive because on `H = H_A ⊕ H_B` the `Q` block is
  block diagonal and the `E_B` shift it picks up cancels against `e₀ = E_A+E_B`.
  Iterating `E` to self-consistency would converge to the FOIS-CI eigenvalue and
  destroy exactly that cancellation.
* `CMX(2)` `E = e₀ - I₂²/I₃` from the 2nd and 3rd connected moments. Extensive
  by construction (cumulants are additive). Essentially free here since it
  reuses CEPA's subspace matrix.

Well-posedness: CEPA needs `e₀ < λ_min(H_QQ)` and CMX(2) needs `I₃ > 0`. Both
hold comfortably along a flow and improve with it, but both fail at iteration 0
on a symmetric unflowed `H`, where the reference is exactly degenerate with the
first-order space. `check=true` tests each and warns.

`thresh` clips the FOIS. It also truncates the moments, so `I₂` drifts from
`variance(H,ψ)` as `thresh` grows; use `thresh=0` for exact moments.

Returns a NamedTuple: `e0, cepa, cmx2, I2, I3, lambda_min, dim, x, basis`.
"""
function subspace_estimates(Hin::AnyPauliSum{N,T}, ψ::Ket{N};
                            thresh      = 1e-4,
                            max_dim     = 0,
                            basis       = nothing,
                            x0          = nothing,
                            tol         = 1e-8,
                            maxiter     = 200,
                            direct_max  = 2000,
                            check       = true,
                            memory_budget = 8 * 2^30,
                            op_mode     = nothing,
                            verbose     = 1) where {N,T}

    O  = Hin isa SparsePauliVector ? PauliSum(Hin) : Hin
    xz = pack_x_z(O)

    # An explicit `basis` pins the subspace. That matters for differencing the
    # estimator across a truncation, est(A) - est(H): if each side selects its
    # own FOIS the two systematic errors sit on different subspaces and do not
    # cancel, so a ~1e-9 Ha signal is swamped by ~1e-3 Ha of basis noise.
    if basis === nothing
        v, basis = fois_space(xz, ψ, thresh=thresh, max_dim=max_dim)
    else
        v = matvec(xz, ψ)
    end
    e0  = real(expectation_value(xz, ψ))
    dim = length(basis)

    # One operator, shared by both estimators. Representation is chosen from a
    # projected memory cost, so a flow that grows H past what an explicit block
    # can hold degrades to the matrix-free kernel instead of being OOM-killed.
    op = SubspaceOp(xz, basis, ψ; memory_budget=memory_budget, mode=op_mode,
                    verbose=verbose)

    # --- CMX(2) ---------------------------------------------------------
    # w = (H - e₀)|ψ⟩. The 2nd and 3rd cumulants ARE the 2nd and 3rd central
    # moments, so I₂ = ⟨w|w⟩ and I₃ = ⟨w|(H-e₀)|w⟩ with no cumulant recursion
    # and no cancellation against powers of e₀ (raw moments at |e₀| ~ 100 lose
    # every digit by the 3rd order).
    w  = ComplexF64[get(v, k, zero(T)) for k in basis]
    w[1] -= e0
    I2 = real(dot(w, w))
    I3 = real(dot(w, op(w))) - e0 * I2

    ecmx2 = e0
    if I3 > 0
        ecmx2 = e0 - I2^2 / I3
    elseif check
        @warn "CMX(2) ill-posed: I₃ ≤ 0, no correction applied" I2 I3 e0
    end

    # --- CEPA -----------------------------------------------------------
    nq  = dim - 1
    x   = ComplexF64[]
    λmin = NaN
    ecepa = e0

    if nq > 0
        b = w[2:dim]                          # ⟨q|H|ψ⟩ for q ≠ ψ
        Aop = qblock(op, e0)                  # x -> (e₀ - H_QQ)x
        Hqq = y -> begin
            full = zeros(ComplexF64, dim); @views full[2:end] .= y
            op(full)[2:end]
        end

        if check
            vals, _, _ = KrylovKit.eigsolve(Hqq, copy(b), 1, :SR;
                                            ishermitian = true, issymmetric = false,
                                            tol = 1e-6, maxiter = 100)
            λmin = real(vals[1])
            if e0 >= λmin - 1e-8
                @warn "CEPA ill-posed: e₀ ≥ λ_min(H_QQ), denominator is not negative definite" e0 λmin (e0 - λmin)
            end
        end

        xg = zeros(ComplexF64, nq)
        if x0 isa KetSum
            for (i, k) in enumerate(basis[2:dim]); xg[i] = get(x0, k, zero(ComplexF64)); end
        elseif x0 !== nothing && length(x0) == nq
            xg .= x0
        end

        if op.mode === :sparse && nq <= direct_max
            Ad = e0*Matrix{ComplexF64}(LinearAlgebra.I, nq, nq) - Matrix(op.M[2:dim, 2:dim])
            # near-degenerate reference and FOIS => pinv, matching the exact
            # least-squares reference form rather than throwing or returning noise
            x = (check && abs(e0 - λmin) < 1e-8) ? pinv(Ad) * b : Ad \ b
        else
            x, info = KrylovKit.linsolve(Aop, b, xg;
                                         ishermitian = true, issymmetric = false,
                                         tol = tol, maxiter = maxiter)
            info.converged == 0 && @warn "CEPA linsolve did not converge; energy is unreliable" info.normres tol maxiter
        end
        ecepa = e0 + real(dot(x, b))
    end

    verbose < 2 || @printf(" e0 = %14.8f  E(cepa) = %14.8f  E(cmx2) = %14.8f  dim = %8i\n",
                           e0, ecepa, ecmx2, dim)

    return (e0=e0, cepa=ecepa, cmx2=ecmx2, I2=I2, I3=I3,
            lambda_min=λmin, dim=dim, x=x, basis=basis,
            mode=op.mode, nnz_est=op.nnz_est)
end

"""
    cepa(H, ref::Ket{N}; thresh=1e-4, verbose=4, x0=nothing, tol=1e-6) where N

CEPA ground-state estimate: `E = e₀ + b†(e₀ - H_QQ)⁻¹b` over the first-order
interacting space. See [`subspace_estimates`](@ref) for the method, its
extensivity, and its well-posedness conditions.

Returns `(e0, e, x, basis)` with `basis` the `Q` space (reference excluded).
"""
function cepa(H::AnyPauliSum{N,T}, ref::Ket{N}; thresh=1e-4, verbose=4, x0=nothing, tol=1e-6) where {N,T}
    r = subspace_estimates(H, ref; thresh=thresh, x0=x0, tol=tol, verbose=0)
    verbose < 1 || @printf(" E0 = %12.8f E(cepa) = %12.8f dim: %8i\n", r.e0, r.cepa, r.dim - 1)
    return r.e0, r.cepa, r.x, r.basis[2:end]
end

"""
    cmx2(H, ψ::Ket{N}; thresh=0) where N

CMX(2) ground-state estimate `E = e₀ - I₂²/I₃` from the 2nd and 3rd connected
moments about `ψ`. Size extensive. Returns `(e0, e, I2, I3)`.

`thresh=0` (the default here) keeps the moments exact; a nonzero value clips
`H|ψ⟩` and makes `I₂`, `I₃` approximate.
"""
function cmx2(H::AnyPauliSum{N,T}, ψ::Ket{N}; thresh=0, verbose=1) where {N,T}
    r = subspace_estimates(H, ψ; thresh=thresh, check=true, verbose=0)
    verbose < 1 || @printf(" E0 = %12.8f E(cmx2) = %12.8f I2 = %10.6f I3 = %10.6f\n",
                           r.e0, r.cmx2, r.I2, r.I3)
    return r.e0, r.cmx2, r.I2, r.I3
end

"""
    best_reference(H::AnyPauliSum{N}; kwargs...) where N

Computational basis state minimizing `⟨k|H|k⟩`, to be used as the DBF reference.

Exhaustive below `exhaustive_max` qubits, otherwise a greedy bit-flip descent
from `restarts` starts. `preserve_weight=true` restricts moves to bit *swaps*,
which fixes the Hamming weight and so preserves particle number under JW
regardless of which bit value the mapping treats as occupied.

Reference quality is the single biggest lever on every post-flow estimator, and
`⟨k|H|k⟩` alone does not reveal it. Two traps worth checking for:

* An isotropic coupling plus a transverse field puts the exact ground state on
  an axis off `z`, so **every** computational basis state has overlap `2⁻ᴺ`.
* A `Z₂` spin-flip symmetry (`∏Xᵢ` commuting with `H`) makes the ground state a
  cat, capping the overlap at `0.5`. There the overlap is a *misleading* metric:
  a flow can drive `⟨ψ|H|ψ⟩` to 7e-4 of exact while the overlap sits at 0.494.

`variance(H, ψ)` is the metric that stays honest in both cases, so it is
reported here and is what to watch along a flow.
"""
function best_reference(H::AnyPauliSum{N,T};
                        preserve_weight = false,
                        start           = nothing,
                        restarts        = 8,
                        exhaustive_max  = 20,
                        seed            = 1,
                        verbose         = 1) where {N,T}

    Hd = diag(H)
    energy(k) = real(expectation_value(Hd, k))

    best = start === nothing ? Ket{N}(Int128(0)) : start
    if N <= exhaustive_max && !preserve_weight
        ebest = Inf
        for i in 0:(Int128(2)^N - 1)
            k = Ket{N}(i); e = energy(k)
            e < ebest && (ebest = e; best = k)
        end
    else
        rng = MersenneTwister(seed)
        ebest = energy(best)
        w0 = count_ones(best.v)
        for r in 1:restarts
            k = r == 1 ? best : Ket{N}(rand(rng, Int128(0):(Int128(2)^N - 1)))
            if preserve_weight && r > 1
                # random state of the same Hamming weight as the start
                bits = randperm(rng, N)[1:w0]
                k = Ket{N}(reduce(|, (Int128(1) << (b-1) for b in bits), init=Int128(0)))
            end
            e = energy(k)
            improved = true
            while improved
                improved = false
                moves = preserve_weight ?
                    [(i,j) for i in 1:N, j in 1:N if i != j] : [(i,0) for i in 1:N]
                for (i,j) in moves
                    m = Int128(1) << (i-1)
                    j > 0 && (m |= Int128(1) << (j-1))
                    kn = Ket{N}(k.v ⊻ m)
                    # a swap only changes the state if the two bits differ
                    preserve_weight && count_ones(kn.v) != count_ones(k.v) && continue
                    en = energy(kn)
                    if en < e - 1e-12
                        k = kn; e = en; improved = true
                    end
                end
            end
            e < ebest && (ebest = e; best = k)
        end
    end

    if verbose >= 1
        @printf(" best_reference: ⟨ψ|H|ψ⟩ = %14.8f   variance = %12.6f   ψ = %s\n",
                ebest, real(variance(H, best)), string(best))
    end
    return best
end


"""
    cmx_moments(O, ψ::Ket{N}) where N

Exact connected moments `(I₁, I₂, I₃)` of `O` about `ψ`, with no clipping.

`I₂` and `I₃` are the 2nd and 3rd central moments (which *are* the 2nd and 3rd
cumulants), computed on the full first-order interacting space. The unclipped
basis is exactly the set of `O`'s x-strings, so `dim == n_x` and the two
`SubspaceOp` traversals cost the same; `:free` is used to avoid materialising a
matrix for what is a single matvec.
"""
function cmx_moments(Oin::AnyPauliSum{N,T}, ψ::Ket{N}) where {N,T}
    O  = Oin isa SparsePauliVector ? PauliSum(Oin) : Oin
    xz = pack_x_z(O)
    v, basis = fois_space(xz, ψ, thresh=0)
    e0 = real(expectation_value(xz, ψ))
    w  = ComplexF64[get(v, k, zero(T)) for k in basis]
    w[1] -= e0
    I2 = real(dot(w, w))
    I3 = subspace_quadform(xz, basis, ψ, w) - e0 * I2
    return (e0, I2, I3)
end

"""
    cmx_energy(I1, I2, I3)

`E = I₁ - I₂²/I₃`, falling back to `I₁` when `I₃ ≤ 0` (where CMX(2) is not a
lowering and the ratio is meaningless).
"""
cmx_energy(I1, I2, I3) = I3 > 0 ? I1 - I2^2/I3 : I1


"""
    subspace_quadform(O::XZPauliSum{T}, basis, ψ, w) where T

`⟨w|O|w⟩` for `w` on the FOIS basis, with the pair scan pruned by support
overlap. Exact — it skips only pairs that provably cannot contribute.

`⟨i_a|O|i_b⟩ ≠ 0` requires `x_a ⊻ x_b ∈ X`, hence
`weight(x_a ⊻ x_b) = w_a + w_b - 2·overlap ≤ w_max`, i.e.

    overlap ≥ omin(w_a, w_b) = ⌈(w_a + w_b - w_max)/2⌉

When `omin ≥ 2`, an inverted index from support *pairs* to basis indices
enumerates every candidate; when `omin ≤ 1` the weight gives no usable
constraint and that weight class is scanned in full. For a 2-body molecular H
all x-strings have weight ≤ 4, so the dominant w4×w4 block needs overlap ≥ 2 and
the index covers essentially everything (HBC: ~2.5k candidates per row instead
of 242k). Once the flow generates weight-6/8 strings the w4×w4 block becomes
unconstrained and the gain shrinks accordingly.

This is what CMX needs: `I₃ = ⟨w|O|w⟩ - e₀·I₂` is a single quadratic form.
"""
function subspace_quadform(O::XZPauliSum{T}, basis::Vector{Ket{N}}, ψ::Ket{N},
                           w::Vector{ComplexF64}) where {N,T}
    dim = length(basis)
    dim == 0 && return 0.0
    xs     = collect(keys(O))
    tab    = XTable(xs)
    groups = Vector{Vector{Tuple{Int128,T}}}([O[x] for x in xs])
    xb  = Int128[k.v ⊻ ψ.v for k in basis]
    wt  = Int[count_ones(x) for x in xb]
    wmax = isempty(wt) ? 0 : maximum(wt)

    # support positions of each basis x-string, and the pair buckets
    pos = [Int[i for i in 1:N if (x >> (i-1)) & 1 == 1] for x in xb]
    buckets = [Int32[] for _ in 1:N*N]
    @inbounds for a in 1:dim
        p = pos[a]
        for i in 1:length(p), j in i+1:length(p)
            push!(buckets[(p[i]-1)*N + p[j]], Int32(a))
        end
    end
    classes = sort(unique(wt))
    bycls   = Dict(c => Int32[a for a in 1:dim if wt[a] == c] for c in classes)

    gdiag = get(tab, Int128(0))
    nch = max(1, Threads.nthreads())
    partial = zeros(Float64, nch)
    @threads for ch in 1:nch
        stamp = zeros(Int32, dim); gen = Int32(0); cand = Int32[]
        acc = 0.0
        @inbounds for a in ch:nch:dim
            xa = xb[a]; ia = ψ.v ⊻ xa; wa = wt[a]; wA = w[a]
            if gdiag != 0
                acc += real(conj(wA) * _phase_sum(groups[gdiag], Int128(0), ia) * wA)
            end
            gen += Int32(1); empty!(cand)
            p = pos[a]
            for i in 1:length(p), j in i+1:length(p)
                for b in buckets[(p[i]-1)*N + p[j]]
                    if stamp[b] != gen; stamp[b] = gen; push!(cand, b); end
                end
            end
            for cls in classes                       # omin <= 1 => no usable constraint
                if wa + cls - wmax <= 2
                    for b in bycls[cls]
                        if stamp[b] != gen; stamp[b] = gen; push!(cand, b); end
                    end
                end
            end
            for b in cand
                b > a || continue                    # Hermitian: upper triangle, doubled
                xc = xa ⊻ xb[b]
                g = get(tab, xc); g == 0 && continue
                acc += 2*real(conj(wA) * _phase_sum(groups[g], xc, ia) * w[b])
            end
        end
        partial[ch] = acc
    end
    return sum(partial)
end
