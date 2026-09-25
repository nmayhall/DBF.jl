# =============================================================================
#  Build the 1D Hubbard Hamiltonian in the UHF molecular-orbital basis, with the
#  HF determinant mapped to |0…0⟩ — adopting the doped-Hubbard LNO pipeline
#  (data-vdbf1/hubbard_doped/lno_integrals_to_pauli_U4.jl): MO integrals → JW
#  PauliSum → reference-determinant PH transform (conjugate by X on occupied
#  qubits). Then ⟨0|H|0⟩ = E_HF and ⟨HF|GS⟩ → 1 as U → 0.
#
#  Spin ordering: alpha[1:L], beta[L+1:2L]. Half-filling: na=nb=L/2.
# =============================================================================

using PauliOperators
using LinearAlgebra
using Printf

# ---- site-basis 1D Hubbard (OBC, t=1) one-body per spin; two-body is U on-site (ab only)
function hubbard_site_onebody(L, t, μ)
    h = zeros(L, L)
    for p in 1:L-1; h[p, p+1] = -t; h[p+1, p] = -t; end
    for p in 1:L;   h[p, p] -= μ; end
    return h
end

# ---- UHF (AFM broken symmetry), self-consistent
function uhf(h, U, na, nb; iters=2000, mix=0.3, tol=1e-12)
    L = size(h, 1)
    da = Float64[isodd(i) ? 1.0 : 0.0 for i in 1:L]
    db = Float64[iseven(i) ? 1.0 : 0.0 for i in 1:L]
    Ca = Matrix{Float64}(I, L, L); Cb = copy(Ca)
    for it in 1:iters
        Fa = h + U * Diagonal(db)
        Fb = h + U * Diagonal(da)
        _, Ca = eigen(Symmetric(Fa))
        _, Cb = eigen(Symmetric(Fb))
        da_new = Float64[sum(Ca[i, p]^2 for p in 1:na) for i in 1:L]
        db_new = Float64[sum(Cb[i, p]^2 for p in 1:nb) for i in 1:L]
        Δ = maximum(abs.(da_new .- da)) + maximum(abs.(db_new .- db))
        da = mix .* da_new .+ (1 - mix) .* da
        db = mix .* db_new .+ (1 - mix) .* db
        Δ < tol && break
    end
    return Ca, Cb
end

# ---- transform integrals to the UHF-MO basis
function mo_integrals(h, U, Ca, Cb)
    L = size(h, 1)
    h1a = Ca' * h * Ca
    h1b = Cb' * h * Cb
    eri_ab = zeros(L, L, L, L)               # U Σ_i Ca[i,p]Ca[i,q]Cb[i,r]Cb[i,s]
    for p in 1:L, q in 1:L, r in 1:L, s in 1:L
        eri_ab[p, q, r, s] = U * sum(Ca[i, p] * Ca[i, q] * Cb[i, r] * Cb[i, s] for i in 1:L)
    end
    return h1a, h1b, eri_ab                   # eri_aa = eri_bb = 0 for Hubbard
end

# ---- JW helpers (from lno_integrals_to_pauli_U4.jl) ----
function _fermion_op(n, i; dagger=true)
    bit = i - 1; act = Int128(1) << bit; zstr = act - Int128(1)
    xt = PauliBasis(Pauli{n}(1, zstr, act))
    yt = PauliBasis(Pauli{n}(1, zstr | act, act))
    return dagger ? 0.5*(xt - im*yt) : 0.5*(xt + im*yt)
end
function _add_scaled!(dest, s, src)
    for (p, c) in src; dest[p] = get(dest, p, 0.0im) + s*c; end
    return dest
end

# Build H_transformed in the UHF-MO basis with HF determinant → |0…0⟩.
function build_hf_hamiltonian(L, U; t=1.0, tol=1e-12)
    μ = U/2; na = L ÷ 2; nb = L ÷ 2; n = 2L
    h = hubbard_site_onebody(L, t, μ)
    Ca, Cb = uhf(h, U, na, nb)
    h1a, h1b, eri_ab = mo_integrals(h, U, Ca, Cb)

    cre = [_fermion_op(n, i; dagger=true)  for i in 1:n]
    ann = [_fermion_op(n, i; dagger=false) for i in 1:n]
    H = PauliSum(n, ComplexF64)
    for q in 1:L, p in 1:L                       # alpha one-body
        abs(h1a[p,q]) > tol && _add_scaled!(H, h1a[p,q], cre[p]*ann[q])
    end
    for q in 1:L, p in 1:L                       # beta one-body (offset L)
        abs(h1b[p,q]) > tol && _add_scaled!(H, h1b[p,q], cre[p+L]*ann[q+L])
    end
    for s in 1:L, r in 1:L, q in 1:L, p in 1:L   # alpha-beta two-body
        v = eri_ab[p,q,r,s]
        abs(v) > tol || continue
        _add_scaled!(H, v, cre[p]*cre[r+L]*ann[s+L]*ann[q])
    end
    coeff_clip!(H, tol)
    for pk in keys(H); H[pk] = complex(real(H[pk]), 0.0); end   # Hermitian → real

    # reference-determinant PH transform: occupied = 1:na alpha, L+1:L+nb beta
    occ = vcat(1:na, L .+ (1:nb))
    refX = PauliBasis(Pauli(n, X=collect(occ)))
    for (p, c) in H
        PauliOperators.commute(refX, p) || (H[p] = -c)
    end
    return H, na, nb, n
end
