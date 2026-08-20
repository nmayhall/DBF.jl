using Test
using Random
using Printf
using PauliOperators
using LinearAlgebra
using SparseArrays
using DBF

full_basis(N) = [Ket{N}(Int128(i)) for i in 0:2^N-1]

@testset "subspace_sparse" begin
    Random.seed!(3)
    N = 8
    H = rand(PauliSum{N}, n_paulis=200); H = H + H'
    xz = pack_x_z(H)

    # matches the existing dense subspace builder, on a full and a partial basis
    fb = full_basis(N)
    @test norm(Matrix(subspace_sparse(xz, fb)) - Matrix(xz, fb)) < 1e-12
    sub = fb[1:2:end]
    @test norm(Matrix(subspace_sparse(xz, sub)) - Matrix(xz, sub)) < 1e-12

    # and reproduces the matrix-free subspace_matvec it replaces
    v = KetSum(sub, T=ComplexF64)
    for (i,k) in enumerate(sub); v[k] = i/length(sub); end
    ref = Vector(DBF.subspace_matvec(xz, v), sub)
    @test norm(subspace_sparse(xz, sub) * Vector(v, sub) - ref) < 1e-10
end

@testset "matrix-free subspace operator" begin
    Random.seed!(11)
    N = 8
    H = rand(PauliSum{N}, n_paulis=300); H = H + H'
    ψ = Ket{N}(Int128(0))
    xz = pack_x_z(H)
    v, basis = DBF.fois_space(xz, ψ, thresh=0)
    dim = length(basis)

    opS = DBF.SubspaceOp(xz, basis, ψ; mode=:sparse)
    opF = DBF.SubspaceOp(xz, basis, ψ; mode=:free)
    x = ComplexF64[cis(0.7i)/i for i in 1:dim]

    # the matrix-free kernel is structurally identical, not an approximation:
    # same pairs, same phases, only the traversal order differs
    @test norm(opF(x) - opS(x)) < 1e-10
    @test norm(opS(x) - subspace_sparse(xz, basis)*x) < 1e-10

    # Hermitian symmetry (only the upper triangle is traversed)
    y = ComplexF64[cis(1.3i)*i for i in 1:dim]
    @test abs(dot(y, opF(x)) - dot(opF(y), x)) < 1e-10

    # both representations give identical estimates. direct_max=0 forces both
    # through linsolve: the sparse path would otherwise take the dense direct
    # solve, and on this random H the CEPA denominator is indefinite, so the two
    # solvers legitimately disagree there and the comparison would test nothing.
    a = subspace_estimates(H, ψ, thresh=0, check=false, verbose=0, op_mode=:sparse, direct_max=0)
    b = subspace_estimates(H, ψ, thresh=0, check=false, verbose=0, op_mode=:free,   direct_max=0)
    @test a.mode == :sparse && b.mode == :free
    @test abs(a.cepa - b.cepa) < 1e-8
    @test abs(a.cmx2 - b.cmx2) < 1e-10
    @test abs(a.I3   - b.I3)   < 1e-10

    # a zero memory budget must fall back rather than attempt the build
    c = subspace_estimates(H, ψ, thresh=0, check=false, verbose=0, memory_budget=0, direct_max=0)
    @test c.mode == :free
    @test abs(c.cepa - a.cepa) < 1e-8

    # and on a well-posed physical H the two modes agree through the DEFAULT
    # solver paths (sparse direct vs matrix-free Krylov), which is how they will
    # actually be used
    Hp = DBF.heisenberg_1D(8, 0.8, 0.8, 1.0, x=0.4, z=0.5)
    ψp = best_reference(Hp, verbose=0)
    ap = subspace_estimates(Hp, ψp, thresh=0, verbose=0, op_mode=:sparse)
    bp = subspace_estimates(Hp, ψp, thresh=0, verbose=0, op_mode=:free)
    @test abs(ap.cepa - bp.cepa) < 1e-8
    @test abs(ap.cmx2 - bp.cmx2) < 1e-10

    # sampled nnz estimate should be in the right ballpark of the truth
    @test opS.nnz_est > 0
    @test 0.5 < opS.nnz_est / nnz(subspace_sparse(xz, basis)) < 2.0
end

@testset "fois max_dim cap" begin
    Random.seed!(13)
    N = 8
    H = rand(PauliSum{N}, n_paulis=300); H = H + H'
    ψ = Ket{N}(Int128(0)); xz = pack_x_z(H)
    _, full = DBF.fois_space(xz, ψ, thresh=0)
    _, capped = DBF.fois_space(xz, ψ, thresh=0, max_dim=25)
    @test length(full) > 26
    @test length(capped) == 26                       # reference + 25
    @test capped[1] == ψ
    @test Set(capped) ⊆ Set(full)
    # the cap must keep the LARGEST couplings, not an arbitrary subset
    v = DBF.matvec(xz, ψ)
    kept = minimum(abs(v[k]) for k in capped[2:end])
    dropped = setdiff(Set(full[2:end]), Set(capped[2:end]))
    @test all(abs(v[k]) <= kept + 1e-12 for k in dropped)

    r = subspace_estimates(H, ψ, thresh=0, max_dim=25, check=false, verbose=0)
    @test r.dim == 26
end

@testset "cepa / cmx2 values" begin
    Random.seed!(5)
    N = 8
    H = rand(PauliSum{N}, n_paulis=300); H = H + H'
    ψ = Ket{N}(Int128(0))
    xz = pack_x_z(H)

    r = subspace_estimates(H, ψ, thresh=0, check=false, verbose=0)

    # exact reference from the dense Hamiltonian
    fb = full_basis(N)
    Hm = Matrix(subspace_sparse(xz, fb))
    e = zeros(ComplexF64, 2^N); e[1] = 1                 # fb[1] === Ket{N}(0)
    e0 = real(e' * (Hm * e))
    w  = (Hm - e0*I) * e
    I2 = real(w' * w)
    I3 = real(w' * ((Hm - e0*I) * w))

    @test abs(r.e0 - e0) < 1e-12
    @test abs(r.I2 - I2) < 1e-10
    @test abs(r.I3 - I3) < 1e-10

    # I2 is the variance: the sharpest check that the moment path is right
    @test abs(r.I2 - real(variance(H, ψ))) < 1e-10

    # CMX(2) is only a lowering when I3 > 0. A random PauliSum need not satisfy
    # that (the reference can sit above the centroid of its interacting space),
    # so assert whichever branch this H actually lands in.
    if I3 > 0
        @test abs(r.cmx2 - (e0 - I2^2/I3)) < 1e-10
    else
        @test r.cmx2 == e0                       # guard refuses the correction
    end

    # CEPA against the explicit b'(e0 - H_QQ)^-1 b form
    v, basis = DBF.fois_space(xz, ψ, thresh=0)
    Q = basis[2:end]
    b = ComplexF64[v[k] for k in Q]
    A = e0*Matrix(I, length(Q), length(Q)) - Matrix(subspace_sparse(xz, Q))
    @test abs((r.cepa - e0) - real(b' * (pinv(A) * b))) < 1e-8

    # public wrappers agree with the shared-build path
    _, ec, _, _ = cepa(H, ψ, thresh=0, verbose=0)
    _, ex, _, _ = cmx2(H, ψ, verbose=0)
    @test abs(ec - r.cepa) < 1e-10
    @test abs(ex - r.cmx2) < 1e-10
end

@testset "cmx2 on a physical H" begin
    N = 8
    H = DBF.heisenberg_1D(N, 0.8, 0.8, 1.0, x=0.4, z=0.5)
    ψ = best_reference(H, verbose=0)
    r = subspace_estimates(H, ψ, thresh=0, check=false, verbose=0)

    fb = full_basis(N)
    Hm = Matrix(subspace_sparse(pack_x_z(H), fb))
    e = zeros(ComplexF64, 2^N); e[ψ.v + 1] = 1
    e0 = real(e' * (Hm * e))
    w  = (Hm - e0*I) * e
    I2 = real(w' * w); I3 = real(w' * ((Hm - e0*I) * w))

    @test I3 > 0                                  # well posed for a physical H
    @test abs(r.cmx2 - (e0 - I2^2/I3)) < 1e-10
    @test r.cmx2 < r.e0                           # and is a lowering

    # CMX(2) is not variational, so it may sit below E_FCI -- the property that
    # matters is that it is closer to it than the bare expectation value
    efci = minimum(real(eigvals(Hermitian(Hm))))
    @test abs(r.cmx2 - efci) < abs(r.e0 - efci)

    # Guards warn rather than returning a silent divergence.
    # CEPA needs e0 < lambda_min(H_QQ); the worst reference in the spectrum
    # violates it, since its first-order space reaches below it.
    Hd = diag(H)
    ψbad = Ket{N}(Int128(argmax([real(expectation_value(Hd, Ket{N}(Int128(i)))) for i in 0:2^N-1]) - 1))
    @test_logs (:warn, r"CEPA ill-posed"i) match_mode=:any subspace_estimates(H, ψbad, thresh=0, check=true, verbose=0)

    # CMX(2) needs I3 > 0; clipping the whole FOIS away leaves I3 == 0
    @test_logs (:warn, r"CMX\(2\) ill-posed"i) match_mode=:any subspace_estimates(H, ψ, thresh=1e12, check=true, verbose=0)
end

@testset "size extensivity" begin
    # The invariant that motivates tracking CEPA and CMX(2) rather than FOIS-CI
    # or PDS: on M non-interacting fragments the estimate must be exactly M
    # times the one-fragment estimate.
    frag = DBF.heisenberg_1D(4, 0.6, 0.6, 1.0, x=0.5, z=0.3)
    ψ1 = best_reference(frag, verbose=0)

    r1 = subspace_estimates(frag, ψ1, thresh=0, check=false, verbose=0)

    H, ψ = frag, ψ1
    for M in 2:3
        H = osum(H, frag)
        ψ = otimes(ψ, ψ1)
        r = subspace_estimates(H, ψ, thresh=0, check=false, verbose=0)
        @test abs(r.e0   - M*r1.e0)   < 1e-9
        @test abs(r.cepa - M*r1.cepa) < 1e-9 * max(1, abs(M*r1.cepa))
        @test abs(r.cmx2 - M*r1.cmx2) < 1e-9 * max(1, abs(M*r1.cmx2))
    end

    # contrast: the variational eigenvalue over the same space is NOT extensive,
    # so a non-additivity here is the expected behaviour, not a regression
    H2 = osum(frag, frag); ψ2 = otimes(ψ1, ψ1)
    xz1, xz2 = pack_x_z(frag), pack_x_z(H2)
    fois(xz, k) = real(eigvals(Hermitian(Matrix(subspace_sparse(xz, DBF.fois_space(xz, k, thresh=0)[2]))))[1])
    @test abs(fois(xz2, ψ2) - 2*fois(xz1, ψ1)) > 1e-6
end

@testset "best_reference" begin
    Random.seed!(7)
    N = 8
    H = rand(PauliSum{N}, n_paulis=200); H = H + H'
    Hd = diag(H)
    brute = argmin([real(expectation_value(Hd, Ket{N}(Int128(i)))) for i in 0:2^N-1])

    ψ = best_reference(H, verbose=0)
    @test ψ == Ket{N}(Int128(brute - 1))

    # greedy path (forced by lowering the exhaustive cutoff) finds the same
    # optimum on a problem small enough to check
    ψg = best_reference(H, exhaustive_max=4, restarts=40, verbose=0)
    @test real(expectation_value(Hd, ψg)) <= real(expectation_value(Hd, ψ)) + 1e-10

    # weight-preserving search stays in its particle-number sector
    ψ0 = Ket{N}(Int128(0b00001111))
    ψw = best_reference(H, preserve_weight=true, start=ψ0, restarts=10, verbose=0)
    @test count_ones(ψw.v) == count_ones(ψ0.v)
end

@testset "dbf_groundstate estimator hooks" begin
    N = 8
    H = DBF.heisenberg_1D(N, 0.8, 0.8, 1.0, x=0.4, z=0.5)
    ψ = best_reference(H, verbose=0)
    out = dbf_groundstate(H, ψ, max_iter=4, verbose=0, compute_var_error=false,
                          compute_cepa=true, compute_cmx=true, estimator_thresh=1e-6)

    for k in ("cepa_per_grad", "cmx2_per_grad")
        @test haskey(out, k)
        @test length(out[k]) == length(out["energies_per_grad"])
        @test all(isfinite, out[k])
    end

    # stride records NaN on skipped iterations (never a carried-forward value,
    # which a downstream fit would misread as a converged plateau)
    o2 = dbf_groundstate(H, ψ, max_iter=6, verbose=0, compute_var_error=false,
                         compute_cepa=true, estimator_thresh=1e-6, estimator_stride=3)
    c2 = o2["cepa_per_grad"]
    @test length(c2) == length(o2["energies_per_grad"])
    @test count(isfinite, c2) < length(c2)          # some were skipped
    @test isfinite(c2[1]) && isfinite(c2[end])      # first and last always computed

    # both estimators must sit below the variational energy they correct
    efci = minimum(real(eigvals(Hermitian(Matrix(H)))))
    for (ev, ec, ex) in zip(out["energies_per_grad"], out["cepa_per_grad"], out["cmx2_per_grad"])
        @test ec <= ev + 1e-8
        @test ex <= ev + 1e-8
    end
    # and the corrected estimate must beat the bare expectation value
    @test abs(out["cepa_per_grad"][end] - efci) < abs(out["energies_per_grad"][end] - efci)
end
