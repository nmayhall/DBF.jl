using Test
using Random
using Printf
using PauliOperators
using LinearAlgebra
using DBF
using BenchmarkTools
using OrderedCollections
using LinearMaps
using KrylovKit

@testset "sci" begin
# function test()
    Random.seed!(2)
    N = 9
    H = rand(PauliSum{N}, n_paulis=100)
    DBF.coeff_clip!(H)
    H += H'

    xzH = DBF.pack_x_z(H)
    vref = KetSum(N, T=ComplexF64)
    vref[rand(Ket{N})] = 1
    sig = DBF.matvec(xzH,vref)
    sig2 = DBF.matvec(xzH,sig)

    # display(vref)
    # display(sig)
    @test norm(Matrix(H)*Vector(vref) - Vector(sig) ) < 1e-15
    @test norm(Matrix(H)*Matrix(H)*Vector(vref) - Vector(DBF.matvec(xzH,sig)) ) < 1e-12

    @show length(sig), length(sig2)
    # #project sig2 onto sig
    for (ki,si) in sig2
        if haskey(sig, ki) == false
            sig2[ki] = 0
        else
            # sig[ki] = si
        end
    end
    @show norm(Vector(sig2)) 
    @show norm(Vector(DBF.subspace_matvec(xzH, sig))) 
    @test norm(Vector(sig2) - Vector(DBF.subspace_matvec(xzH, sig)) ) < 1e-12

    Hm = Matrix(H)
    
    # project onto sig
    Hmsp = zeros(ComplexF64, length(sig), length(sig))
    for (idxi, (ki,si)) in enumerate(sig)
        for (idxj, (kj,sj)) in enumerate(sig)
            Hmsp[idxi, idxj] = Hm[ki.v, kj.v]
        end
    end

    @test norm(Matrix(xzH, collect(keys(sig))) - Matrix(H, [i for i in keys(sig)])) < 1e-15
    @time Matrix(xzH, collect(keys(sig))) 
    @time Matrix(H, [i for i in keys(sig)])
    # @test norm(Matrix(xzH, sig)*Vector([s for (k,s) in sig]) - Matrix(DBF.subspace_matvec(xzH,sig)))

    # Krylov
    k0 = vref
    k1 = DBF.matvec(xzH, k0)
    k2 = DBF.subspace_matvec(xzH, k1)
    k3 = DBF.subspace_matvec(xzH, k2)

    basis = [i for i in keys(k1)]
    dim = length(basis)

    Hmap = LinearMap(xzH, basis)
    vguess = zeros(dim)
    vguess[1] = 1
    
    # @show Hmap * vguess
    
    time = @elapsed e, v, info = KrylovKit.eigsolve(Hmap, vguess, 1, :SR, 
                                                verbosity   = 1, 
                                                maxiter     = 10, 
                                                issymmetric = true, 
                                                ishermitian = true, 
                                                eager       = true,
                                                tol         = 1e-8)
    @show e
    @show eref = minimum(eigvals(Matrix(xzH, collect(keys(k1)))))
    @test abs(e[1] - eref) < 1e-12
end

@testset "cepa" begin
# function test()
    println("\n CEPA test")
    Random.seed!(2)
    N = 8
    
    H = DBF.heisenberg_1D(N, -1, -1, -1, x=.0)
    H += H'
    for i in 1:N
        if i%2 == 1
            H = Pauli(N, X=[i]) * H * Pauli(N, X=[i])
        end
    end
    
    
    # kidx = argmin([real(expectation_value(H,Ket{N}(ψi))) for ψi in 0:2^N-1])
    # @show ψ = Ket{N}(kidx) 
    
    ψ = Ket{N}(Int128(0))
    @show ψ

    @time H, g, θ = DBF.dbf_groundstate(H, ψ,
                                n_body=1,
                                max_iter=120,
                                conv_thresh=1e-3,
                                evolve_coeff_thresh=1e-4,
                                evolve_weight_thresh=49,
                                grad_coeff_thresh=1e-3,
                                search_n_top=100)
    # @time H, g, θ = DBF.dbf_diag(H, 
    #                             max_iter=120,
    #                             conv_thresh=1e-3,
    #                             evolve_coeff_thresh=1e-3,
    #                             search_n_top=10)
    n1 = norm(H)^2
    for (i,(c,p)) in enumerate(zip(DBF.get_weight_counts(H), DBF.get_weight_probs(H)))
        @printf(" Weight: %4i %12i %12.8f\n", i, c, p/n1)
    end
       
    basis = collect(keys(rand(KetSum{N}, n_terms=23)))
    Hmap = LinearMap(DBF.pack_x_z(H), basis)
    Hmat = Matrix(H)
    Hmatss = zeros(length(basis), length(basis)) 
    idx = PauliOperators.index
    for (i,keti) in enumerate(basis)
        for (j,ketj) in enumerate(basis)
            Hmatss[i,j] = Hmat[idx(keti), idx(ketj)]
        end
    end
    vvec = rand(length(basis))
    vvec = vvec/norm(vvec)
    v = KetSum(basis)
    fill!(v,vvec,basis)
    s = DBF.subspace_matvec(pack_x_z(H),v)
    sfull = DBF.matvec(pack_x_z(H),v)
    @test norm(Vector(s) - Vector(project(sfull, basis))) < 1e-12
    
    # Compare subspace matvec with full followed by projection
    @test norm(Vector(s,basis) - Hmatss*vvec) < 1e-12
    
    # Compare the LinearMap (which does subspace matvec) to full matvec, followed by projection into basis
    @test norm(Hmap*vvec - Vector(project(DBF.matvec(pack_x_z(H),v), basis), basis)) < 1e-12


    # eexact,_ = eigvals(Matrix(H))
    # @printf(" Exact: %12.8f\n", eexact[1])

    e0, e, v, basis = DBF.fois_ci(H, ψ, thresh=1e-2, tol=1e-5)
    # @test abs(e[1] - -58.34688027) < 1e-6
    v0 = KetSum(basis)
    fill!(v0, v[1], basis)
    
    e0, e, v, basis = DBF.fois_ci(H, ψ, thresh=1e-4, v0=v0, tol=1e-5)
    # @test abs(e[1] - -58.34688027) < 1e-6
    
    
    e0, e, x, basis = DBF.cepa(H, ψ, thresh=1e-3, tol=1e-5)
    x0 = KetSum(basis)
    fill!(x0, x, basis)
    
    e0, e, x, basis = DBF.cepa(H, ψ, thresh=1e-4, x0=x0, tol=1e-5)
    
    e0, e2 = DBF.pt2(H, ψ)
    @printf(" E0 = %12.8f EPT2 = %12.8f \n", e0, e0+e2)
    # @test abs(e0+e2 - -57.74827344) < 1e-6
    
    # I can't quite test these yet, because the dbf_groundstate gives different results
    # sometimes. I assume it's an issue of degenerate operators leading to different sequences. 
end

# test()