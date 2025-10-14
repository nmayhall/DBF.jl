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
    H += H'

    xzH = DBF.pack_x_z(H)
    vref = OrderedDict{Ket{N}, ComplexF64}()
    vref[rand(Ket{N})] = 1
    sig = DBF.matvec(xzH,vref)
    sig2 = DBF.matvec(xzH,sig)

    # display(vref)
    # display(sig)
    @test norm(Matrix(H)*Matrix(vref) - Matrix(sig) ) < 1e-15
    @test norm(Matrix(H)*Matrix(H)*Matrix(vref) - Matrix(DBF.matvec(xzH,sig)) ) < 1e-12

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

    @test norm(Matrix(xzH, sig) - Matrix(H, [i for i in keys(sig)])) < 1e-15
    @time Matrix(xzH, sig) 
    @time Matrix(H, [i for i in keys(sig)])
    # @test norm(Matrix(xzH, sig)*Vector([s for (k,s) in sig]) - Matrix(DBF.subspace_matvec(xzH,sig)))

    # Krylov
    k0 = vref
    k1 = DBF.matvec(xzH, k0)
    k2 = DBF.subspace_matvec(xzH, k1)
    k3 = DBF.subspace_matvec(xzH, k2)
    # tmp = deepcopy(k1)
    # @code_warntype DBF.subspace_matvec!(tmp, xzH, k1)

    basis = [i for i in keys(k1)]
    dim = length(basis)
    function my_matvec(v)
        n = length(v)
        s = zeros(eltype(v), size(v))
        vin = deepcopy(k1)
        for idx in 1:n
            vin[basis[idx]] = v[idx]
        end
        tmp = DBF.subspace_matvec(xzH, vin)
        for idx in 1:n
            s[idx] = tmp[basis[idx]]
        end
        return s
    end
    vguess = zeros(ComplexF64, dim)
    vguess[1] = 1

    Hmap = LinearMap(my_matvec, dim, dim; issymmetric=true, ismutating=false, ishermitian=true)

    # @show Hmap * vguess
    time = @elapsed e, v, info = KrylovKit.eigsolve(Hmap, vguess, 1, :SR, 
                                                verbosity   = 1, 
                                                maxiter     = 10, 
                                                issymmetric = true, 
                                                ishermitian = true, 
                                                eager       = true,
                                                tol         = 1e-8)
    @show e
    @show eref = minimum(eigvals(Matrix(xzH, k1)))
    @test abs(e[1] - eref) < 1e-12
    display(v)
end

# test()