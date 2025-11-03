using Test
using Random
using Printf
using PauliOperators
using LinearAlgebra
using DBF
using BenchmarkTools

@testset "helpers" begin
# function test()
    Random.seed!(2)
    N = 6
    A = rand(PauliSum{N}, n_paulis=200)
    A += A'
    B = rand(PauliSum{N}, n_paulis=200)
    B += B'
    
    # @show DBF.largest(A*B-B*A)
    # @show DBF.max_of_commutator(A,B,clip=-1)
    # @show DBF.max_of_commutator2(A,B,n_top=10000)
    # @btime DBF.largest($A*$B-$B*$A)
    # @btime DBF.max_of_commutator($A,$B,clip=-1)
    a1 = DBF.largest(A*B-B*A)
    a2 = DBF.max_of_commutator(A,B,clip=-1)
    a3 = DBF.max_of_commutator2(A,B,n_top=100000)
    coeff, G = findmax(v -> abs(v), a3) 
    a3 = PauliSum(G*a3[G])

    @test norm(a1 - a2) < 1e-12 
    @test norm(a1 - a3) < 1e-12 

    A = rand(PauliSum{N}, n_paulis=2000)

    #
    # diag_GOG test
    G = rand(PauliBasis{N})
    d1 = diag(G*A*G)
    d2 = DBF.diag_GOG(G,A)
    
    # @btime diag($G*$A*$G)
    # @btime DBF.diag_GOG($G,$A)
    
    @test norm(d1 - d2) ≈ 0
   
    
    #
    # diag_commutator test
    d1 = diag(G*A - A*G)
    d2 = DBF.diag_commutator(G,A)
    @show length(d1), length(d2)
    @test norm(d1 - d2) ≈ 0


    #
    H = rand(PauliSum{N}, n_paulis=100) 
    k = KetSum{N,ComplexF64}()
    for i in 1:10
        k[Ket{N}(i)] = rand() + 1im * rand()
    end
    @test abs(DBF.expectation_value(H,k) - Vector(k)'*Matrix(H)*Vector(k)) < 1e-14


    # 
    
    H = rand(PauliSum{N}, n_paulis=100)
    H += H' 
    Z = PauliSum(N)
    for i in 1:N
        Z += PauliBasis(Pauli(N, Z=[i]))
    end
    ref = Z*H - H*Z
    tst = DBF.commute_with_Zs(H)
    @test norm(ref + -1*tst) < 1e-14
   
    for i in 1:N
        Z = DBF.create_0_projector(N, i)
        ref = Z*H - H*Z
        tst = DBF.commutator(Z,H)
        @test norm(ref + -1*tst) < 1e-13
    end


    N = 5
    a = rand(PauliSum{N}, n_paulis=1000)
    b = rand(PauliSum{N}, n_paulis=1000)
    @test abs(tr(a'*b)/2^N - inner_product(a,b)) < 1e-12

    N = 5
    a = rand(KetSum{N}, n_terms=1000)
    b = rand(KetSum{N}, n_terms=1000)
    @test abs(Vector(a)'*Vector(b) - inner_product(a,b)) < 1e-12


    ## momoents 
    N = 5
    H = rand(PauliSum{N}, n_paulis=1000)
    H += H'
    Hmat = Matrix(H)
    ψ = Ket{N}(0) 

    k1 = expectation_value(H,ψ)
    k2 = DBF.variance(H,ψ)
    _,_,k3 = DBF.skewness(H,ψ)
   
    ψv = Vector(ψ)
    m1 = ψv'*Hmat*ψv 
    m2 = ψv'*Hmat*Hmat*ψv 
    m3 = ψv'*Hmat*Hmat*Hmat*ψv 

    @test abs(m1-k1) < 1e-12
    @test abs((m2 - m1^2) - k2) < 1e-12
    @test abs((m3 - 3*m2*m1 + 2*m1^3) - k3) < 1e-10

end

# test()