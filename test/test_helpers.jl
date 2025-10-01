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
    
    @show DBF.largest(A*B-B*A)
    @show DBF.max_of_commutator(A,B,clip=-1)
    @show DBF.max_of_commutator2(A,B,clip=-1)
    # @btime DBF.largest($A*$B-$B*$A)
    # @btime DBF.max_of_commutator($A,$B,clip=-1)
    a1 = DBF.largest(A*B-B*A)
    a2 = DBF.max_of_commutator(A,B,clip=-1)
    a3 = DBF.max_of_commutator2(A,B,clip=-1)
    
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
end

# test()