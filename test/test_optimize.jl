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

# @testset "sci" begin
function test()
    Random.seed!(2)
    N = 6
    H = rand(PauliSum{N}, n_paulis=100)
    H += H'
    k = rand(Ket{N})
    vref = KetSum(N, T=ComplexF64)
    vref[k] = 1

    gvec = Vector{PauliBasis{N}}([rand(PauliBasis{N}) for i in 1:20])

    DBF.optimize_sequence(H, vref, gvec)
end

test()