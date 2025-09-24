using DBF
using PauliOperators
using LinearAlgebra
using Test

@testset "DBF.jl" begin

    include("test_theta_opt.jl")
    include("test_evolve.jl")
    include("test_diag_dbf.jl")
end
