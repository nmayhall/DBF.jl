using DBF
using PauliOperators
using LinearAlgebra
using Test

@testset "DBF.jl" begin

    include("test_theta_opt.jl")
    include("test_evolve.jl")
    include("test_diag_dbf.jl")
    include("test_groundstate_dbf.jl")
    include("test_adapt.jl")
    include("test_helpers.jl")
end
