module DBF

using PauliOperators
using Printf
using LinearAlgebra
using OrderedCollections

# Hmap = FermiCG.LinOpMat{T}(matvec, length(vec), true)
XZPauliSum{T} = Dict{Int128,Dict{Int128,T}}


include("helpers.jl")
include("hamiltonians.jl")
include("diagonalization.jl")
include("groundstate.jl")
include("disentangle.jl")
include("adapt.jl")
include("evolve.jl")
include("sci.jl")


export evolve
export inner_product
export offdiag
export inner_product
export dbf_diag
export dbf_groundstate
export dbf_disentangle
export adapt 

end
