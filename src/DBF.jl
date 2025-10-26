module DBF

using PauliOperators
using Printf
using LinearAlgebra
using OrderedCollections

# Hmap = FermiCG.LinOpMat{T}(matvec, length(vec), true)
XZPauliSum{T} = Dict{Int128,Vector{Tuple{Int128,T}}}


include("helpers.jl")
include("hamiltonians.jl")
include("diagonalization.jl")
include("groundstate.jl")
include("disentangle.jl")
include("adapt.jl")
include("evolve.jl")
include("schrodinger_picture.jl")


export evolve
export inner_product
export offdiag
export inner_product
export dbf_diag
export dbf_groundstate
export dbf_disentangle
export adapt
export coeff_clip!
export pack_x_z
export project 
export hadamard
export cnot
export X_gate 
export Y_gate 
export Z_gate 
export S_gate 
export T_gate 

end
