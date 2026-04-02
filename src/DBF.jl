module DBF

using PauliOperators
using Printf
using LinearAlgebra
using OrderedCollections
using Polynomials: Polynomials
import PauliOperators: truncate!

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


# Functions now exported by PauliOperators:
#   evolve, evolve!, inner_product, offdiag, coeff_clip!, weight_clip!,
#   hadamard, cnot, X_gate, Y_gate, Z_gate, S_gate, T_gate,
#   variance, commutator, anticommutator, majorana_weight, majorana_weight_clip!

export dbf_diag
export dbf_groundstate
export dbf_disentangle
export adapt
export pack_x_z
export project
export extrapolate_energy
export plot_extrapolation

end
