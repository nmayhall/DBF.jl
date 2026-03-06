using Random

# ============================================================
# Abstract Types
# ============================================================

abstract type TruncationStrategy end
abstract type CorrectionAccumulator end


# ============================================================
# Truncation Strategy Types
# ============================================================

"""
    NoTruncation()

Identity truncation — does nothing.
"""
struct NoTruncation <: TruncationStrategy end

"""
    CoeffTruncation(thresh::Float64)

Remove Pauli terms with |coefficient| < `thresh`.
"""
struct CoeffTruncation <: TruncationStrategy
    thresh::Float64
end
CoeffTruncation() = CoeffTruncation(1e-6)

"""
    WeightTruncation(max_weight::Int)

Remove Pauli terms with Pauli weight > `max_weight`.
"""
struct WeightTruncation <: TruncationStrategy
    max_weight::Int
end

"""
    MajoranaWeightTruncation(max_weight::Int)

Remove Pauli terms with Majorana weight > `max_weight`.
"""
struct MajoranaWeightTruncation <: TruncationStrategy
    max_weight::Int
end

"""
    CompositeTruncation(strategies...)

Apply multiple truncation strategies in sequence.
"""
struct CompositeTruncation <: TruncationStrategy
    strategies::Vector{TruncationStrategy}
end
CompositeTruncation(s::TruncationStrategy...) = CompositeTruncation(collect(TruncationStrategy, s))

"""
    StochasticTruncation(n_keep::Int; rng=Random.default_rng())

Stochastically sample `n_keep` terms via importance sampling with probabilities
proportional to |c_i|^2. Kept terms are rescaled to preserve expected norm.
"""
struct StochasticTruncation <: TruncationStrategy
    n_keep::Int
    rng::AbstractRNG
end
StochasticTruncation(n_keep::Int) = StochasticTruncation(n_keep, Random.default_rng())


# ============================================================
# apply! — raw truncation dispatch
# ============================================================

function apply!(O::PauliSum{N}, ::NoTruncation) where N
    return O
end

function apply!(O::PauliSum{N}, s::CoeffTruncation) where N
    return coeff_clip!(O, thresh=s.thresh)
end

function apply!(O::PauliSum{N}, s::WeightTruncation) where N
    return weight_clip!(O, s.max_weight)
end

function apply!(O::PauliSum{N}, s::MajoranaWeightTruncation) where N
    return majorana_weight_clip!(O, s.max_weight)
end

function apply!(O::PauliSum{N}, s::CompositeTruncation) where N
    for strategy in s.strategies
        apply!(O, strategy)
    end
    return O
end

function apply!(O::PauliSum{N}, s::StochasticTruncation) where N
    length(O) <= s.n_keep && return O

    # Efraimidis-Spirakis weighted sampling without replacement:
    # assign key_i = rand()^(1/w_i), keep the n_keep largest keys
    keys_vec = collect(keys(O))
    weights = [abs2(O[k]) for k in keys_vec]
    norm_sq = sum(weights)
    sampling_keys = [rand(s.rng)^(1.0/w) for w in weights]

    # Find the n_keep largest sampling keys via partialsortperm
    kept_idx = partialsortperm(sampling_keys, 1:s.n_keep, rev=true)
    kept_set = Set(keys_vec[i] for i in kept_idx)

    # Compute norm of kept terms before rescaling
    kept_norm_sq = sum(abs2(O[k]) for k in kept_set)

    # Remove unsampled terms
    filter!(p -> p.first in kept_set, O)

    # Rescale to preserve norm
    if kept_norm_sq > 0
        scale = sqrt(norm_sq / kept_norm_sq)
        for k in keys(O)
            O[k] *= scale
        end
    end

    return O
end


# ============================================================
# Correction Accumulator Types
# ============================================================

"""
    NoCorrection()

Track nothing during truncation. Zero overhead (compiled away).
"""
struct NoCorrection <: CorrectionAccumulator end

"""
    EnergyCorrection(ψ::Ket{N})

Track accumulated change in ⟨ψ|O|ψ⟩ due to truncation.
"""
mutable struct EnergyCorrection{N} <: CorrectionAccumulator
    ψ::Ket{N}
    accumulated_energy::Float64
end
EnergyCorrection(ψ::Ket{N}) where N = EnergyCorrection{N}(ψ, 0.0)

"""
    EnergyVarianceCorrection(ψ::Ket{N})

Track accumulated changes in both ⟨ψ|O|ψ⟩ and Var(O,ψ) due to truncation.
"""
mutable struct EnergyVarianceCorrection{N} <: CorrectionAccumulator
    ψ::Ket{N}
    accumulated_energy::Float64
    accumulated_variance::Float64
end
EnergyVarianceCorrection(ψ::Ket{N}) where N = EnergyVarianceCorrection{N}(ψ, 0.0, 0.0)

# Accessor helpers for printing/logging when accumulator type is not known statically
_get_accumulated_energy(::NoCorrection) = 0.0
_get_accumulated_energy(c::EnergyCorrection) = c.accumulated_energy
_get_accumulated_energy(c::EnergyVarianceCorrection) = c.accumulated_energy

_get_accumulated_variance(::NoCorrection) = 0.0
_get_accumulated_variance(::EnergyCorrection) = 0.0
_get_accumulated_variance(c::EnergyVarianceCorrection) = c.accumulated_variance


# ============================================================
# measure — snapshot quantities before/after truncation
# ============================================================

measure(::PauliSum, ::NoCorrection) = nothing

function measure(O::PauliSum{N}, corr::EnergyCorrection{N}) where N
    return (energy = real(expectation_value(O, corr.ψ)),)
end

function measure(O::PauliSum{N}, corr::EnergyVarianceCorrection{N}) where N
    return (energy  = real(expectation_value(O, corr.ψ)),
            variance = real(variance(O, corr.ψ)))
end


# ============================================================
# accumulate! — update accumulator with before/after diffs
# ============================================================

accumulate!(::NoCorrection, before, after) = nothing

function accumulate!(corr::EnergyCorrection, before, after)
    corr.accumulated_energy += after.energy - before.energy
end

function accumulate!(corr::EnergyVarianceCorrection, before, after)
    corr.accumulated_energy  += after.energy  - before.energy
    corr.accumulated_variance += after.variance - before.variance
end


# ============================================================
# truncate! — unified entry point
# ============================================================

"""
    truncate!(O::PauliSum, strategy::TruncationStrategy,
              corr::CorrectionAccumulator=NoCorrection())

Apply `strategy` to truncate `O` in-place. If a `CorrectionAccumulator` is
provided, measure quantities before and after truncation and accumulate the
differences.

Users can define new strategies by subtyping `TruncationStrategy` and
implementing `apply!(O, s)`. New correction types are defined by subtyping
`CorrectionAccumulator` and implementing `measure(O, corr)` and
`accumulate!(corr, before, after)`.
"""
function truncate!(O::PauliSum, strategy::TruncationStrategy,
                   corr::CorrectionAccumulator=NoCorrection())
    before = measure(O, corr)
    apply!(O, strategy)
    after  = measure(O, corr)
    accumulate!(corr, before, after)
    return O
end
