"""
Some Gen distributions used in the SLAM tasks.
"""

using Gen
import GridWorlds

struct MappedUniform <: Gen.Distribution{Any} end
Gen.random(::MappedUniform, mins, maxs) = [Gen.uniform(min, max) for (min, max) in zip(mins, maxs)]
function Gen.logpdf(::MappedUniform, v, mins, maxs)
    if length(v) != length(mins) || length(v) != length(maxs)
        return -Inf
    end
    return sum(logpdf(Gen.uniform, val, min, max) for (val, min, max) in zip(v, mins, maxs); init=0.0)
end
mapped_uniform = MappedUniform()
(::MappedUniform)(args...) = random(mapped_uniform, args...)

struct BernoulliMap <: Gen.Distribution{GridWorlds.GridWorld} end

function Gen.random(::BernoulliMap, width::Int, height::Int, wall_prob::Real)
    cells = Matrix{GridWorlds.GridCell}(undef, width, height)
    for i in 1:width
        for j in 1:height
            cells[i, j] = bernoulli(wall_prob) ? GridWorlds.wall : GridWorlds.empty
        end
    end
    return GridWorlds.FGridWorld(cells, nothing, (width, height))
end

function Gen.logpdf(::BernoulliMap, w::GridWorlds.GridWorld, width::Int, height::Int, wall_prob::Real)
    num_empty = length(GridWorlds.empty_cells(w))
    num_walls = width * height - num_empty
    return log(wall_prob) * num_walls + log(1 - wall_prob) * num_empty
end
bernoulli_map = BernoulliMap()
(::BernoulliMap)(args...) = random(bernoulli_map, args...)


struct MixtureMeasurement <: Gen.Distribution{Vector{Float64}} end

function Gen.random(::MixtureMeasurement, is_wall::Vector{Bool}, wall_dists::Vector{Float64}, σ_wall::Float64, strang_dists_min::Vector{Float64}, strang_dists_max::Vector{Float64})
    measurements = Vector{Float64}(undef, length(is_wall))
    # sample measurements from wall
    measurements[is_wall] = broadcasted_normal(wall_dists, σ_wall)
    # sample measurements from the strange object
    measurements[.!is_wall] = mapped_uniform(strang_dists_min, strang_dists_max)
    return measurements
end

function Gen.logpdf(::MixtureMeasurement, measurements::Vector{Float64}, is_wall::Vector{Bool}, wall_dists::Vector{Float64}, σ_wall::Float64, strang_dists_min::Vector{Float64}, strang_dists_max::Vector{Float64})
    retval = 0.0
    # logpdf of measurements from wall
    retval += logpdf(broadcasted_normal, measurements[is_wall], wall_dists, σ_wall)
    # logpdf of measurements from the strange object
    retval += logpdf(mapped_uniform, measurements[.!is_wall], strang_dists_min, strang_dists_max)
    return retval
end
mixture_measurement = MixtureMeasurement()
(::MixtureMeasurement)(args...) = random(mixture_measurement, args...)

struct ConstantLogPDFTerm <: Gen.Distribution{Any} end

function Gen.random(::ConstantLogPDFTerm, term::Real)
    return 0.0
end

function Gen.logpdf(::ConstantLogPDFTerm, x, term::Real)
    return term
end
constant_log_pdf_term = ConstantLogPDFTerm()
(::ConstantLogPDFTerm)(args...) = random(constant_log_pdf_term, args...)