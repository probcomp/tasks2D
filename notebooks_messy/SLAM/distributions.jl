"""
Some Gen distributions used in the SLAM tasks.
"""

using Gen
import GridWorlds

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