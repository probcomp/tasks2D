"""
Tasks2D.Distributions

Some Gen distributions used in the grid tasks.
"""
using Gen
import FunctionalCollections: PersistentVector
import GridWorlds

# Snippets to sample map (when a map is not given)
# Use IID Bernoulli to sample each map cell
# @dist flip_wall(wall_prob::Real) = bernoulli(wall_prob)) ? GridWorlds.wall : GridWorlds.empty
# @dist flip_wall(wall_prob::Real) = [GridWorlds.wall, GridWorlds.empty][categorical([wall_prob, 1-wall_prob])]
# ^ for whatever reason this errors out because bernoulli(wall_prob) cannot be interpreted as Bool

@gen (static) function generate_gridcell(wall_prob::Real)
    is_wall ~ bernoulli(wall_prob)
    return is_wall ? GridWorlds.wall : GridWorlds.empty
end

@gen (static) function generate_iid_bernoulli_map(width::Int, height::Int, wall_prob::Real)
    raw_map ~ Map(Map(generate_gridcell))(fill(fill(wall_prob, height), width))
    return GridWorlds.FGridWorld(raw_map, nothing, (width, height))
end

# Handle PersistentVector{Any} returns by the Map combinator
function GridWorlds.FGridWorld(grid::PersistentVector{PersistentVector{Any}}, pos::Union{Nothing,Tuple{Int,Int}}, size::Tuple{Int,Int})
    grid = PersistentVector(PersistentVector{GridWorlds.GridCell}.(grid))
    GridWorlds.FGridWorld(grid, pos, size)
end