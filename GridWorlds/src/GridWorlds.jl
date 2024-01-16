"""
Basic functionality for working with 2D grid worlds.
"""
module GridWorlds

# Gridworld interface & accessor functions
include("gridworld.jl")
export GridWorld, GridCell, empty, wall, agent
export agentpos, place_agent, move_agent, newpos, empty_cells

# Functional implementation of a gridworld
include("fgridworld.jl")

# Ray-tracing in gridworlds
include("raytrace.jl")
export ray_trace_distances, points_from_raytracing, wall_segments

# Loading maps from datasets, or from manually created specifications
include("maps.jl")
export load_custom_map, load_houseexpo_gridworld

# Gridworld visualizations
include("viz/visualize.jl")

end # module GridWorlds
