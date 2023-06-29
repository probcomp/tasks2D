using VoxelRayTracers # For lineworld -> gridworld
using GridWorlds      # For reasoning about motion in the gridded environment
const GW = GridWorlds
using AStarSearch     # For gridworld path planning

const state_addr = GenPOMDPs.state_addr

# Parameters for the grid discretization used for planning
function get_planning_params(walls, bounding_box, ϵ=0.25)
    (grid, edges, l_to_g, g_to_l) = line_to_grid(walls, bounding_box, ϵ)
    w = GridWorlds.boolmatrix_to_grid(grid, (length(edges[1]), length(edges[2])));
    return (w, grid, edges, l_to_g, g_to_l)
end

function line_to_grid(_segs, _bb, ϵ)
    (x1, y1), (x2, y2) = _bb
    edges = ((x1 - ϵ):ϵ:(x2 + ϵ), (y1 - ϵ):ϵ:(y2 + ϵ))

    grid = [false for _ in edges[1], _ in edges[2]]
    for seg in _segs
        if Geo.diff(seg) ≈ [0, 0]
            continue
        end
        ray = (position=seg.x, velocity=Geo.diff(seg))
        for hit in eachtraversal(ray, edges)
                                    # TODO: is this a hack or no?
            if hit.exit_time ≤ 1. #|| (hit.entry_time == 1.0 && (ray.velocity[1] > 0 || ray.velocity[2] > 0))
                grid[hit.voxelindex] = true
            end
        end
    end

    linecoords_to_gridcoords(x, y) = (
        Int(round((x - edges[1][1] + ϵ) / ϵ)),
        Int(round((y - edges[2][1] + ϵ) / ϵ))
    )
    gridcoords_to_linecoords(x, y) = (
        edges[1][x],
        edges[2][y]
    )

    return grid, edges, linecoords_to_gridcoords, gridcoords_to_linecoords
end

taxi_dist((x, y), (x2, y2)) = abs(x - x2) + abs(y - y2)
function find_action_using_grid_search(planning_params, start_linecoords, goal_linecoords)::Symbol
    (w, grid, edges, l_to_g, g_to_l) = planning_params
    
    actions = (:up, :down, :left, :right, :stay)
    initialpos = l_to_g(start_linecoords...)
    goalpos    = l_to_g(goal_linecoords...)

    results = astar(
        # state to neighbors
        pos -> unique(GW.newpos(w, pos, dir) for dir in actions),
        initialpos, # Initial world state
        goalpos; # Goal world state
        heuristic = ((pos, goal) -> taxi_dist(pos, goal)),
        isgoal = ((pos, goal) -> pos == goal),
        timeout = 10.
    )

    length(results.path) == 1 && return :stay
    
    next_state = results.path[2]
    # println("next_state: ", g_to_l(next_state...))
    action = actions[findfirst(
        GW.newpos(w, initialpos, dir) == next_state for dir in actions
    )]

    # @assert action isa Symbol
    # println(action)

    return action
end