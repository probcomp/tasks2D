using Gen         # Gen probabilistic programming library
import GenParticleFilters # Additional particle filtering functionality for Gen
import GridWorlds # Simple gridworld functionality
import LineWorlds
const L = LineWorlds
import LineWorlds: cast # Ray caster
import GenPOMDPs  # Beginnings of a Gen POMDP library

import Tasks2D
import LinearAlgebra
import Dates

# Distribution to sample uniformly from a Julia Set
using Tasks2D.Distributions: uniform_from_set

ModelState = NamedTuple{(:pos, :t, :hit_wall), Tuple{Vector{Float64}, Int, Bool}}
@gen (static) function uniform_agent_pos(params)
    w = params.map # a map, represented as a GridWorlds.GridWorld
    
    cell ~ uniform_from_set(GridWorlds.empty_cells(w))
    
    # Cell (i, j) corresponds to the region from i-1 to i and j-1 to j
    x ~ uniform(cell[1] - 1, cell[1])
    y ~ uniform(cell[2] - 1, cell[2])
    
    return ModelState(([x, y], 0, false))
end

function pos_to_init_cm(pos)
    (x, y) = pos
    i = Int(ceil(x))
    j = Int(ceil(y))
    return choicemap(
        (:cell, (i, j)),
        (:x, x),
        (:y, y)
    )
end
function state_to_pos(state)
    return state[1]
end

function det_next_pos(pos, a, Δ)
    (x, y) = pos
    a == :up    ? [x, y + Δ] :
    a == :down  ? [x, y - Δ] : 
    a == :left  ? [x - Δ, y] :
    a == :right ? [x + Δ, y] :
    a == :stay  ? [x, y]     :
                error("Unrecognized action: $a")
end

function handle_wall_intersection(prev, new, gridworld)
    walls = GridWorlds.nonempty_segments(gridworld)
    move = L.Segment(prev, new)
    
    min_collision_dist = Inf
    vec_to_min_dist_collision = nothing
    for i in 1:(size(walls)[1])
        wall = walls[i, :]
        # print("wall: $wall")
        do_intersect, dist = L.Geometry.cast(move, L.Segment(wall))

        if do_intersect && dist ≤ L.Geometry.norm(move)
            if dist < min_collision_dist
                min_collision_dist = dist
                vec_to_min_dist_collision = L.Geometry.diff(move)
            end
        end
    end
    
    if !isnothing(vec_to_min_dist_collision)
        dist = min_collision_dist
        if dist < 0.05
            return (prev, true)
        else
            normalized_vec = (vec_to_min_dist_collision / L.Geometry.norm(vec_to_min_dist_collision))
            collision_pt = prev + (dist - 0.04) * normalized_vec
            return (collision_pt, true)
        end
    end
    
    return (new, false)
end

@gen (static) function motion_model(state, a, params)
    (pos, t_prev, prev_hit_wall) = state
    w, σ = params.map, params.step.σ
    
    next_pos_det = det_next_pos(pos, a, params.step.Δ)
    noisy_next_pos ~ broadcasted_normal(next_pos_det, params.step.σ)
    (next_pos, hit_wall) = handle_wall_intersection(pos, noisy_next_pos, w)
    
    return ModelState((next_pos, t_prev + 1, prev_hit_wall || hit_wall))
end
pos_to_step_cm(pos) = choicemap(
    (:noisy_next_pos, pos)
)
function nonoise_nextpos(pos, action, Δ, gridworld)
    (a, _) = action
    newpos = det_next_pos(pos, a, Δ)
    final_pos, did_collide = handle_wall_intersection(pos, newpos, gridworld)
    return final_pos
end

@gen function observe_noisy_distances(state, params)
    (pos, t, _) = state

    p = reshape([pos..., params.obs.orientation], (1, 3))
    _as = L.create_angles(params.obs.fov, params.obs.n_rays)
    wall_segs = GridWorlds.wall_segments(params.map)
    strange_segs = GridWorlds.strange_segments(params.map)
    @assert isempty(strange_segs) "Strange objects not supported in this model variant"
    
    w, s_noise, outlier, outlier_vol, zmax = params.obs.wall_sensor_args
    dists_walls = L.cast(p, wall_segs; num_a=params.obs.n_rays, zmax)
    dists_walls = reshape(dists_walls, (:,))

    wall_measurements = dists_walls
    obs ~ Gen.broadcasted_normal(wall_measurements, s_noise)

    return obs
end

######################
# Action controllers #
######################
@gen function _random_controller(st, obs)
    aidx ~ categorical([0.25, 0.25, 0.25, 0.25])
    action = [:up, :right, :left, :down][aidx]
    return (action, nothing)
end
random_controller = GenPOMDPs.Controller(_random_controller, nothing)

@gen function _meandering_controller(current_direction, obs)
    change_dir ~ bernoulli(0.2)
    if change_dir || isnothing(current_direction)
        aidx ~ categorical([0.25, 0.25, 0.25, 0.25])
        action = [:up, :right, :left, :down][aidx]
    else
        action = current_direction
    end
    return (action, action)
end
meandering_controller = GenPOMDPs.Controller(_meandering_controller, nothing)

@gen function _meandering_wallavoiding_controller(current_direction, obs)
    # d, r, u, l
    directional_dist_inds = [Int(floor(i)) for i in LinRange(1, length(obs) + 1, 5)[1:4]]
    relevant_dists = obs[directional_dist_inds]
    near_walls = Dict()
    near_walls[:down] = relevant_dists[1] < 1
    near_walls[:right] = relevant_dists[2] < 1
    near_walls[:up] = relevant_dists[3] < 1
    near_walls[:left] = relevant_dists[4] < 1

    if near_walls[:down]
        action = :up
    elseif near_walls[:right]
        action = :left
    elseif near_walls[:up]
        action = :down
    elseif near_walls[:left]
        action = :right
    else
        action = nothing
    end

    if isnothing(action)
        change_dir ~ bernoulli(0.2)
        if change_dir || isnothing(current_direction)
            aidx ~ categorical([0.25, 0.25, 0.25, 0.25])
            action = [:up, :right, :left, :down][aidx]
        else
            action = current_direction
        end
    end

    i = 0
    while near_walls[action]
        # println("near_walls[$action]")
        if i > 20
            break
        end
        i += 1
        aidx = {(:aidx, i)} ~ categorical([0.25, 0.25, 0.25, 0.25])
        action = [:up, :right, :left, :down][aidx]
    end
    # println("near walls: $(near_walls[action])")

    return (action, action)
end
meandering_wallavoiding_controller = GenPOMDPs.Controller(_meandering_wallavoiding_controller, nothing)