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

ModelState = NamedTuple{(:pos, :is_windy, :t, :hit_wall), Tuple{Vector{Float64}, Bool, Int, Bool}}
@gen (static) function model_init(params)
    w = params.map # a map, represented as a GridWorlds.GridWorld
    
    cell ~ uniform_from_set(GridWorlds.empty_cells(w))
    
    # Cell (i, j) corresponds to the region from i-1 to i and j-1 to j
    x ~ uniform(cell[1] - 1, cell[1])
    y ~ uniform(cell[2] - 1, cell[2])
    
    is_windy ~ bernoulli(params.init_wind_prob)

    return ModelState(([x, y], is_windy, 0, false))
end
@gen (static) function motion_model(state, a, params)
    (pos, was_windy, t_prev, prev_hit_wall) = state
    w = params.map
    
    wind_prob = was_windy ? params.stay_windy_prob : params.become_windy_prob
    is_windy ~ bernoulli(wind_prob)

    next_pos_det = det_next_pos(pos, a, params.step.Δ)
    noisy_next_pos ~ broadcasted_normal(next_pos_det, is_windy ? params.step.σ_windy : params.step.σ_normal)
    (next_pos, hit_wall) = handle_wall_intersection(pos, noisy_next_pos, w)
    
    return ModelState((next_pos, is_windy, t_prev + 1, prev_hit_wall || hit_wall))
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

############

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

pos_to_step_cm(pos) = choicemap(
    (:noisy_next_pos, pos)
)
function nonoise_nextpos(pos, a, Δ, gridworld)
    newpos = det_next_pos(pos, a, Δ)
    final_pos, did_collide = handle_wall_intersection(pos, newpos, gridworld)
    return final_pos
end