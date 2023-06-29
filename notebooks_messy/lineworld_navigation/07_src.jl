using Revise, Plots
import LineWorlds
const L = LineWorlds
const Geo = L.Geometry

import GridWorlds
const GW = GridWorlds

using Gen
import GenPOMDPs

using GenSMCP3
# We need this to help with dualizing and undualizing values...
const DFD = GenSMCP3.GenTraceKernelDSL.DynamicForwardDiff

using VoxelRayTracers # For lineworld -> gridworld
using AStarSearch     # For gridworld path planning

mvuniform = L.ProductDistribution(uniform);
@gen function uniform_agent_pos(params)
    w = params.map # a map, represented as list of line segments
    
    mins, maxs = PARAMS.bounding_box
    pos ~ mvuniform(mins, maxs)
    
    return pos
end
@load_generated_functions()

### Motion model ###
# The next position, under a 0-noise model, with no walls.
function det_next_pos(pos, a, Δ)
    (x, y) = DFD.value(pos)
    a == :up    ? [x, y + Δ] :
    a == :down  ? [x, y - Δ] : 
    a == :left  ? [x - Δ, y] :
    a == :right ? [x + Δ, y] :
    a == :stay  ? [x, y]     :
                error("Unrecognized action: $a")
end
function handle_wall_intersection(prev, new, walls)
    move = L.Segment(DFD.value(prev), DFD.value(new))
    for wall in walls
        do_intersect, dist = L.Geometry.cast(move, wall)
        if do_intersect && dist ≤ L.Geometry.norm(move)
            return prev
        end
    end
    return new
end
@gen function motion_model(prev_pos, action, PARAMS)
    np = det_next_pos(prev_pos, action, PARAMS.step.Δ)
    pos ~ broadcasted_normal(np, PARAMS.step.σ)
    
    next_pos = handle_wall_intersection(prev_pos, pos, PARAMS.map)
    
    return next_pos
end

### Observation model ###
@gen function sensor_model(pos, params)
    p = L.Pose(pos, params.orientation)
    
    w, s_noise, outlier, outlier_vol, zmax = params.obs.sensor_args
    
    # segs = L.stack(Vector.(params.map))
    _as = L.create_angles(params.obs.fov, params.obs.n_rays)
    zs = L.cast([p], params.map; num_a=params.obs.n_rays, zmax)
    @assert !any(isinf.(zs))
    # zs = reshape(dists, (1, :))
    ỹ  = L.get_2d_mixture_components(zs, _as, w; fill_val_z=zmax)[1,:,:,:]
    @assert !any(isinf.(ỹ))

    # Observe a list of points, in the agent's coordinate frame.
    obs ~ L.sensordist_2dp3(ỹ, s_noise, outlier, outlier_vol)  

    if any(any(isinf.(o)) for o in obs)
        # @warn "got an inf obs ỹ = $ỹ"
    end
    
    return obs
end;

# POMDP of this environment
pomdp = GenPOMDPs.GenPOMDP(
    uniform_agent_pos,       # INIT   : params                      ⇝ state
    motion_model,            # STEP   : prev_state, actions, params ⇝ state
    sensor_model,            # OBS    : state, params               ⇝ observation
    (state, action) -> 0.    # UTILITY: state, action, params       → utility
)

# Generative function over trajectories of the POMDP,
# given a fixed action sequence.
trajectory_model = GenPOMDPs.ControlledTrajectoryModel(pomdp)

@load_generated_functions()

T(tr) = get_args(tr)[1]
currentpos(tr) = tr[state_addr(T(tr), :pos)]

function get_actobs_seq(groundtruth_trace)
    return map(
        trace -> (
            GenPOMDPs.observation_sequence(trace),
            GenPOMDPs.action_sequence(trace)
        ),
        groundtruth_trace
    )
end

#################
### Inference ###
#################

import GenParticleFilters
const GPF = GenParticleFilters

using GenSMCP3

# SMCP3 forward proposal
GenSMCP3.@kernel function forward_iterated_grid_proposal(tr, is_initial_step, params, new_action, new_obs, grid_args)
    if is_initial_step
        T = 0
        pos = grid_args.initial_pos
    else
        T_prev, actions, params_ = get_args(tr)
        @assert params_ == params # we have to pass in `params` since we can't get it otherwise for the initial step
        T = T_prev + 1

        # Form the grid around what would be the position of the agent at the next timestep
        # in the determinized version of the motion model.
        pos = det_next_pos(tr[state_addr(T_prev, :pos)], new_action, params.step.Δ)
    end

    j, args, grid_proposed_positions = nothing, nothing, nothing # make these available outside loop scope
    for i = 1:grid_args.n_iters
        args = L.grid_schedule(grid_args.init_grid_args, i)
        
        grid_proposed_positions, _ = L.vector_grid(GenSMCP3.GenTraceKernelDSL.DynamicForwardDiff.value(pos), args...)

        if !is_initial_step
            # The observations are highly informative, so it is okay to ignore the motion
            # logprobs in the proposal.  This results in a bit of a speedup
            # since currently the motion model involves some somewhat slow ray intersections
            # which are not parallelized.

            # Get the logprobs under the motion model.  Also get the final
            # position the agent will end up at if each given position is sampled.
            # (This may not equal the sampled position, due to wall collisions.)
            # motion_logprobs_rets = [
            #     Gen.assess(motion_model, (DFD.value(pos), new_action, params), choicemap((:pos, DFD.value(newpos))))
            #     for newpos in reshape(grid_proposed_positions, (:,))
            # ]
            # motion_logprobs = [logp for (logp, r) in motion_logprobs_rets]
            # grid_final_positions = [r for (logp, r) in motion_logprobs_rets]
            motion_logprobs = [0. for _ in reshape(grid_proposed_positions, (:,))]
            grid_final_positions = reshape(grid_proposed_positions, (:,))
        else
            motion_logprobs = [0. for _ in reshape(grid_proposed_positions, (:,))]
            grid_final_positions = grid_proposed_positions
        end

        # Evaluate the observation probabilities.
        poses = [L.Pose(p, params.orientation) for p in grid_final_positions]
        _segs = params.map
        _as = L.create_angles(params.obs.fov, params.obs.n_rays)
        log_obs_probs,  = L.eval_pose_vectors(
            Vector.(poses)::Array{Vector{Float64}},
            new_obs::Vector{Vector{Float64}},
            _segs::Vector{L.Segment},
            params.obs.sensor_args...; sorted=false,
            fov=params.obs.fov, num_a=params.obs.n_rays
        )

        # Sample a next position from the grid.
        # println("size(log_obs_probs): ", size(log_obs_probs))
        # println("size(motion_logprobs): ", size(motion_logprobs))
        # println("log_obs_probs: ", log_obs_probs)
        # println("motion_logprobs: ", motion_logprobs)
        # println("positions = ", grid_final_positions)
        # println("obs = ", new_obs)
        log_ps = log_obs_probs .+ motion_logprobs
        log_ps_tau = log_ps ./ grid_args.tau
        probs      = L.normalize_exp(log_ps_tau)
        probs      = L.raise_probs(probs, grid_args.pmin)
        j = {(:j, i)} ~ categorical(probs)
        pos = grid_final_positions[j]

        # println("sampled pos: $pos ; probability: $(probs[j]), logp_obs: $(log_obs_probs[j]), logp_motion: $(motion_logprobs[j])")
    end

    # Sample the final position in continuous space.
    vs = grid_proposed_positions
    # println("bounds for uniform draw: ", vs[j][1:2] -  args.r[1:2]/2, vs[j][1:2] + args.r[1:2]/2)
    x′  = {:pos}  ~ mvuniform(vs[j][1:2] -  args.r[1:2]/2, vs[j][1:2] + args.r[1:2]/2)

    # println("setting $((state_addr(T, :pos))) to $x′")
    return (
        choicemap((state_addr(T, :pos), x′)),
        EmptyChoiceMap() # Backward proposal is deterministic.
    )
end

# SMCP3 backward proposal
GenSMCP3.@kernel function backward_iterated_grid_proposal(tr, is_initial_step, grid_args)
    (T, actions, params) = get_args(tr)
    new_action = T > 0 ? actions[T] : nothing
    T_prev = T - 1

    fwd_ch = choicemap()

    nextpos = tr[state_addr(T, :pos)]
    fwd_ch[:pos] = nextpos

    if is_initial_step
        nextpos_det = grid_args.initial_pos
    else
        nextpos_det = det_next_pos(tr[state_addr(T_prev, :pos)], new_action, params.step.Δ)
    end

    # Step backward through the grid proposals, with the inverted schedule.
    for i=grid_args.n_iters:1
        args = L.grid_schedule(grid_args.init_grid_args, i)
        j = L.grid_index(nexpos, nextpos_det, args...; linear=true)
        fwd_ch[(:j, i)] = j
    end
    
    return (
        EmptyChoiceMap(), # The old trace just drops values; no constraints needed.
        fwd_ch
    )
end

function resampling_rule(pf_state, resampling_args)
    ng = resampling_args.n_groups
    gs = Int(resampling_args.n_particles/ng)
    for i=1:ng
        substate = pf_state[(1 + gs*(i-1)):(gs*i)]
        if GPF.get_ess(substate) < 1 + 1e-4
            GPF.pf_resample!(substate)
        end
    end
end
# _resampling_args has :n_particles and :n_groups
macro get_pf(PARAMS, _grid_args, _t0_grid_args, _resampling_args)
    quote
    GenPOMDPs.pf(
        pomdp, $(esc(PARAMS)),
        obs -> ( # Initialize via a 3-particle SMCP3 proposal.
            $(esc(_resampling_args)).n_particles, # n particles
            SMCP3Update(
                forward_iterated_grid_proposal, backward_iterated_grid_proposal,
                # is_initial_step, params, new_action, new_obs, grid_args
                (true, $(esc(PARAMS)), nothing, obs[:obs], $(esc(_t0_grid_args))), # no action before the initial observation
                # is_initial_step, grid_args
                (true, $(esc(_t0_grid_args)),)
            )
        ),
        (action, obs) -> ( # Update with the SMCP3 proposal.
            SMCP3Update(
                forward_iterated_grid_proposal, backward_iterated_grid_proposal,
                # is_initial_step, params, new_action, new_obs, grid_args
                (false, $(esc(PARAMS)), action, obs[:obs], $(esc(_grid_args))),
                # is_initial_step, grid_args
                (false, $(esc(_grid_args)),)
            ),
        );
        pre_update = state -> resampling_rule(state, $(esc(_resampling_args))),
    );
    end # quote
end

#####################
### Visualization ###
#####################

### Visualize first timestep particles ###

using Random
function visualize_tr_pf(tr, states, mint=nothing, maxt=nothing; goal=nothing, fps=1,
    title="Generated trajectory from POMDP",
    saveas=nothing,
    minweight=0.1
)
    poses = [
        L.Pose(position, PARAMS.orientation)
        for position in GenPOMDPs.state_sequence(tr)
    ];
    ptclouds = [
        obss[:obs].*pose
        for (obss, pose) in zip(
            GenPOMDPs.observation_sequence(tr),
            poses
        )
    ];

    if isnothing(mint)
        mint = 0
    end
    if isnothing(maxt)
        maxt = min(T(tr), length(states)-1)
    end
    
    # -----------------------
    col = palette(:default)
    ani = Animation()
    # states = [state]
    for t=mint:maxt
        p = poses[t + 1]
        y = ptclouds[t + 1]
    
        agent_plt = plot(
            size=(500,500), aspect_ratio=:equal, title=title, grid=false,
            xlim=(_bb[1][1]-3, _bb[2][1]+3),
            ylim=(_bb[1][2]-3, _bb[2][2]+3),
            legend=:bottomleft,
        )
        plot!(_segs, c=:black, linewidth=1, label=nothing)
#         plot!([p], c=:red, r=1.0, linewidth=2, label=nothing)
        scatter!(Random.shuffle(y[1:2:end]), c=col[1], markersize=4, alpha=.7, markerstrokewidth=1, label="Obs in pose coords `_ys[t].*_ps[t]`")
    
        particles = states[t + 1]
        labeled=false
        for (w, tr) in zip(
                GenParticleFilters.get_norm_weights(particles),
                GenParticleFilters.get_traces(particles)
            )
            pos = tr[GenPOMDPs.state_addr(t)]
            if !labeled && sqrt(w) > 0.1
                scatter!([pos], c=:black, seriesalpha=max(sqrt(w), minweight), label="Particle")
                labeled=true
            else
                scatter!([pos], c=:black, seriesalpha=max(sqrt(w), minweight), label=nothing)
            end
        end
        
        if !isnothing(goal)
           scatter!([goal], c=:green, marker=:x, label="Goal") 
        end
        
        scatter!([p.x], c=:red, marker=:x, label="True agent position")
    
        annotate!(_bb[1][1]-1, _bb[2][2]+1, text("t=$t", :black, :right, 12))

        frame(ani, agent_plt)
    end
    if !isnothing(saveas)
        return gif(ani, "$saveas.gif", fps=fps) # gif(ani, fname, fps=10)
    else
        return gif(ani, fps=fps)
    end
end


################
### Policies ###
###############

const state_addr = GenPOMDPs.state_addr


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
function find_action_using_grid_search(start_linecoords, goal_linecoords)
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

    length(results.path) == 1 && return (:stay, results.path)
    
    next_state = results.path[2]
    # println("next_state: ", g_to_l(next_state...))
    action = actions[findfirst(
        GW.newpos(w, initialpos, dir) == next_state for dir in actions
    )]

    return (action, results.path)
end

"""
st  - finite state machine state (direction of wall we're following)
nbs - neighbors of the agent on grid which are filled (set of dirs)
returns (action, next_st)

dirs = [:L, :R, :U, :D]
"""
function _wall_follow(st, nbs)
    isnothing(st) && :L ∉ nbs ? (:L, st) :
    isnothing(st) && :L ∈ nbs ? (:D, :L) :
    #
    st == :L && :L ∉ nbs ? (:L, :U) :
    st == :L && :D ∉ nbs ? (:D, :L) :
    st == :L && :R ∉ nbs ? (:R, :D) :
    st == :L             ? (:U, :R) :
    #
    st == :R && :R ∉ nbs ? (:R, :D) :
    st == :R && :U ∉ nbs ? (:U, :R) :
    st == :R && :L ∉ nbs ? (:L, :U) :
    st == :R             ? (:D, :L) :
    #
    st == :D && :D ∉ nbs ? (:D, :L) :
    st == :D && :R ∉ nbs ? (:R, :D) :
    st == :D && :U ∉ nbs ? (:U, :R) :
    st == :D             ? (:L, :U) :
    #
    st == :U && :U ∉ nbs ? (:U, :R) :
    st == :U && :L ∉ nbs ? (:L, :U) :
    st == :U && :D ∉ nbs ? (:D, :L) :
    st == :U             ? (:R, :D) :
    
    error("Unrecognized st/nbs pair.")
end

initial_wall_follow_state() = Any[nothing]
function wall_follow_from_pos(pos, st)
    (x, y) = pos

    # Find which sides of the agent have walls
    # (the agent's "neighbors")
    nbs = Set()
    δ = 1.25*ϵ
    for (a, newpos) in (
        (:U, [x, y + δ]), (:D, [x, y - δ]), (:L, [x - δ, y]), (:R, [x + δ, y])
    )
        if handle_wall_intersection([x, y], newpos, PARAMS.map) != newpos
            push!(nbs, a)
        end
    end
    
    (a, st) = _wall_follow(st, nbs)
    
    act = (;U=:up,L=:left,D=:down,R=:right)[a]
    
    return (act, st)

end
function wall_follow(sts, pos)
    (a, st) = wall_follow_from_pos(pos, sts[end])
    sts = vcat(sts, [st])
    
    # Prevent cycles
    if length(sts)>8 && sts[end-3:end] == sts[end-7:end-4] && length(Set(sts[end-3:end])) == 4
        sts[end] = nothing
    end
    
    return (a, sts)
end

function there_is_ambiguity(pf_state, threshold=5e-5)
    tr1 = (GPF.sample_unweighted_traces(pf_state, 1)[1])
    pos = currentpos(tr1)
    
    p_different = 0.
    for (tr, p_particle) in zip(GPF.get_traces(pf_state), GPF.get_norm_weights(pf_state))
        if Geo.norm(currentpos(tr) - pos) > .5
            p_different += p_particle
        end
    end
    
    return p_different > threshold
end