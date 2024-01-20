import LineWorlds
L = LineWorlds
using GenSMCP3
import GenTraceKernelDSL
mvuniform = L.ProductDistribution(uniform);

GenSMCP3.@kernel function fwd_iterated_grid(
    tr, is_initial_step, trajectory_model, params, obs, _grid_args,
    pos_to_init_cm, state_to_pos, new_action, get_det_next_pos,
    grid_saver=nothing
)
    if is_initial_step
        T = 0
        pos = _grid_args.initial_pos
    else
        T_prev, _ = get_args(tr)
        T = T_prev + 1

        # pos = state_to_pos(GenTraceKernelDSL.get_trace(tr)[GenPOMDPs.state_addr(T - 1)]) #
        pos = get_det_next_pos(state_to_pos(GenTraceKernelDSL.get_trace(tr)[GenPOMDPs.state_addr(T - 1)]), new_action, params.step.Δ, params.map)
    end
    j, grid_args, grid_proposed_positions = nothing, nothing, nothing # make these available outside loop scope
    n_iters = _grid_args.n_iters + (!isnothing(_grid_args.special_init_grid_args) ? 1 : 0)
    for i = 1:n_iters #_grid_args.n_iters
        if !isnothing(_grid_args.special_init_grid_args)
            if i == 1
                grid_args = _grid_args.special_init_grid_args
            else
                grid_args = L.grid_schedule(_grid_args.init_grid_args, _grid_args.scaling_factor, i - 1)
            end
        else
            grid_args = L.grid_schedule(_grid_args.init_grid_args, _grid_args.scaling_factor, i)
        end
        # grid_args = L.grid_schedule(_grid_args.init_grid_args, _grid_args.scaling_factor, i)
        grid_proposed_positions, _ = L.vector_grid(GenSMCP3.GenTraceKernelDSL.DynamicForwardDiff.value(pos), grid_args...)
        grid_final_positions = reshape(grid_proposed_positions, (:,))

        if !isnothing(grid_saver)
            grid_saver(grid_final_positions)
        end

        # Evaluate the observation probabilities.
        poses = [L.Pose(p, params.obs.orientation) for p in grid_final_positions]
        # if T == 1
        #     println(grid_final_positions)
        # end

        _segs = [L.Segment(r...) for r in eachrow(GridWorlds.wall_segments(params.map))]
        as = L.create_angles(params.obs.fov, params.obs.n_rays)
        obspts = [dist * [cos(a), sin(a)] for (a, dist) in zip(as, obs[:obs])]
        # obspts = collect([x, y] for (x, y) in zip(GridWorlds.points_from_raytracing_continuous(0, 0, obs[:obs])...))
        log_obs_probs,  = L.eval_pose_vectors(
            Vector.(poses)::Array{Vector{Float64}},
            (obspts)::Vector{Vector{Float64}},
            _segs::Vector{L.Segment},
            params.obs.wall_sensor_args; sorted=false,
            fov=params.obs.fov, num_a=params.obs.n_rays
        )
        # println(log_obs_probs)
        # println(poses)

        # Sample a next position from the grid.
        log_ps = log_obs_probs
        log_ps_tau = log_ps ./ _grid_args.tau
        probs      = L.normalize_exp(log_ps_tau)
        probs      = L.raise_probs(probs, _grid_args.pmin)
        # println(probs)
        j = {(:j, i)} ~ categorical(probs)
        pos = grid_final_positions[j]        
    end
    vs = grid_proposed_positions
    # if T == 1
    #     println((vs[j][1:2] -  grid_args.r[1:2]/2, vs[j][1:2] + grid_args.r[1:2]/2))
    # end
    x′  = {:pos}  ~ mvuniform(vs[j][1:2] -  grid_args.r[1:2]/2, vs[j][1:2] + grid_args.r[1:2]/2)
    # print("Proposing position: ", x′, "\n")
    return (
        GenPOMDPs.nest_choicemap(pos_to_init_cm(x′), GenPOMDPs.state_addr(T)),
        EmptyChoiceMap() # Backward proposal is deterministic.
    )
end
GenSMCP3.@kernel function bwd_iterated_grid(tr, is_initial_step, _grid_args, state_to_pos, get_det_next_pos)
    (T, actions, params) = get_args(tr)
    new_action = T > 0 ? actions[T] : nothing
    
    fwd_ch = choicemap()
    fwd_ch[:pos] = state_to_pos(tr[GenPOMDPs.state_addr(T)])

    if is_initial_step
        init_pos = _grid_args.initial_pos
    else
        # init_pos = state_to_pos(tr[GenPOMDPs.state_addr(T - 1)]) # 
        init_pos = get_det_next_pos(state_to_pos(tr[GenPOMDPs.state_addr(T - 1)]), new_action, params.step.Δ, params.map)
    end

    for i=_grid_args.n_iters:1
        grid_args = L.grid_schedule(_grid_args.init_grid_args, _grid_args.scaling_factor, i)
        j = L.grid_index(nexpos, init_pos, grid_args...; linear=true)
        fwd_ch[(:j, i)] = j
    end

    return (EmptyChoiceMap(), fwd_ch)
end


###

GenSMCP3.@kernel function fwd_iterated_grid_init(
    tr, trajectory_model, params, obs, t0_grid_args,
    pos_to_init_cm
)
    pos = t0_grid_args.initial_pos
    j, grid_args, grid_proposed_positions = nothing, nothing, nothing # make these available outside loop scope
    for i = 1:t0_grid_args.n_iters
        grid_args = L.grid_schedule(t0_grid_args.init_grid_args, i)
        grid_proposed_positions, _ = L.vector_grid(GenSMCP3.GenTraceKernelDSL.DynamicForwardDiff.value(pos), grid_args...)
        grid_final_positions = reshape(grid_proposed_positions, (:,))

        pos_choicemap_grid = [pos_to_init_cm(pos) for pos in grid_final_positions]
        full_cm_grid = [
            Gen.merge(
                GenPOMDPs.nest_choicemap(pos_choicemap_grid[j], GenPOMDPs.state_addr(0)),
                GenPOMDPs.nest_choicemap(obs, GenPOMDPs.obs_addr(0))
            )
            for j in 1:length(grid_final_positions)
        ]
        wts = [Gen.generate(trajectory_model, (0, [], params), cm)[2] for cm in full_cm_grid]
        wts = L.normalize_exp(wts)
        j = {(:j, i)} ~ categorical(wts)
        pos = grid_final_positions[j]
    end
    vs = grid_proposed_positions
    x′  = {:pos}  ~ mvuniform(vs[j][1:2] -  grid_args.r[1:2]/2, vs[j][1:2] + grid_args.r[1:2]/2)

    return (
        GenPOMDPs.nest_choicemap(pos_to_init_cm(x′), GenPOMDPs.state_addr(0)),
        EmptyChoiceMap() # Backward proposal is deterministic.
    )
end
GenSMCP3.@kernel function bwd_iterated_grid_init(tr, t0_grid_args, state_to_pos)
    fwd_ch = choicemap()
    fwd_ch[:pos] = state_to_pos(tr[GenPOMDPs.state_addr(0)])

    pos = t0_grid_args.initial_pos
    for i=t0_grid_args.n_iters:1
        args = L.grid_schedule(t0_grid_args.init_grid_args, i)
        j = L.grid_index(nexpos, pos, args...; linear=true)
        fwd_ch[(:j, i)] = j
    end

    return (EmptyChoiceMap(), fwd_ch)
end

function default_pf_args(PARAMS; n_particles=1,
    coarsest_stepsize=0.4, # at the coarsest level of the grid, this is the grid spacing
    sigma_multiplier=3 # number of times the motion model sigma to use for the grid range
)
    # Do some math to figure out the bounding box center,
    # and a grid schedule which will overlay roughly 400 grid points
    # over the whole grid.  We will use
    # this grid schedule for the coarsest level of the coarse-to-fine
    # proposals at the initialization step.
    avg(x, y) = (x+y)/2
    _bb = ((0, 0), Tuple(PARAMS.map.size))
    center = [avg(_bb[1][1], _bb[2][1]), avg(_bb[1][2], _bb[2][2])]
    Δy = _bb[2][2] - _bb[1][2]
    Δx = _bb[2][1] - _bb[1][1]
    nstepsy = Int(ceil(Δy/coarsest_stepsize))
    nstepsx = Int(ceil(Δx/coarsest_stepsize));
    # y_to_x = Δy/Δx
    # step = sqrt(Δx^2 * y_to_x / 400)
    # @assert Δy/step * Δx/step ≈ 400
    # nstepsy = Int(floor(Δy/step))
    # nstepsx = Int(floor(Δx/step));
    # @assert nstepsy*nstepsx<500

    target_fineness = params.obs.wall_sensor_args.s_noise / 4
    n_steps_c2f = Int(ceil(log(2/3, target_fineness/coarsest_stepsize)))

    stepsize = coarsest_stepsize
    # go out to 3 sigma in each direction, for each dimension
    range = PARAMS.step.σ * sigma_multiplier * 2
    gridsize_init = Int(ceil(range / stepsize))
    update_grid_args = (;
        tau = 1., pmin = 1e-6, n_iters = n_steps_c2f,
        special_init_grid_args=nothing,
        init_grid_args = (; k = [gridsize_init, gridsize_init], r = [stepsize, stepsize]),
        scaling_factor=(2/3) # each iteration, the grid size is scaled down by this factor
    )

    # 6 iters of coarse to fine, since we have to start at a much
    # coarser level to scan over the whole environment (not just the regions
    # where the agent could have stepped in the last timestep)
    t0_grid_args = (;
        update_grid_args...,
        special_init_grid_args = (; k=[nstepsx, nstepsy], r = [stepsize, stepsize]),
        initial_pos = center,
        scaling_factor=(2/3) # each iteration, the grid size is scaled down by this factor
    )

    resampling_args = (; n_particles, n_groups=1, ess_threshold=0.)

    return (t0_grid_args, update_grid_args, resampling_args)
end

### GFI-based grid enumeration code [slower than the vectorized version] ###

# pos_choicemap_grid = [pos_to_init_cm(pos) for pos in grid_final_positions]
# full_cm_grid = [
#     Gen.merge(
#         GenPOMDPs.nest_choicemap(pos_choicemap_grid[j], GenPOMDPs.state_addr(T)),
#         GenPOMDPs.nest_choicemap(obs, GenPOMDPs.obs_addr(T))
#     )
#     for j in 1:length(grid_final_positions)
# ]
# if is_initial_step
#     wts = [Gen.generate(trajectory_model, (0, [], params), cm)[2] for cm in full_cm_grid]
# else
#     wts = [Gen.update(
#         GenTraceKernelDSL.get_trace(tr),
#         (T, [get_args(tr)[2]..., new_action], params),
#         (UnknownChange(), NoChange(), NoChange()),
#         cm
#     )[2] for cm in full_cm_grid]
# end
# wts = L.normalize_exp(wts)
# j = {(:j, i)} ~ categorical(wts)
# pos = grid_final_positions[j]
