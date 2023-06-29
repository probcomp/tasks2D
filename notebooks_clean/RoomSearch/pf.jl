import GenParticleFilters
const GPF = GenParticleFilters
using GenSMCP3

### Main particle filter constructors ###
"""
Top-level function for constructing a particle filter for localization based on grid coarse-to-fine.
"""
# (This is a macro rather than a function due to silly reasons which I can fix once I update
# some dependencies.  Main issue is that loading Gen static generative functions requires
# evaluation at the top-level scope.  This is fixed in the next version of Gen, but it isn't
# available as an official release [yet].)
# Tip for reading this code if you're not a Julia pro:
# read `$(esc(X))` simply as `X`; the reason we need the $(esc(...))
# is due to Julia's [often annoying] macro hygeine rules.
macro get_localization_pf(pomdp, PARAMS, _grid_args, _t0_grid_args)
    quote
    # Call GenPOMDP's built-in method for constructing particle filters for POMDP environments.
    GenPOMDPs.pf(
        $(esc(pomdp)), $(esc(PARAMS)),

        ### Specify how to initialize the particle filter. ###
        obs -> ( # Initialize via a 3-particle SMCP3 proposal.
            1, # n particles
            SMCP3Update(
                forward_iterated_grid_proposal, backward_iterated_grid_proposal,
                # is_initial_step, params, new_action, new_obs, grid_args
                (true, $(esc(PARAMS)), nothing, obs[:obs], $(esc(_t0_grid_args)), choicemap((:_, false))), # no action before the initial observation
                # is_initial_step, grid_args
                (true, $(esc(_t0_grid_args)),)
            )
        ),

        ### Specify the particle filter update. ###
        (action, obs) -> ( # Update with the SMCP3 proposal.
            SMCP3Update(
                forward_iterated_grid_proposal, backward_iterated_grid_proposal,
                # is_initial_step, params, new_action, new_obs, grid_args
                (false, $(esc(PARAMS)), action, obs[:obs], $(esc(_grid_args)), choicemap((:_, false))),
                # is_initial_step, grid_args
                (false, $(esc(_grid_args)),)
            ),
        );

        pre_update=(_ -> ()), # no pre-update step
    );
    end # quote
end

macro get_stratified_pf_update(pomdp, PARAMS, _grid_args, _t0_grid_args)
    quote
    function update(pf_state, action, obs, t)
        addr = GenPOMDPs.state_addr(t, :extended_placements => :object_is_in_room)
        updater = GenPOMDPs.pf_updater(
            (action, obs) -> ( # Stratified update with the SMCP3 proposal.

                # stratified update, with 1 particle having the object in the room and one not
                ( choicemap((addr, true)), choicemap((addr, false)) ),
                
                # use SMCP3 to propose the rest
                SMCP3Update(
                    forward_iterated_grid_proposal, backward_iterated_grid_proposal,
                    # is_initial_step, params, new_action, new_obs, grid_args
                    (false, $(esc(PARAMS)), action, obs[:obs], $(esc(_grid_args))),
                    # is_initial_step, grid_args
                    (false, $(esc(_grid_args)),)
                ),
            )
        )
        
        new_pf_state = updater(pf_state, action, obs)
        @assert get_args(new_pf_state.traces[1])[1] > get_args(pf_state.traces[1])[1] "old t = $(get_args(pf_state.traces[1])[1]), new t = $(get_args(new_pf_state.traces[1])[1])"
        return new_pf_state
    end
    end # quote
end

### Below here: SMCP3 proposals, and default gridding arguments. ###

T(tr) = get_args(tr)[1]
currentpos(tr) = tr[state_addr(T(tr), :pos)]

# SMCP3 forward proposal
GenSMCP3.@kernel function forward_iterated_grid_proposal(tr, is_initial_step, params, new_action, new_obs, grid_args,
    constraint_on_room_assignment
)
    if is_initial_step
        T = 0
        pos = grid_args.initial_pos
    else
        T_prev, _, params = get_args(tr)
        # @assert params_ == params # we have to pass in `params` since we can't get it otherwise for the initial step
        T = T_prev + 1

        # Form the grid around what would be the position of the agent at the next timestep
        # in the determinized version of the motion model.
        pos = det_next_pos(tr[state_addr(T_prev, :pos)], new_action, params.step.Δ)

        obj_placements = tr[state_addr(T_prev)][2]
    end

    did_room_placement = Gen.to_array(constraint_on_room_assignment, Any)[1]

    if did_room_placement
        _segs = vcat(params.map, segments_for_object_in(room_containing(pos, params), params.target_pose_relative_to_room))
    elseif !is_initial_step && target_object_placed(obj_placements)
        _segs = vcat(params.map, segments_for_object_in(get_target_room(obj_placements), params.target_pose_relative_to_room))
    else
        _segs = params.map
    end

    j, args, grid_proposed_positions = nothing, nothing, nothing # make these available outside loop scope
    for i = 1:grid_args.n_iters
        args = L.grid_schedule(grid_args.init_grid_args, i)
        
        grid_proposed_positions, _ = L.vector_grid(GenSMCP3.GenTraceKernelDSL.DynamicForwardDiff.value(pos), args...)
        if !is_initial_step
            grid_final_positions = reshape(grid_proposed_positions, (:,))
        else
            grid_final_positions = grid_proposed_positions
        end

        # Evaluate the observation probabilities.
        poses = [L.Pose(p, params.obs.orientation) for p in grid_final_positions]
        log_obs_probs,  = L.eval_pose_vectors(
            Vector.(poses)::Array{Vector{Float64}},
            new_obs::Vector{Vector{Float64}},
            _segs::Vector{L.Segment},
            params.obs.sensor_args...; sorted=false,
            fov=params.obs.fov, num_a=params.obs.n_rays
        )

        # Sample a next position from the grid.
        log_ps = log_obs_probs
        log_ps_tau = log_ps ./ grid_args.tau
        probs      = L.normalize_exp(log_ps_tau)
        probs      = L.raise_probs(probs, grid_args.pmin)
        j = {(:j, i)} ~ categorical(probs)
        pos = grid_final_positions[j]
    end

    # Sample the final position in continuous space.
    vs = grid_proposed_positions
    x′  = {:pos}  ~ mvuniform(vs[j][1:2] -  args.r[1:2]/2, vs[j][1:2] + args.r[1:2]/2)

    if did_room_placement && room_containing(x′, params) != room_containing(pos, params)
        # Then we placed the object in the wrong room!
        @warn "Proposal proceeded with object placed in wrong room."
    end

    cm = choicemap((state_addr(T, :pos), x′))
    addr = state_addr(T, :extended_placements => :object_is_in_room)
    room = room_containing(x′, params)
    if !is_initial_step && !haskey(obj_placements, room) && room in params.viable_rooms && !has_value(constraint_on_room_assignment, addr)
        # println("adding room placement constraint at time $T")
        cm[addr] = did_room_placement
    end

    # println("setting $((state_addr(T, :pos))) to $x′")
    return (
        cm,
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


############## Default Args #####################

function default_pf_args(PARAMS; n_particles)
    # Do some math to figure out the bounding box center,
    # and a grid schedule which will overlay roughly 400 grid points
    # over the whole grid.  We will use
    # this grid schedule for the coarsest level of the coarse-to-fine
    # proposals at the initialization step.
    avg(x, y) = (x+y)/2
    _bb = PARAMS.bounding_box
    center = [avg(_bb[1][1], _bb[2][1]), avg(_bb[1][2], _bb[2][2])]
    Δy = _bb[2][2] - _bb[1][2]
    Δx = _bb[2][1] - _bb[1][1]
    y_to_x = Δy/Δx
    step = sqrt(Δx^2 * y_to_x / 400)
    @assert Δy/step * Δx/step ≈ 400
    nstepsy = Int(floor(Δy/step))
    nstepsx = Int(floor(Δx/step));
    @assert nstepsy*nstepsx<500

    # 3 iters of coarse-to-fine
    update_grid_args = (;
        tau = 1., pmin = 1e-6, n_iters = 3,
        init_grid_args = (; k = [9, 9], r = [PARAMS.step.Δ/6, PARAMS.step.Δ/6]),
    )

    # 6 iters of coarse to fine, since we have to start at a much
    # coarser level to scan over the whole environment (not just the regions
    # where the agent could have stepped in the last timestep)
    t0_grid_args = (;
        update_grid_args..., tau=1., n_iters=6,
        init_grid_args = (; k=[nstepsx, nstepsy], r = [step, step]), initial_pos = center
    )

    return (update_grid_args, t0_grid_args)
end

function overwrite_params(params; p_kidnapped, step, sensor_args)
    return (;
        params...,
        p_kidnapped,
        step = (;
            params.step...,
            step...
        ),
        obs = (;
            params.obs...,
            sensor_args = (;
                params.obs.sensor_args...,
                sensor_args...
            )
        )
    )
end