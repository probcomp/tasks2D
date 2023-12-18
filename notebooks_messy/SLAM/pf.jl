import GenPOMDPs
import GenParticleFilters
import Gen
import GenSMCP3
import GenSMCP3: SMCP3Update

function pf(pomdp, params, pf_initialize_params, pf_update_params, init_constraints::ChoiceMap=choicemap(), pre_update=(_ -> ()),
    post_update=(_ -> ()))
    return (
        pf_initializer(pomdp, params, pf_initialize_params, init_constraints),
        GenPOMDPs.pf_updater(pf_update_params...; pre_update, post_update)
    )
end

# override default pf to allow for additional constraints
function pf_initializer(pomdp::GenPOMDPs.GenPOMDP, pomdp_params, pf_params, init_constraints::ChoiceMap)
    controlled_trajectory_model = GenPOMDPs.ControlledTrajectoryModel(pomdp)

    function initialize(obs0)
        constraints = choicemap()
        Gen.set_submap!(constraints, GenPOMDPs.state_addr(0), init_constraints)
        Gen.set_submap!(constraints, GenPOMDPs.obs_addr(0), obs0)

        pf_state = GenParticleFilters.pf_initialize(
            controlled_trajectory_model,
            (0, [], pomdp_params),
            constraints,
            pf_params...
        )
        return pf_state
    end

    return initialize
end


function get_pf(pomdp, t_to_params, num_particles, initial_constraints=choicemap())
    # Call GenPOMDP's built-in method for constructing particle filters for POMDP environments.
    GenPOMDPs.pf(
        pomdp, t_to_params,

        ### Specify how to initialize the particle filter. ###
        obs -> ( # Initialize via a 3-particle SMCP3 proposal.
            num_particles,
            GenSMCP3.SMCP3Update(
                _forward_proposal, _backward_proposal,
                # is_initial_step, t_to_params, new_action, new_obs
                (true, t_to_params, nothing, obs[:obs], initial_constraints), # no action before the initial observation
                # is_initial_step
                (true,)
            )
        ),

        ### Specify the particle filter update. ###
        (action, obs) -> ( # Update with the SMCP3 proposal.
            GenSMCP3.SMCP3Update(
                _forward_proposal, _backward_proposal,
                # is_initial_step, t_to_params, new_action, new_obs
                (false, t_to_params, action, obs[:obs]),
                # is_initial_step
                (false,)
            ),
        );

        # ### Specify the rejuvenation rule. [This argument to GenPOMDPs.pf can also be used to add MCMC rejuvenation, etc.] ###
        # pre_update=state -> resampling_rule(state, $(esc(_resampling_args))),
    )
end

# SMCP3 forward proposal
GenSMCP3.@kernel function _forward_proposal(prev_trace, is_initial_step, t_to_params, new_action, new_obs, init_constraints=choicemap())
    if is_initial_step
        t = 0
        # sample initial position using prior distribution
        trace, _ = generate(uniform_agent_pos, (T_TO_PARAMS_INTRO,), init_constraints)
    else
        prev_t, _ = get_args(prev_trace)
        t = prev_t + 1

        last_state = prev_trace[GenPOMDPs.state_addr(prev_t)]
        # sample initial position using prior distribution
        trace, _ = generate(motion_model, (last_state, new_action, T_TO_PARAMS_INTRO,))
    end
    change_chp = choicemap(get_choices(trace))

    # redraw map where ray hit wall
    state = get_retval(trace)
    world = state.world
    obs_points = GridWorlds.points_from_raytracing(state.pos..., new_obs; is_continuous=true)
    for (_x, _y) in obs_points
        x_op = state.pos[1] > _x ? ceil : floor
        y_op = state.pos[2] > _y ? ceil : floor
        x = clamp(x_op(Int, _x), 1, world.size[1])
        y = clamp(y_op(Int, _y), 1, world.size[2])
        world = GridWorlds.replace(world, (x, y), GridWorlds.wall)
    end

    change_chp[:world] = world

    return (
        GenPOMDPs.nest_choicemap(change_chp, GenPOMDPs.state_addr(t)),
        EmptyChoiceMap() # Backward proposal is deterministic.
    )
end

# SMCP3 backward proposal
GenSMCP3.@kernel function _backward_proposal(trace, is_initial_step)
    (t, actions, t_to_params) = get_args(trace)
    new_action = t > 0 ? actions[t] : nothing
    t_prev = t - 1

    if is_initial_step
        var_names = [:world, :cell, :x, :y]
    else
        var_names = [:world, :noisy_next_pos]
    end


    fwd_ch = choicemap()
    for field in var_names
        fwd_ch[field] = trace[GenPOMDPs.state_addr(t, field)]
    end

    return (
        EmptyChoiceMap(), # The old trace just drops values; no constraints needed.
        fwd_ch
    )
end