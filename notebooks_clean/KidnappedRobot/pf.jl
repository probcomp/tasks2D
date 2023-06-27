#=
This file implements a particle filter which uses a coarse-to-fine proposal
both at the initialization step, and each update step.

This is implemented using SMCP3.
=#

import GenParticleFilters
const GPF = GenParticleFilters
using GenSMCP3

"""
Top-level function for constructing a particle filter based on grid coarse-to-fine.
"""
# (This is a macro rather than a function due to silly reasons which I can fix once I update
# some dependencies.)
# Reading tip: read `$(esc(X))` simply as `X`; the reason we need the $(esc(...))
# is a distractor.
macro get_pf(pomdp, PARAMS, _grid_args, _t0_grid_args, _resampling_args)
    quote
    # Call GenPOMDP's built-in method for constructing particle filters for POMDP environments.
    GenPOMDPs.pf(
        $(esc(pomdp)), $(esc(PARAMS)),

        ### Specify how to initialize the particle filter. ###
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

        ### Specify the particle filter update. ###
        (action, obs) -> ( # Update with the SMCP3 proposal.
            SMCP3Update(
                forward_iterated_grid_proposal, backward_iterated_grid_proposal,
                # is_initial_step, params, new_action, new_obs, grid_args
                (false, $(esc(PARAMS)), action, obs[:obs], $(esc(_grid_args))),
                # is_initial_step, grid_args
                (false, $(esc(_grid_args)),)
            ),
        );

        ### Specify the rejuvenation rule. [This argument to GenPOMDPs.pf can also be used to add MCMC rejuvenation, etc.] ###
        pre_update = state -> resampling_rule(state, $(esc(_resampling_args))),
    );
    end # quote
end

#=
Below here are some details of the grid proposal implementation, and the resampling rule.

The resampling rule is not actually used since all the particle filters are 1-particle filters.
This resampling rule is just legacy from some of my older experiments.
=#

T(tr) = get_args(tr)[1]
currentpos(tr) = tr[state_addr(T(tr), :pos)]

# SMCP3 forward proposal
GenSMCP3.@kernel function forward_iterated_grid_proposal(tr, is_initial_step, params, new_action, new_obs, grid_args)
    if is_initial_step
        T = 0
        pos = grid_args.initial_pos
    else
        T_prev, _ = get_args(tr)
        # @assert params_ == params # we have to pass in `params` since we can't get it otherwise for the initial step
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
            grid_final_positions = reshape(grid_proposed_positions, (:,))
        else
            grid_final_positions = grid_proposed_positions
        end

        # Evaluate the observation probabilities.
        poses = [L.Pose(p, params.obs.orientation) for p in grid_final_positions]
        _segs = params.map
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