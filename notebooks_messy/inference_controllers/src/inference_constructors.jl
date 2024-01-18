include("inference2.jl")

function make_bpf(pomdp, params, n_particles)
    trajectory_model = GenPOMDPs.ControlledTrajectoryModel(pomdp)
    (t0_grid_args, update_grid_args, _) = default_pf_args(params, coarsest_stepsize=0.2)
    return GenPOMDPs.pf(
        pomdp, params,
        obs -> (
            n_particles,
            GenSMCP3.SMCP3Update(
                fwd_iterated_grid, bwd_iterated_grid,
                (true, trajectory_model, params, obs, t0_grid_args, pos_to_init_cm, state_to_pos, nothing, nonoise_nextpos),
                (true, t0_grid_args, state_to_pos, nonoise_nextpos)
            )
        ),
        (action, obs) -> (), # bootstrap update,
        pre_update = GenPOMDPs.stratified_resample_if_ess_below_one_plus_onetenth_particlecount
    )
end
"""
Make a coase-to-fine particle filter, with parms.step.σ * sigma_multiplier
as the total spatial range of the coarsest grid, and coarsest_stepsize as
the step size of the coarsest grid.
"""
function make_c2f_pf(pomdp, params, n_particles; sigma_multiplier, coarsest_stepsize=0.2)
    trajectory_model = GenPOMDPs.ControlledTrajectoryModel(pomdp)
    (t0_grid_args, update_grid_args, _) = default_pf_args(params; sigma_multiplier, coarsest_stepsize)

    return GenPOMDPs.pf(
        pomdp, params,
        obs -> (
            n_particles,
            GenSMCP3.SMCP3Update(
                fwd_iterated_grid, bwd_iterated_grid,
                (
                    true, trajectory_model, params, obs, t0_grid_args,
                    pos_to_init_cm, state_to_pos, nothing, nonoise_nextpos
                ),
                (true, t0_grid_args, state_to_pos, nonoise_nextpos)
            )
        ),
        (action, obs) -> (
            SMCP3Update(
                fwd_iterated_grid, bwd_iterated_grid,
                (
                    false, trajectory_model, params, obs, update_grid_args,
                    pos_to_step_cm, state_to_pos, action, nonoise_nextpos
                ),
                (false, update_grid_args, state_to_pos, nonoise_nextpos)
            
            ),
        ),
        pre_update = GenPOMDPs.stratified_resample_if_ess_below_one_plus_onetenth_particlecount
    );
end

function make_pf(pomdp, params, spec)
    """
    Accepts a spec of one of the following forms:
    - (; type=:bootstrap, n_particles)
    - (; type=:c2f, n_particles, sigma_multiplier [, coarsest_stepsize])
    """
    if spec.type == :bootstrap
        return make_bpf(pomdp, params, spec.n_particles)
    else
        if haskey(spec, :coarsest_stepsize)
            return make_c2f_pf(pomdp, params, spec.n_particles; sigma_multiplier=spec.sigma_multiplier, coarsest_stepsize=spec.coarsest_stepsize)
        else
            return make_c2f_pf(pomdp, params, spec.n_particles; sigma_multiplier=spec.sigma_multiplier)
        end
    end
end