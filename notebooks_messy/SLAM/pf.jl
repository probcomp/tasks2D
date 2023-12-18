import GenPOMDPs
import GenParticleFilters
import Gen

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
