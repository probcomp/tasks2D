import Pkg
Pkg.activate("../../Tasks2D/")

using Revise
includet("src/model2.jl")
# Loads in: model_init, motion_model, observe_noisy_distances

include("src/action_controllers.jl");
# Loads in: meandering_wallavoiding_controller

import LinearAlgebra

# POMDP for an agent moving around in the environment
pomdp = GenPOMDPs.GenPOMDP(
    model_init,              # INIT   : params                     ⇝ state
    motion_model,            # STEP   : prev_state, action, params ⇝ state
    observe_noisy_distances, #| OBS    : state, params              ⇝ observation
    (state, action) -> 0.    # UTILITY: state, action, params      → utility
);

rollout_model = GenPOMDPs.RolloutModel(pomdp, meandering_wallavoiding_controller);

_get_params(map) = (;
    # map = GridWorlds.load_custom_map(5),
    map=map,
    step = (; Δ = .5, σ_windy = 0.5, σ_normal = 0.05),
    obs = (; fov = 2π, n_rays = 90, orientation = π/2,
        # I think currently only σ is used
        wall_sensor_args = (;
            w = 5, s_noise = 0.02,
            outlier = 0.0001,
            outlier_vol = 100.0,
            zmax = 100.0, σ=0.005
        )
    ),
    init_wind_prob = 0.02,
    become_windy_prob = 0.02,
    stay_windy_prob = 0.9
)

MAPSTR = """
wwwwwwwwwwwwwwwww
w               w
w        w      w
w        w      w
w        w      w
w        w      w
w               w
w               w
w               w
wwwwwwwwwwwwwwwww
"""
params = _get_params(GridWorlds.mapstr_to_gridworld(MAPSTR));

trace = Gen.simulate(rollout_model, (100, params))
GenPOMDPs.state_sequence(trace)[1:10]

@gen function ipomdp_init(params)
    state ~ model_init(params.params)
    return state
end
@gen function ipomdp_step(prev, inference_action, params)
    action = params.actions[prev.t + 1]
    state ~ motion_model(prev, action, params.params)
    return state
end
@gen function ipomdp_obs(prev, params)
    observation ~ observe_noisy_distances(prev, params.params)
    return observation
end

"""
    InferenceAction(found_position:Bool, inferred_position::Vector, inference_runtime::Float64)
- found_position - did the inference algorithm find a position, or give up?
- inferred_position - the position that the inference algorithm inferred, if any
- inference_runtime is the time it took to run the inference algorithm, in milliseconds.
"""
InferenceAction = NamedTuple{
    (:found_position, :inferred_position, :inference_runtime),
    Tuple{Bool, Vector{Float64}, Float64}
}
function errcost(pos, ia::InferenceAction)
    if ia.found_position
        return LinearAlgebra.norm(pos - ia.inferred_position)^2
    else
        return 1.
    end
end
function timecost(ia::InferenceAction)
    # Scale inference runtime by 2000, so it is better to spend
    # 2 seconds and get the position right than to totally give up.
    return ia.inference_runtime/2000
end
function cost(pos, ia::InferenceAction)
    return errcost(pos, ia) + timecost(ia)
end

inference_pomdp = GenPOMDPs.GenPOMDP(
    ipomdp_init,
    ipomdp_step,
    ipomdp_obs,
    # Utility = -cost
    (state, inference_action) -> -cost(state.pos, inference_action)
);

function is_err(pos, ia)
    return LinearAlgebra.norm(ia.inferred_position - pos) > 0.3
end
function n_errors(itrace::Gen.Trace)
    return sum(
        is_err(state.pos, ia) for (state, ia) in zip(
            GenPOMDPs.state_sequence(itrace),
            GenPOMDPs.action_sequence(itrace)
        )
    )
end
function errcost(itrace::Gen.Trace)
    return sum(
        errcost(state.pos, ia) for (state, ia) in zip(
            GenPOMDPs.state_sequence(itrace),
            GenPOMDPs.action_sequence(itrace)
        )
    )

end
function timecost(itrace::Gen.Trace)
    return sum(
        timecost(ia) for ia in GenPOMDPs.action_sequence(itrace)
    )
end

function pomdp_to_ipomdp_choicemap(pomdp_choicemap, T)
    cm = choicemap()
    for t=0:T
        Gen.set_submap!(cm, GenPOMDPs.state_addr(t, :state),
            get_submap(pomdp_choicemap, GenPOMDPs.state_addr(t))
        )
        Gen.set_submap!(cm, GenPOMDPs.obs_addr(t, :observation),
            get_submap(pomdp_choicemap, GenPOMDPs.obs_addr(t))
        )
    end
    return cm
end

ictm = GenPOMDPs.ControlledTrajectoryModel(inference_pomdp)
ias = [InferenceAction((true, x.pos, 0.)) for x in GenPOMDPs.state_sequence(trace)]
ipomdp_params = (params=params, actions=GenPOMDPs.action_sequence(trace))
itrace, _ = GenPOMDPs.generate(
    ictm, (100, ias, ipomdp_params),
    GenPOMDPs.nest_choicemap(
        get_submap(get_choices(trace), GenPOMDPs.state_addr(0)),
        GenPOMDPs.nest_at(GenPOMDPs.state_addr(0), :state)
    )
)
GenPOMDPs.undiscounted_utility(inference_pomdp, itrace)

function get_timed_inference_controller(_inference_controller)
    @gen function timed_inference_controller(cstate, observation)
        t = time()
        new_cstate, inference_action = {:inner} ~ _inference_controller(cstate, observation)
        Δt = time() - t
        
        return InferenceAction((inference_action.found_position, inference_action.inferred_position, Δt)), new_cstate
    end
    return timed_inference_controller
end

include("src/inference_constructors.jl")

FixedPFControllerState = NamedTuple{
    (:pf, :ipomdp_params, :pfstate),
    Tuple{Any, Any, Any}
}
# Main inference controller function
@gen function __fixed_pf_inference_controller(cstate, observation)
    pf_init, pf_update = cstate.pf
    observation = choicemap((:obs, observation))
    if isnothing(cstate.pfstate)
        # Initialize the PF
        new_pf_state = pf_init(observation)
    else
        # Update the PF
        t = GenPOMDPs.state_sequence(cstate.pfstate.traces[1])[end].t + 1
        action = cstate.ipomdp_params.actions[t]
        new_pf_state = pf_update(cstate.pfstate, action, observation)
    end

    tr = Gen.sample_unweighted_traces(new_pf_state, 1)[1]
    inference_action = InferenceAction((true, GenPOMDPs.state_sequence(tr)[end].pos, NaN))
    new_cstate = FixedPFControllerState((cstate.pf, cstate.ipomdp_params, new_pf_state))
    return new_cstate, inference_action
end
_fixed_pf_inference_controller = get_timed_inference_controller(__fixed_pf_inference_controller)

include("src/inference_constructors.jl")
function FixedC2FInferenceController(sigma_multiplier, coarsest_stepsize)
    return GenPOMDPs.Controller(
        _fixed_pf_inference_controller,
        FixedPFControllerState((
            make_c2f_pf(pomdp, params, 2; sigma_multiplier, coarsest_stepsize),
            ipomdp_params, nothing)
        )
    )
end
fixed_c2f_ic = FixedC2FInferenceController(4, 0.2)
fixed_c2f_rollout_model = GenPOMDPs.RolloutModel(inference_pomdp, fixed_c2f_ic)
c2f_itrace, _ = GenPOMDPs.generate(
    fixed_c2f_rollout_model, (10, ipomdp_params),
    pomdp_to_ipomdp_choicemap(get_choices(trace), 10)
);

ICParams = NamedTuple{
    (:pf_init_sequence, :pf_update_sequence, :threshold)
}
AdaptiveInferenceControllerState = NamedTuple{
    (:ic_params, :ipomdp_params, :pfstate),
    Tuple{Any, Any, Any}
}
function fails_threshold_test(pfstate, threshold)
    tr = Gen.sample_unweighted_traces(pfstate, 1)[1]
    score = Gen.project(tr, select(GenPOMDPs.obs_addr(get_args(tr)[1])))
    return score < threshold
end
@gen function __adaptive_pf_init(ic_params, obs)
    i = 1
    pfstate = ic_params.pf_init_sequence[i](obs)
    while fails_threshold_test(pfstate, ic_params.threshold) && i < length(ic_params.pf_init_sequence)
        i += 1
        pfstate = ic_params.pf_init_sequence[i](obs)
    end
    return pfstate
end
@gen function __adaptive_pf_update(ic_params, prev_pfstate, action, obs)
    i = 1
    pfstate = ic_params.pf_update_sequence[i](prev_pfstate, action, obs)
    while fails_threshold_test(pfstate, ic_params.threshold) && i < length(ic_params.pf_update_sequence)
        i += 1
        pfstate = ic_params.pf_update_sequence[i](prev_pfstate, action, obs)
    end
    return pfstate
end
@gen function __adaptive_inference_controller(cstate, observation)
    observation = choicemap((:obs, observation))
    if isnothing(cstate.pfstate)
        new_pf_state = __adaptive_pf_init(cstate.ic_params, observation)
    else
        # Update the PF
        t = GenPOMDPs.state_sequence(cstate.pfstate.traces[1])[end].t + 1
        action = cstate.ipomdp_params.actions[t]
        new_pf_state = __adaptive_pf_update(cstate.ic_params, cstate.pfstate, action, observation)
    end

    tr = Gen.sample_unweighted_traces(new_pf_state, 1)[1]
    inference_action = InferenceAction((true, GenPOMDPs.state_sequence(tr)[end].pos, NaN))
    new_cstate = AdaptiveInferenceControllerState((cstate.ic_params, cstate.ipomdp_params, new_pf_state))
    return new_cstate, inference_action
end
_adaptive_inference_controller = get_timed_inference_controller(__adaptive_inference_controller)

# Generate data

n_steps_calibration = 100
n_calibration_trs = 1000
calibration_trs = [Gen.simulate(rollout_model, (n_steps_calibration, params)) for _ in 1:n_calibration_trs];
obs_log_likelihoods = [
    Gen.project(tr, select(GenPOMDPs.obs_addr(step)))
    for tr in calibration_trs, step in 1:n_steps_calibration
];
get_percentile_threshold(percentile) = sort(collect(Iterators.flatten(obs_log_likelihoods)))[Int(length(obs_log_likelihoods) * percentile/100)]
threshold_1_in_10000 = get_percentile_threshold(0.01)
threshold_1_in_1000 = get_percentile_threshold(0.1)
threshold_1_in_100 = get_percentile_threshold(1.0);

pf_init_sequence = [
    make_c2f_pf(pomdp, params, 1; sigma_multiplier=m, coarsest_stepsize=r)[1]
    for (m, r) in [(1., 0.4), (1, 0.2), (1, 0.1)]
]
pf_update_sequence = [
    make_c2f_pf(pomdp, params, 1; sigma_multiplier=m, coarsest_stepsize=r)[2]
    for (m, r) in [(0.5, 0.4), (1.5, 0.4), (2.5, 0.2), (4., 0.2), (5., 0.1)]

]
adaptive_inference_controller = GenPOMDPs.Controller(
    _adaptive_inference_controller,
    AdaptiveInferenceControllerState((
        ICParams((pf_init_sequence, pf_update_sequence, threshold_1_in_1000)),
        ipomdp_params, nothing)
    )
);

###
# These 3 parameters will control how long it takes to generate
# the data here.
sigma_multipliers = [0.5, 2., 4.] # [0.5, 1., 2., 4.] # [0.5, 1., 2., 4.] #[0.5, 1., 2., 3., 4., 5.]
n_traces = 20 #8
trace_len = 100

ic_groups = [
    ("Adaptive Inference Controller", [adaptive_inference_controller]),
    ("Fixed Schedule Inference; Coarsest Grid Step Size = 0.2", [
        FixedC2FInferenceController(s, 0.2) for s in sigma_multipliers
    ]),
    ("Fixed Schedule Inference; Coarsest Grid Step Size = 0.4", [
        FixedC2FInferenceController(s, 0.4) for s in sigma_multipliers
    ])
]
ipomdp_rollout_model_groups = [
    (label, [GenPOMDPs.RolloutModel(inference_pomdp, ic) for ic in ics])
    for (label, ics) in ic_groups
]

traces = [Gen.simulate(rollout_model, (trace_len, params)) for _=1:n_traces]
# ipomdp_trace_groups = [
#     [
#         [
#             begin
#                 println("On group $k, irm $j, trace $i")
#                 GenPOMDPs.generate(irm, (trace_len, ipomdp_params),
#                     pomdp_to_ipomdp_choicemap(get_choices(tr), trace_len)
#                 )[1]
#             end
#             for (i, tr) in enumerate(traces)
#         ]
#         for (j, irm) in enumerate(irms)
#     ]
#     for (k, (label, irms)) in enumerate(ipomdp_rollout_model_groups)
# ];

ipomdp_trace_groups = [[Any[nothing for _=1:n_traces] for _=1:length(irms)] for (_, irms) in ipomdp_rollout_model_groups]
Threads.@threads for i in 1:length(traces)
    tr = traces[i]
    for (k, (label, irms)) in enumerate(ipomdp_rollout_model_groups)
        for (j, irm) in enumerate(irms)
            println("Trace $i -- group $k, irm $j")
            ipomdp_trace_groups[k][j][i] = GenPOMDPs.generate(irm, (trace_len, ipomdp_params),
                pomdp_to_ipomdp_choicemap(get_choices(tr), trace_len)
            )[1]
        end
    end
end

mean(v) = sum(v)/length(v)
mean_runtimes_per_group = [
    [
        mean([a.inference_runtime for tr in trs for a in GenPOMDPs.action_sequence(tr)])
        for trs in group_trs
    ]
    for group_trs in ipomdp_trace_groups
]

errors_per_thousand_per_group = [
    [
        sum([n_errors(tr) for tr in trs]) #/ (trace_len * length(group_trs)) * 1000
        for trs in group_trs
    ]
    for group_trs in ipomdp_trace_groups
]

display(repr(mean_runtime_per_group))
display(repr(errors_per_thousand_per_group))