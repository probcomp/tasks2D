module VizUtils

import Makie
import GenPOMDPs
import ..Utils

function get_posobs_seq(groundtruth_trace)
    return map(
        trace -> (
            [p for (p, t, h) in GenPOMDPs.state_sequence(trace)],
            [reshape(o, (:,)) for o in GenPOMDPs.observation_retval_sequence(trace)]
        ),
        groundtruth_trace
    )
end
function get_obs_seq(groundtruth_trace)
    return map(
        trace -> [reshape(o, (:,)) for o in GenPOMDPs.observation_retval_sequence(trace)],
        groundtruth_trace
    )
end

using Dates

function get_save_tr(tr)
    function save_tr(viz_actions)
        filename = "saves/" * string(now()) * "__pomdp_trace.jld"
        Utils.serialize_trace_and_viz_actions(filename, tr[];
            viz_actions=viz_actions,
            # args_to_serializeable = args -> (args[1:2]..., args[3].params)
        )
    end
    return save_tr
end

function get_take_action(_take_action)
    function take_action(a)
        _take_action((a, Dates.now()))
    end
    return take_action
end
function get_interactive_trace(args...; kwargs...)
    (trace, _take_action) = GenPOMDPs.interactive_world_trace(args...; kwargs...)
    return (trace, get_take_action(_take_action))
end
function make_trace_interactive(args...; kwargs...)
    (trace, _take_action) = GenPOMDPs.make_trace_interactive(args...; kwargs...)
    return (trace, get_take_action(_take_action))
end

function get_did_hitwall_observable(trace)
    return map(trace -> GenPOMDPs.state_sequence(trace)[end][3], trace)
end
function close_window(f)
    glfw_window = GLMakie.to_native(display(f))
    GLMakie.GLFW.SetWindowShouldClose(glfw_window, true)
end

function get_action_times_observable(trace)
    return map(trace -> [t for (a, t) in GenPOMDPs.action_sequence(trace)], trace)
end
function get_timing_args(trace; speedup_factor=1, max_delay=5) # 5 seconds max delay
    return (get_action_times_observable(trace), speedup_factor, max_delay)
end

struct ConstantTToParams
    params
end
(p::ConstantTToParams)(t) = p.params

struct SwitchTToParams
    params1
    params2
    switch
end
(p::SwitchTToParams)(t::Makie.Observable) = p(t[])
(p::SwitchTToParams)(t) = p.switch(t) ? p.params1 : p.params2

struct TimedSwitchTToParams
    params1
    params2
    timewindows_params1
    timewindows_params2
end
(p::TimedSwitchTToParams)(t::Makie.Observable) = p(t[])
function (p::TimedSwitchTToParams)(t)
    for (start, stop) in p.timewindows_params1
        if start ≤ t ≤ stop
            return p.params1
        end
    end
    for (start, stop) in p.timewindows_params2
        if start ≤ t ≤ stop
            return p.params2
        end
    end
    error("No time window found for time $t")
end

end # module