"""
Tasks2D.Utils
"""
module Utils

using Gen
import Serialization
import Dates

export serialize_pomdp_trace, deserialize_pomdp_trace

### Top-level serialization methods ###

function serialize_trace_and_viz_actions(filename, trace; args_to_serializeable=identity, viz_actions=nothing)
    Serialization.serialize(filename, Dict(
        "trace" => pomdp_trace_to_serializable(trace; args_to_serializeable=args_to_serializeable),
        "viz_actions" => viz_actions
    ))
    println("Trace & viz actions serialized to $filename.")
end
function deserialize_trace_and_viz_actions(filename, pomdp_trajectory_model; args_from_serializeable=identity)
    s = Serialization.deserialize(filename)
    tr = pomdp_trace_from_serializable(s["trace"], pomdp_trajectory_model; args_from_serializeable=args_from_serializeable)
    return tr, s["viz_actions"]
end

function serialize_trace_and_pf_states(filename, trace, pf_states; metadata=Dict())
    Serialization.serialize(
        filename,
        Dict(
            "trace" => pomdp_trace_to_serializable(trace),
            "pf_states" => [pf_state_to_serializable(pf_state) for pf_state in pf_states],
            "metadata" => metadata
        )
    )
    println("Trace & pf states serialized to $filename.")
end
function deserialize_trace_and_pf_states(filename, pomdp_trajectory_model)
    s = Serialization.deserialize(filename)
    tr = pomdp_trace_from_serializable(s["trace"], pomdp_trajectory_model)
    pf_states = [pf_state_from_serializable(pf_state, pomdp_trajectory_model) for pf_state in s["pf_states"]]
    return tr, pf_states
end

function serialize_pomdp_trace(filename, trace; args_to_serializeable = identity)
    Serialization.serialize(filename, pomdp_trace_to_serializable(trace; args_to_serializeable=args_to_serializeable))
    println("Trace serialized to $filename.")
end
function deserialize_pomdp_trace(filename, pomdp_trajectory_model; args_from_serializeable = identity)
    s = Serialization.deserialize(filename)
    return pomdp_trace_from_serializable(s, pomdp_trajectory_model; args_from_serializeable=args_from_serializeable)
end

### Conversion to/from serializable ###

function pomdp_trace_to_serializable(trace; args_to_serializeable = identity)
    args = args_to_serializeable(get_args(trace))
    dict = Dict(
        "args" => args,
        "choices" => choicemap_to_serializable(get_choices(trace))
    )
    return dict
end
function pomdp_trace_from_serializable(s, pomdp_trajectory_model; args_from_serializeable = identity)
    args = args_from_serializeable(s["args"])
    choices = serializable_to_choicemap(s["choices"])
    tr, _ = Gen.generate(pomdp_trajectory_model, args, choices)
    return tr
end

function pf_state_to_serializable(pf_state; args_to_serializeable = identity)
    dict = Dict(
        "parents" => pf_state.parents,
        "log_ml_est" => pf_state.log_ml_est,
        "log_weights" => pf_state.log_weights,
        "traces" => [
            Dict(
                "args" => args_to_serializeable(get_args(trace)),
                "choices" => choicemap_to_serializable(get_choices(trace))
            )
            for trace in pf_state.traces
        ]
    )
    return dict
end
function pf_state_from_serializable(s, pomdp_trajectory_model; args_from_serializeable = identity)
    traces = [
        Gen.generate(pomdp_trajectory_model, args_from_serializeable(trace["args"]), serializable_to_choicemap(trace["choices"]))[1]
        for trace in s["traces"]
    ]
    return Gen.ParticleFilterState(
        traces,
        copy(traces),
        s["log_weights"],
        s["log_ml_est"],
        s["parents"],
    )
end

function choicemap_to_serializable(cm)
    address_to_value = collect(get_values_shallow(cm))
    address_to_sub = [
        (addr, choicemap_to_serializable(submap))
        for (addr, submap) in get_submaps_shallow(cm)
    ]
    return (address_to_value, address_to_sub)
end
function serializable_to_choicemap(s)
    (a_to_v, a_to_s) = s
    cm = choicemap()
    for (a, v) in a_to_v
        cm[a] = v
    end
    for (a, s) in a_to_s
        Gen.set_submap!(cm, a, serializable_to_choicemap(s))
    end
    return cm
end

### Dists ###
struct MappedUniform <: Gen.Distribution{Any}; end
Gen.random(::MappedUniform, mins, maxs) = [Gen.uniform(min, max) for (min, max) in zip(mins, maxs)]
function Gen.logpdf(::MappedUniform, v, mins, maxs)
    if length(v) != length(mins) || length(v) != length(maxs)
        return -Inf
    end
    return sum(logpdf(Gen.uniform, val, min, max) for (val, min, max) in zip(v, mins, maxs))
end
mapped_uniform = MappedUniform()
(::MappedUniform)(args...) = random(mapped_uniform, args...)

end # module