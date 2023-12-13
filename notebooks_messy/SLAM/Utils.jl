"""
Tasks2D.Utils
"""
module Utils

using Gen
import Serialization
import Dates

export serialize_pomdp_trace, deserialize_pomdp_trace

function serialize_trace_and_viz_actions(filename, trace; args_to_serializeable=identity, viz_actions=nothing)
    args = args_to_serializeable(get_args(trace))
    Serialization.serialize(filename, Dict(
        "args" => args,
        "choices" => choicemap_to_serializable(get_choices(trace)),
        "viz_actions" => viz_actions
    ))
    println("Trace & viz actions serialized to $filename.")
end
function deserialize_trace_and_viz_actions(filename, pomdp_trajectory_model; args_from_serializeable=identity)
    s = Serialization.deserialize(filename)
    args = args_from_serializeable(s["args"])
    choices = serializable_to_choicemap(s["choices"])
    viz_actions = s["viz_actions"]
    tr, _ = Gen.generate(pomdp_trajectory_model, args, choices)
    return tr, viz_actions
end

function serialize_pomdp_trace(filename, trace; args_to_serializeable = identity)
    args = args_to_serializeable(get_args(trace))
    Serialization.serialize(filename, Dict(
        "args" => args,
        "choices" => choicemap_to_serializable(get_choices(trace))
    ))
    println("Trace serialized to $filename.")
end
function deserialize_pomdp_trace(filename, pomdp_trajectory_model; args_from_serializeable = identity)
    s = Serialization.deserialize(filename)
    args = args_from_serializeable(s["args"])
    choices = serializable_to_choicemap(s["choices"])
    tr, _ = Gen.generate(pomdp_trajectory_model, args, choices)
    return tr
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