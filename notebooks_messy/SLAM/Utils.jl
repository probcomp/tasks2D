"""
Tasks2D.Utils
"""
module Utils

using Gen
import Serialization
import Dates

export serialize_pomdp_trace, deserialize_pomdp_trace

function serialize_pomdp_trace(filename, trace)
    Serialization.serialize(filename, Dict(
        "args" => get_args(trace),
        "choices" => choicemap_to_serializable(get_choices(trace))
    ))
    println("Trace serialized to $filename.")
end
function deserialize_pomdp_trace(filename, pomdp_trajectory_model)
    s = Serialization.deserialize(filename)
    args = s["args"]
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

end # module