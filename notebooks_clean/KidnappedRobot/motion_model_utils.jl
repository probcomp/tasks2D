# Due an inconvenient element of the implementation of the Gen SMCP3 inference library,
# which deals with Dualed values for automatic differentiation,
# we will need to unpack Dualed values in the below functions.
using GenSMCP3
const DFD = GenSMCP3.GenTraceKernelDSL.DynamicForwardDiff

function det_next_pos(pos, a, Δ)
    (x, y) = DFD.value(pos) # Unpack a potentially Dualed value
    a == :up    ? [x, y + Δ] :
    a == :down  ? [x, y - Δ] : 
    a == :left  ? [x - Δ, y] :
    a == :right ? [x + Δ, y] :
    a == :stay  ? [x, y]     :
                error("Unrecognized action: $a")
end
function handle_wall_intersection(prev, new, walls)
    move = L.Segment(DFD.value(prev), DFD.value(new))
    for wall in walls
        do_intersect, dist = L.Geometry.cast(move, wall)
        if do_intersect && dist ≤ L.Geometry.norm(move)
            return prev
        end
    end
    return new
end