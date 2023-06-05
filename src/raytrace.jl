#=
Note that there are two coordinate systems in play: discrete grid coordinates
and continuous coordinates.
The discrete grid coordinate (x, y) corresponds to a square of continuous coordinates
with bottom left corner (x, y) and top right corner (x+1, y+1).
=#

# Raytracing off line segments, in continuous coordinates.
include("_raytrace_lowlevel.jl")

"""
Compute the distances to the walls via raytracing.
"""
# TODO: memoize this for performance
function ray_trace_distances(w::GridWorld, n_rays)
    (x, y) = agentpos(w)
    pose = [x + .5; y + .5; 0.] # Pose from the center of the agent
    poses = reshape(pose, (1, 3))
    distances = cast_cpu(poses, wall_segments(w); num_a=n_rays) 
    return reshape(distances, (:,))
end

"""
Given the distances to the walls from raytracing, compute the
points in continuous space where the rays hit the walls.

- ax, ay are the coordinates of the agent in discrete grid space
  (so the agent fills the region from ax-1 to ax and ay-1 to ay
   in continuous space)
- distances are the distances to the walls along each ray (at
 evenly spaced angles from π to 3π)
"""
function points_from_raytracing(ax, ay, distances)
    num_a = length(distances)
    angles = LinRange(π, 3π, num_a + 1)[1:end-1]
    xs = distances .* cos.(angles)
    ys = distances .* sin.(angles)
    return xs .+ ax .- .5, ys .+ ay .- .5
end

"""
Line segments for the walls of a grid world.
"""
function wall_segments(w)
    segments = Vector{Float64}[]
    for x=1:size(w)[1]
        for y=1:size(w)[2]
            if w[x, y] == wall
                push!(segments, [x, y, x + 1, y])
                push!(segments, [x, y, x, y + 1])
                push!(segments, [x, y + 1, x + 1, y + 1])
                push!(segments, [x + 1, y, x + 1, y + 1])
            end
        end
    end
    return Base.stack(segments) |> transpose
end