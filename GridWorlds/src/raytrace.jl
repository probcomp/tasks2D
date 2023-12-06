"""
This file contains code for raytracing in a grid world.
Code for actual raycasting is in the LineWorlds sub-library.
This file just contains util code for interfacing with the line-based
raycasting code.

Note that there are two coordinate systems in play: discrete grid coordinates
and continuous coordinates.
The discrete grid coordinate (x, y) corresponds to a square of continuous coordinates
with bottom left corner (x - 1, y - 1) and top right corner (x, y).
"""

"""
Compute the distances to the walls via raytracing.

Args:
- line_raycaster: line-based raycasting function.  (Typically, `cast` from the `LineWorlds` library.)
  This is a function that takes in (poses, segments; num_a) and returns
  a vector of distances to the walls along each ray. poses is (k, 3); segments is (n, 4).
  num_a is the number of angles (number of rays to cast).
- w - gridworld
- n_rays = number of rays to cast
"""
# TODO: memoize this for performance
function ray_trace_distances(line_raycaster, w::GridWorld, n_rays)
    (x, y) = agentpos(w)
    pose = [x - 0.5; y - 0.5; 0.] # Pose from the center of the agent
    poses = reshape(pose, (1, 3))
    distances = line_raycaster(poses, wall_segments(w); num_a=n_rays) 
    return reshape(distances, (:,))
end

"""
Given the distances to the walls from raytracing, compute the
points in continuous space where the rays hit the walls.

- ax, ay are the coordinates of the agent in continuous space
- distances are the distances to the walls along each ray (at
 evenly spaced angles from π to 3π)
"""
function points_from_raytracing_continuous(ax, ay, distances; angles=nothing)
    if isnothing(angles)
        num_a = length(distances)
        angles = LinRange(-π/2, 3π/2, num_a)
    end

    xs = distances .* cos.(angles)
    ys = distances .* sin.(angles)
    return xs .+ ax, ys .+ ay
end

"""
Here, ax and ay are discrete coordinates for a grid cell the agent is at the center of.
"""
function points_from_raytracing_discrete(ax, ay, distances; kwargs...)
    return points_from_raytracing_continuous(ax .- 0.5, ay .- 0.5, distances; kwargs...)
end

function points_from_raytracing(ax, ay, distances; is_continuous, kwargs...)
    if is_continuous
        points_from_raytracing_continuous(ax, ay, distances; kwargs...)
    else
        points_from_raytracing_discrete(ax, ay, distances; kwargs...)
    end
end

"""
Line segments for the walls of a grid world.
"""
function wall_segments(w)
    segments = Vector{Float64}[]
    for x=1:size(w)[1]
        for y=1:size(w)[2]
            if w[x, y] == wall
                push!(segments, [x - 1, y - 1, x, y - 1])
                push!(segments, [x - 1, y - 1, x - 1, y])
                push!(segments, [x - 1, y, x, y])
                push!(segments, [x, y - 1, x, y])
            end
        end
    end
    return Base.stack(segments) |> transpose
end