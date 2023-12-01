"""
This file contains code for raytracing in a grid world.
Code for actual raycasting is in the LineWorlds sub-library.
This file just contains util code for interfacing with the line-based
raycasting code.

Note that there are two coordinate systems in play: discrete grid coordinates
and continuous coordinates.
The discrete grid coordinate (x, y) corresponds to a square of continuous coordinates
with bottom left corner (x, y) and top right corner (x+1, y+1).
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
    pose = [x + .5; y + .5; 0.] # Pose from the center of the agent
    poses = reshape(pose, (1, 3))
    distances = line_raycaster(poses, wall_segments(w); num_a=n_rays) 
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
    angles = LinRange(π, 3π, num_a)
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