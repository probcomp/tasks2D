######################################################################
# Adapted from https://github.com/probcomp/probabilistic-slam-in-gen #
######################################################################

##################################### 
module Geometry  
#####################################

using Colors, Plots
col = palette(:default);

using ..LineWorldUtils
import ..LineWorldUtils: stack
using LinearAlgebra

mutable struct Segment
    x::Vector{Float64}
    y::Vector{Float64}
end
Segment(x1::Float64,x2::Float64,y1::Float64,y2::Float64) = Segment([x1;x2],[y1;y2])
Segment(xs::Vector{Float64}) = Segment(xs[1:2],xs[3:4])

Base.Vector(s::Segment) = [s.x;s.y]
vec(s::Segment)  = [s.x;s.y]
diff(s::Segment) = s.y - s.x
stack(segs::Vector{Segment}) = reduce(vcat, transpose.(vec.(segs)))
LinearAlgebra.norm(s::Segment) = norm(s.y - s.x)

function segments(xs::Vector{Vector{Float64}})
    segs = [Segment(x, y) for (x,y) in zip(xs[1:end-1],xs[2:end])]
    push!(segs, Segment(xs[end], xs[1]))
    return segs
end

function distance(x::Vector{Float64}, s::Segment)
    x  = x - s.x
    v  = diff(s)
    nv = [-v[2];v[1]]
    if s.x == s.y
        return norm(x - s.x)
    end
    nv = nv/norm(nv)

    d = Inf
    if dot(x, v) < 0
        d = norm(x)
    elseif dot(x, v) <= dot(v,v)
        d = abs(dot(x, nv))
    else
        d = norm(x - v)
    end
    return d
end
distance(s::Segment, x::Vector{Float64}) = dist(x,s);
distance(x::Vector{Float64}, segs::Vector{Segment}) = minimum(distance.([x], segs))
function bounding_box(segs::Vector{Segment})
    S = stack(segs)
    min_x = minimum(S[:,[1,3]])
    max_x = maximum(S[:,[1,3]])
    min_y = minimum(S[:,[2,4]])
    max_y = maximum(S[:,[2,4]])
    return ([min_x;min_y],[max_x;max_y])
end

export Segment, segments, bounding_box, distance, vec

function Plots.plot!(s::Segment; args...)
    plot!([s.x[1],s.y[1]], [s.x[2],s.y[2]]; args...)
end

function Plots.plot!(segs::Vector{Segment}; label=nothing, args...)
    myplot = nothing
    for (i,s) in enumerate(segs)
        if i > 1
            label = nothing
        end
        myplot = plot!(s; label=label, args...)
    end
    return myplot
end

function line_intersect(x, x′, y, y′)
    dx = x′ - x
    dy = y′ - y
    if det([-dx dy]) == 0
        return [Inf;Inf]
    end
    s, t = inv([-dx dy])*(x - y)
    return s,t
end;

function cast(ray::Segment, seg::Segment)
    x,x′ = ray.x, ray.y
    y,y′ = seg.x, seg.y
    dx = x′ - x
    s,t = line_intersect(x, x′, y, y′)
    if s == Inf
       return false, nothing
    elseif 0 <= s && 0 <= t <= 1
        return true, s*norm(dx)
    else
        return false, nothing
    end
end

function cast(x::Vector{Float64}, hd::Float64, seg::Segment)
    ray = Segment(x,x+[cos(hd);sin(hd)])
    return cast(ray, seg)
end

function cast(x::Vector{Float64}, hd::Float64, a::Vector{Float64}, segs::Vector{Segment}; zmax=100)
    z = ones(length(a))*zmax
    for i in 1:length(a), s in segs
        hit, d = cast(x, a[i] + hd, s)
        if hit
            z[i] = min(z[i], d)
        end
    end
    return z
end

export cast

#####################################
end  
#####################################
