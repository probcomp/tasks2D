function intersections(x, x′, y, y′)
    n = ndims(x)
    dx = x′ .- x
    dy = y′ .- y
    v  = x .- y

    dx1 = selectdim(dx, n, 1)
    dx2 = selectdim(dx, n, 2)
    dy1 = selectdim(dy, n, 1)
    dy2 = selectdim(dy, n, 2)
    v1 = selectdim(v, n, 1)
    v2 = selectdim(v, n, 2)

    a, b = -dx1, dy1
    c, d = -dx2, dy2

    det = a.*d .- b.*c
    
    s = 1 ./det .*(  d.*v1 .- b.*v2)
    t = 1 ./det .*(- c.*v1 .+ a.*v2)
    return s,t, x .+ s.*dx, y .+ t.*dy
end

"""
    zs, is = cast(v0, as::Vector, segs::Array; zmax=Inf)

Returns depth measurements and segment-IDs for a given 
pose vector `v0` and a set of segments `segs` and angles `as`.
(The segment IDs are indices in the `segs` array.)

v0 = [xcoord, ycoord, heading]
as = [angle1, angle2, ...]
segs = [x1 y1 x2 y2; x1 y1 x2 y2; ...] # each row is a segment -- TODO CHECK THIS
"""
function cast(v0::Vector, as::Vector, segs::Array; zmax=Inf)
    as_   = (as .+ v0[3])
    segs_ = (segs)

    x_  = v0[1:2]
    x′_ = cat(v0[1] .+ cos.(as_), v0[2] .+ sin.(as_), dims=2)
    y_  = view(segs_,:,1:2)
    y′_ = view(segs_,:,3:4)

    x_  = reshape(x_ , :, 1, 2)
    x′_ = reshape(x′_, :, 1, 2)
    y_  = reshape(y_ , 1, :, 2)
    y′_ = reshape(y′_, 1, :, 2)

    s_, t_, _, _ = intersections(x_, x′_, y_, y′_)

    # Hit map
    h_ = (0 .< s_) .* (0 .<= t_ .<= 1)
    s_[.!h_] .= zmax

    # Segment-ID
    i_  = argmin(s_, dims=2)[:,1]
    i2_ = map(i->i[2],i_);
    
    # Todo: There was an issue when I computed `z_ = s_[i_]`
    #       Not sure why... but using minimum() works
    z_ = minimum(s_, dims=2)[:,1]
    i2_[z_.==zmax] .= 0

    # Depth and Segment-ID
    return z_, i2_
end;

cast(p::Pose, as::Vector, segs::Array; zmax=Inf) = cast(Vector(p), as, segs; zmax=zmax);

create_angles(fov, num_a) = [range(-fov/2, fov/2, num_a)...];

function create_observations(ps::Vector{Pose}, segs::Vector, fov::Real, num_a::Integer, obs_noise=0.)
    as = create_angles(fov, num_a)
    dists = create_observations(ps, segs, as, obs_noise)
    ys = [polar_inv(dists[i], as) for i in keys(dists)]
    return (dists, as, ys)
end
function create_observations(ps::Vector{Pose}, segs::Vector, angles::Vector, obs_noise=0.0)
    segs = stack(Vector.(segs))

    return [
        cast(p, angles, segs, zmax=Inf)[1] .+ randn(length(angles))*obs_noise
        for p in ps
    ]
end

###########
### GPU ###
###########
# using Metal

# function cast_GPU(v0::AbstractVector, as::AbstractVector, segs::AbstractArray; zmax=Float32(Inf))
#     as_   = MtlVector(as .+ v0[3])
#     segs_ = MtlArray(segs)

#     x_  = MtlVector(v0[1:2])
#     x′_ = cat(v0[1] .+ cos.(as_), v0[2] .+ sin.(as_), dims=2)
#     y_  = view(segs_,:,1:2)
#     y′_ = view(segs_,:,3:4)

#     x_  = reshape(x_ , :, 1, 2)
#     x′_ = reshape(x′_, :, 1, 2)
#     y_  = reshape(y_ , 1, :, 2)
#     y′_ = reshape(y′_, 1, :, 2)

#     s_, t_, _, _ = intersections(x_, x′_, y_, y′_)

#     # Hit map
#     h_ = (0 .< s_) .* (0 .<= t_ .<= 1)
#     s_ = s_ .* h_ .+ zmax .* (.!h_)
#     # s_[.!h_] .= zmax
#     # @. s_[(!((0 .< s_) .* (0 .<= t_ .<= 1)))] = zmax

#     # # Segment-ID
#     i_  = argmin(s_, dims=2)[:,1]
#     i2_ = map(i->i[2],i_);
    
#     # # Todo: There was an issue when I computed `z_ = s_[i_]`
#     # #       Not sure why... but using minimum() works
#     z_ = minimum(s_, dims=2)[:,1]
#     # i2_[z_.==zmax] .= 0
#     i2_ = i2_ .* (z_.!=zmax)

#     # # Depth and Segment-ID
#     return z_, i2_
#     # return s_, t_
# end;

# cast_GPU(p::Pose, as::Vector, segs::Array; zmax=Inf) = cast_GPU(Vector(p), as, segs; zmax=zmax);

# function create_observations_GPU(ps::Vector{Pose}, segs::Vector, fov::Real, num_a::Integer, obs_noise=0.)
#     as = create_angles(fov, num_a)
#     dists = create_observations_GPU(ps, segs, as, obs_noise)
#     ys = [polar_inv(dists[i], as) for i in keys(dists)]
#     return (dists, as, ys)
# end
# function create_observations_GPU(ps::Vector{Pose}, segs::Vector, angles::Vector, obs_noise=0.0)
#     segs = stack(Vector.(segs))

#     return [
#         Vector(cast_GPU(p, angles, segs, zmax=Inf)[1] .+ randn(length(angles))*obs_noise)
#         for p in ps
#     ]
# end