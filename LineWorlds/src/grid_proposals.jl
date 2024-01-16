function sortperm_them!(vals, vecs...)
    perm = sortperm(vals)
    id   = 1:length(vals)
    for v in [vals, vecs...]
        v[id] = v[perm]
    end
end;

argdiffs(bs::Array{T,1}) where T <: Real = Tuple(map(b -> Bool(b) ? UnknownChange() : NoChange(), bs));

"""
Discretize into bins of diameter r, bin-centers lie 
at `z - k*r` for intergers `k`.
"""
quantize(x, r; zero=0) = Int.(floor.((x .+ r./2 .- zero)./r))

"""
    get_offset(v0, k, r)

Computes the offset to move the center 
of the grid to `v0`.
"""
function get_offset(v0, k, r)
    center = (r + k.*r)/2
    return v0 - center
end

function first_grid_vec(v0::Vector{Real}, k::Vector{Int}, r::Vector{Real})
    return r + get_offset(v0, k, r) 
end

"""
    vs, ls = vector_grid(v0, k, r)

Returns grid of vectors and their linear indices, given 
a grid center, numnber of grid points along each dimension and
the resolution along each dimension.
"""
function vector_grid(v0::Vector{Float64}, k::Vector{Int}, r::Vector{Float64})
    # Todo: Does it make sense to get a CUDA version of this?
    offset = get_offset(v0, k, r)
    
    shape = Tuple(k)
    cs = CartesianIndices(shape)
    ls = LinearIndices(shape)
    vs = map(I -> [Tuple(I)...].*r + offset, cs);
    return (vs=vs, linear_indices=ls)
end


function grid_index(x, v0, k, r; linear=false)
    I = quantize(x, r, zero=get_offset(v0, k, r));
    if linear
        if any(map(x -> x == 0, I))
            println("I = $I")
        end
        try
            shape = Tuple(k)
            I = LinearIndices(shape)[I...]
        catch e
            println("I = $I ; k = $k")
            error(e)
        end
    end
    return I
end

"""
    log_ps, ps = eval_pose_vectors(
                    vs::Array{Vector{Float64}},
                    x::Vector{Vector{Float64}}, 
                    segs::Vector{Segment},
                    w::Int, s_noise::Float64, outlier::Float64, 
                    outlier_vol::Float64=1.0, zmax::Float64=100.0; sorted=false)

Evaluates a collection of poses 
with respect to different Gaussian mixtures...
"""
function eval_pose_vectors(
            vs::Array{Vector{Float64}},
            x::Vector{Vector{Float64}}, 
            segs,

            # sensor args
            w::Int,  s_noise::Float64, outlier::Float64, outlier_vol::Float64=1.0, zmax::Float64=100.0;
            
            sorted=false, fov, num_a
)
    as = create_angles(fov, num_a)
    
    # Compute sensor measurements and 
    # Gaussian mixture components
    # p_  = CuArray(Vector(p))
    # ps_ = reshape(p_, 1, 3)

    ps   = stack(vs[:])
    x    = stack(x)
    segs = stack(Vector.(segs))

    if _cuda[]
        ps   = CuArray(ps)
        x    = CuArray(x)
        segs = CuArray(segs)
        as   = CuArray(as)
    end
    

    zs = cast(ps, segs; fov, num_a, zmax=zmax)
    ys_tilde = get_2d_mixture_components(zs, as, w)
        
    # Evaluate the the observations with respect to the 
    # different Gaussian mixtures computed above
    log_ps, = sensor_logpdf(x, ys_tilde, s_noise, outlier, outlier_vol; return_pointwise=false);
    
    # Move everyting back to CPU if is not already there
    log_ps = Array(log_ps)

    # Sort by log prob
    # and return 
    if sorted
        perm   = sortperm(log_ps)
        log_ps = log_ps[perm]
        vs     = vs[:][perm]
    end
    
    return log_ps, vs[:]
end;
function eval_pose_vectors(
    vs::Array{Vector{Float64}},
    x::Vector{Vector{Float64}}, 
    segs,
    sensor_args;
    sorted=false, fov, num_a
)
    a = sensor_args    
    w,  s_noise, outlier, outlier_vol, zmax = a.w, a.s_noise, a.outlier, a.outlier_vol, a.zmax
    return eval_pose_vectors(vs, x, segs, w, s_noise, outlier, outlier_vol, zmax; sorted=sorted, fov=fov, num_a=num_a)
end

"""
    normalize_exp(x)

Apply `exp` then normalize. Actually, this is just `softmax` with temperature one, oh well...
"""
function normalize_exp(x)
    b = maximum(x)
    y = exp.(x .- b)
    return y / sum(y)
end;

"""
    raise_probs(p, vmin)

Raise a probaility vector to a minimal value.
"""
raise_probs(p, vmin) = (1 - vmin*length(p))*p .+  vmin

#nbx
# @dist function grid_proposal_from_center(p::Pose, obs_vector::Vector{Vector{Float64}}, grid_args, sensor_args, 
#                             tau::Float64 = 1.0, pmin::Float64 = 1e-6)
#     #
#     # Create pose vector grid
#     # around given pose p
#     #
#     v0 = Vector(p)
#     vs, ls = vector_grid(v0, grid_args...)

#     #
#     # Evaluate the pose grid with respect 
#     # to the observation vector
#     #
#     log_ps, = eval_pose_vectors(vs, obs_vector, _segs, _as, sensor_args...; sorted=false);    

#     #
#     # Sample the new pose p′. 
#     # Two brief notes:
#     #
#     #  1. We have to raise the probabilities to a non-zero 
#     #     value otherwise we run into problems in MH updates.
#     #
#     #  2. Note that we don't assign the new pose to 
#     #     a particular time-step -- this is done in 
#     #     the transforms defined below.
#     #
#     # Todo: Make the size of the uniform sampling area an argument.
#     #
#     log_ps_tau = log_ps ./ tau
#     probs      = normalize_exp(log_ps_tau)
#     probs      = raise_probs(probs, pmin)
#     j ~ categorical(probs)

#     x′  = {:pos}  ~ mvuniform(vs[j][1:2] -  grid_args.r[1:2]/2, vs[j][1:2] + grid_args.r[1:2]/2)
#     # hd′ = {:pose => :hd} ~   uniform(vs[j][3]   -  grid_args.r[3]  /2, vs[j][3]   + grid_args.r[3]  /2)

#     return Pose(x′, hd′), (j, vs, ls, log_ps)
# end;

# #nbx
# @gen function grid_proposal(tr::Gen.Trace, t_chain::Int, obs_vector, grid_args, sensor_args, 
#     u::Control = Control([0.0;0.0],0.0),
#     tau::Float64 = 1.0, pmin::Float64 = 1e-6)

#     #
#     # Convert from chain-time to model-time
#     # extract pose from trace
#     #
#     t = t_chain + 1
#     p = get_pose(tr, t)
#     p = p + u

#     re = {*} ~ grid_proposal_from_center(p, obs_vector, grid_args, sensor_args, tau, pmin)

#     return re
# end


#
# Involution to use the proposal in a MH move
#
# @transform involution (tr, aux) to (tr′, aux′) begin
#     # Here `aux` is the proposal-trace 
#     # and `tr` the slam-trace
#     _,t_chain,_,grid_args, = get_args(aux)

#     x′  = @read(aux[:pose => :x] , :continuous)
#     hd′ = @read(aux[:pose => :hd], :continuous)
#     @write(tr′[add_addr_prefix(t_chain, :pose => :x )],  x′, :continuous)
#     @write(tr′[add_addr_prefix(t_chain, :pose => :hd)], hd′, :continuous)
    
#     x  = @read(tr[add_addr_prefix(t_chain, :pose => :x )], :continuous)
#     hd = @read(tr[add_addr_prefix(t_chain, :pose => :hd)], :continuous)
#     j = grid_index([x;hd], [x′;hd′], grid_args..., linear=true)
#     @write(aux′[:j],   j, :discrete)
#     @write(aux′[:pose => :x],   x, :continuous)
#     @write(aux′[:pose => :hd], hd, :continuous)
# end

# #
# # Transform to use the proposal in a PF update, i.e.
# # read in pose at time t and propose pose at time t+1 
# #
# @transform transform (aux) to (tr) begin
#     # Here `aux` is the proposal-trace 
#     # and `tr` the slam-trace
#     _,t_chain,_,grid_args, = get_args(aux)
    
#     x  = @read(aux[:pose => :x], :continuous)
#     hd = @read(aux[:pose => :hd], :continuous)
#     @write(tr[add_addr_prefix(t_chain+1,:pose => :x)],   x, :continuous)
#     @write(tr[add_addr_prefix(t_chain+1,:pose => :hd)], hd, :continuous)
# end

grid_schedule(grid_args, scaling_factor, i) = (k=grid_args.k, r=grid_args.r*scaling_factor^(i-1))
grid_schedule(grid_args, i) = grid_schedule(grid_args, 1/2, i)

# grid_args = (
#     k = [3,3,3],
#     r = [0.3, 0.3, 3/180*π]
# )
# grid_args_fine = (
#     k = [25,25,25],
#     r = [0.1, 0.1, 1/180*π]
# )


#=
To use this:

obs_vector = _ys[t]
for (i, tr) in enumerate(state.traces)
    for j = 1:M
        proposal_args = (t-1, obs_vector, grid_schedule(grid_args, j), sensor_args);
        tr, acc = Gen.mh(tr, grid_proposal, proposal_args, involution)
    end
    state.new_traces[i] = tr
end
update_refs!(state)
=#