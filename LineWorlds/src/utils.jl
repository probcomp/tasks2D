######################################################################
# Adapted from https://github.com/probcomp/probabilistic-slam-in-gen #
######################################################################

##################################### 
module LineWorldUtils  
#####################################

using LinearAlgebra

unit_vec(a::Float64) = [cos(a);sin(a)];
LinearAlgebra.angle(x::Vector{Float64}) = atan(x[2],x[1]);
peak_to_peak(xs) = (xs .- minimum(xs))./(maximum(xs) - minimum(xs))

polar(x::Vector{Float64}) = [norm(x);atan(x[2],x[1])];
polar_inv(zs::Vector{Float64}, as::Vector{Float64}) = [[z*cos(a);z*sin(a)] for (z,a) in zip(zs,as)];
polar_inv(r_and_phi::Vector{Float64}) = [r_and_phi[1]*cos(r_and_phi[2]);r_and_phi[1]*sin(r_and_phi[2])]
polar_inv(r::Float64, phi::Float64)   = [r*cos(phi);r*sin(phi)]
polar_inv(z::Array, a::Array) = cat(z.*cos.(a), z.*sin.(a), dims=ndims(a)+1);

export unit_vec, polar, angle, stack, peak_to_peak, euclidean, polar_inv

"""
Stacks vectors on top of each other (as rows, along dim 1)
"""
stack(xs::AbstractVector) = reduce(vcat, transpose.(xs));
unstack(x::Matrix) = [x[i,:] for i=1:size(x,1)]

"""
Stacks vectors horizontally (along dim 2)
"""
hstack(xs::AbstractVector) = reduce(hcat,xs);

export stack, hstack, unstack

"""
    rot(hd)

Returns 2D rotation matrix.
"""
rot(hd) = [[cos(hd) -sin(hd)]; [sin(hd) cos(hd)]]

export rot

using Colors, Plots
col = palette(:default);

Plots.scatter!(xs::Vector{Vector{Float64}}; args...) = scatter!([x[1] for x in xs], [x[2] for x in xs]; args...)
Plots.plot!(xs::Vector{Vector{Float64}}; args...)    = plot!([x[1] for x in xs], [x[2] for x in xs]; args...)

# using Fmt: @f_str, format
# function summarize_vars(ex::Expr; fstr=f"{1:<10.10} {2:<}")
#     for sx in ex.args
#         x = getproperty(Main, sx)
#         println(format(fstr, sx, typeof(x)))
#     end
# end;

end # module LineWorldUtils