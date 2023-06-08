using Fmt: @f_str, format # Python-style f-strings

nice_f(x) = f"{$x:0.2f}";

function Plots.plot!(p::Pose; r=0.5, args...)
    Plots.plot!([p.x, p.x + r*unit_vec(p.hd)]; args...)
end;

function Plots.plot!(ps::Vector{Pose}, cs::Vector{Colors.RGBA{Float64}}; r=0.5, args...)
    myplot=nothing
    for (p,c) in zip(ps,cs)
        myplot = Plots.plot!([p.x, p.x + r*unit_vec(p.hd)];c=c, args...)
    end
    return myplot
end;

function Plots.plot!(ps::Vector{Pose}; r=0.5, label=nothing, args...)
    myplot=nothing
    for (i,p) in enumerate(ps)
        if i > 1 label=nothing end
        myplot = Plots.plot!([p.x, p.x + r*unit_vec(p.hd)]; label=label, args...)
    end
    return myplot
end;

# function Plots.plot!(s::Geometry.Segment; args...)
#     Plots.plot!([s.x[1],s.y[1]], [s.x[2],s.y[2]]; args...)
# end

# function Plots.plot!(segs::Vector{Geometry.Segment}; label=nothing, args...)
#     myplot = nothing
#     for (i,s) in enumerate(segs)
#         if i != 1 label=nothing end
#         myplot = Plots.plot!(s;label=label, args...)
#     end
#     return myplot
# end