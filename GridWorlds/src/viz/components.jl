import StatsFuns: logsumexp

setup_figure_from_map(gridworld, args...; kwargs...) =
    setup_figure_from_map(Observable(gridworld), args...; kwargs...)
function setup_figure_from_map(gridworld::Observable, fig_xsize; plot_map=true, hide_decorations=true)
    xsize, ysize = gridworld[].size
    fig_ysize = Int(floor(fig_xsize * ysize / xsize))
    f = Makie.Figure(;size=(fig_xsize, fig_ysize))
    ax = Makie.Axis(f[1, 1], aspect=Makie.DataAspect())
    if hide_decorations
        Makie.hidedecorations!(ax)
    end
    if plot_map
        gridworldplot!(ax, gridworld; squarecolors=DEFAULT_SQUARE_COLORS)
    end
    return f, ax
end

plot_obs!(ax, pos, obs, angles=nothing; kwargs...) = 
    plot_obs!(ax, Observable(pos), Observable(obs), angles; kwargs...)
function plot_obs!(ax, pos::Observable, obs::Observable, angles=nothing; is_continuous, show_lines_to_walls, alpha=1, color=:blue)
    obs_pts = @lift(map(Point2, collect(zip(points_from_raytracing(
        ($pos)..., # agentx, agenty
        ($obs);      # observation points
        angles=angles,
        is_continuous=is_continuous
    )...))))

    if show_lines_to_walls
        linespec = @lift(collect(Iterators.flatten(
            (
                Point2(($pos)...),
                pt,
                Point2(NaN, NaN)
            ) for pt in $obs_pts
        )))
        l = Makie.lines!(ax, linespec; linewidth=0.5, alpha, color)
    end

    sc = Makie.scatter!(ax, obs_pts; alpha, color)

    if show_lines_to_walls
        return [sc, l]
    else
        return sc
    end
end

plot_path!(ax, path; kwargs...) = plot_path!(ax, Observable(path); kwargs...)
function plot_path!(
    ax, path::Observable; alpha=Observable(1.), marker=:circle, color=nothing, colormap=:viridis,
    tail_length=nothing
)
    path_pts = @lift(map(Point2, $path))
    if isnothing(color)
        color = @lift(1:length($path))
    end
    if !isnothing(tail_length)
        # colorrange = @lift((length($path) - tail_length, length($path)))
        colorrange = (1, tail_length)
        path_pts = @lift($path_pts[max(1, length($path_pts) - tail_length):end])
    elseif colormap isa Array && colormap[1] == :white
        colorrange=@lift((-length($path), length($path)))
    else
        colorrange = @lift((1, length($path)))
    end
    l = Makie.lines!(ax, path_pts; color, alpha, colormap, colorrange)
    sc = Makie.scatter!(ax, path_pts; color, alpha, marker, colormap, colorrange)
    return [l, sc]
end
function logweights_to_alphas(logweights)
    norm_weights = exp.(logweights .- logsumexp(logweights))
    return max.(sqrt.(norm_weights), 0.05)
end

### Old plotting methods for PF localization for grid-cell agents ###

# TODO: Allow the plot_specs to explicitly control where to put PF observations
function display_pf_localization!(f::Makie.Figure, t::Makie.Observable, particles;
    plot_specs=DEFAULT_PLOT_SPECS()
)
    for (i, plot) in enumerate(plot_specs)
        if plot.show_map
            ax = Makie.contents(f[1, i])[1]
            display_pf_localization!(ax, t, particles)
        end
    end
end

"""
`particles` is an observable of (weights_seq, positions_seq).
weights_seq[t] gives the particle weights for the belief state at time t.
positions_seq[t] gives the particle positions for the belief state at time t.
"""
function display_pf_localization!(ax::Makie.Axis, t, particles)
    display_pf_localization!(ax,
        @lift(
            ( $particles[1][$t + 1], $particles[2][$t + 1] )
        )
    )
end
"""
`particles` is an observable of (weights, positions)
for the belief state to be displayed.
"""
function display_pf_localization!(ax::Makie.Axis, particles)
    colors = @lift([Makie.RGBA(0, 1, 0, sqrt(w)) for w in $particles[1]])
    boxes = @lift([ # A rectangle for each particle
        Rect2(Vec2([x .- 1, y .- 1]), Vec2([1., 1.]))
        for (x, y) in $particles[2]
    ])
    Makie.poly!(ax, boxes; color=colors)
end
