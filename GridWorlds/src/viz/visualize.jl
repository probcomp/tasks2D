"""
    GridWorlds.Viz

Gridworld visualization module.
"""
module Viz

import Makie
using Makie: Vec2, Point2, Rect2, @lift, lift, Observable
using ..GridWorlds: GridWorld, place_agent, move_agent, points_from_raytracing
using ..GridWorlds: empty, wall, agent, strange

export gridworldplot, visualize_grid, visualize_interactive_grid, interactive_gui, display_pf_localization!

# Used by some plotting methods which can be configured to show multiple panels
DEFAULT_PLOT_SPECS() = [ (; show_map=true, show_agent=true, show_obs=true) ]

include("./draw_gridworld.jl") # Makie plotting recipe for plotting a GridWorld
include("./interactive.jl") # Interactive GUI variants for playing in GridWorlds
include("./components.jl")

function mapviz_stillframe(
    gridworld, pos, obs;
    fig_xsize=800,
    display_agent_as_cell=false,
    show_obs=true,
    show_lines_to_walls=true,
    ray_angles=nothing,
    show_map=false,
    show_pos=true
)
    f, ax = setup_figure_from_map(gridworld, fig_xsize; plot_map=show_map)

    if show_obs
        plot_obs!(ax, pos, obs, ray_angles; is_continuous=!display_agent_as_cell, show_lines_to_walls)
    end

    if show_pos && !display_agent_as_cell
        # If the agent is not displayed as a grid cell, we should
        # display it as a point in continuous space.
        Makie.scatter!(ax, [Point2(pos...)], color=:red)
    end

    return f
end

function plot_paths(
    gridmap,
    paths, # List of sequences of (x, y) positions
    fig_xsize=800
)
    f, ax = setup_figure_from_map(gridmap, fig_xsize)

    # Plot paths
    for path in paths
        plot_path!(ax, path)
    end

    return f
end

function plot_pf_results(
    gridmap,
    gt_path,
    pf_paths, # List of sequences of (x, y) positions
    pf_logweights, # List of sequences of weights
    fig_xsize=800
)
    f, ax = setup_figure_from_map(gridmap, fig_xsize)

    alphas = logweights_to_alphas(pf_logweights)
    pf = nothing
    for (path, alpha) in zip(pf_paths, alphas)
        pf = plot_path!(ax, path; alpha)
    end

    gt = plot_path!(ax, gt_path; marker=:x, colormap=:seaborn_rocket_gradient)

    # l = Makie.Legend(
    #     f[2, 1], [gt, pf], ["Ground Truth Trajectory", "Inferred Trajectory Particles"]
    # )
    # ax.tellheight = true
    # l.tellheight = true
    # l.tellwidth = true

    Makie.axislegend(
        ax, [gt, pf],
        ["Ground Truth Trajectory", "Inferred Weighted Particles"],
        position=:rb
    )

    return f
end

function time_heatmap(
    gridmap,
    paths, # List of (x, y) positions
    timess; # List of times
    fig_xsize=800,
    clip_ms=5_000, # Clip time spent at a square to this many ms
    use_times=true, # otherwise, plot based on frequencies
    title="",
    do_avg=true,
    n_to_avg_over=length(paths),
    times_colorrange=(0, 3),
    steps_colorrange=(0, 4),
    showticks=false
)
    f, ax = setup_figure_from_map(gridmap, fig_xsize, plot_map=false, hide_decorations=!showticks)
    
    xsize, ysize = Base.size(gridmap)
    heatmap = zeros(xsize, ysize)

    for (path, times) in zip(paths, timess)
        for (i, (x, y)) in enumerate(path)
            i == 1 && continue
            if use_times
                if i < length(times)
                    delta_t = (times[i + 1] - times[i]).value
                    delta_t = min(delta_t, clip_ms)
                    delta_t = delta_t/1000 # seconds
                    heatmap[Int(floor(x + 0.5)), Int(floor(y + 0.5))] += delta_t
                end
            else
                heatmap[Int(floor(x + 0.5)), Int(floor(y + 0.5))] += 1
            end
        end
    end
    if do_avg
        heatmap /= n_to_avg_over
    end

    hm = Makie.heatmap!(ax, 0:1.0:(xsize), 0:1.0:(ysize), heatmap, colormap=:tempo,
        colorrange=(use_times ? times_colorrange : steps_colorrange)
    )

    gridworldplot!(ax, gridmap; squarecolors=DEFAULT_SQUARE_COLORS)

    Makie.Colorbar(f[1, 2], hm, label=(use_times ? "Time spent at square (s)" : "# Steps spent in square"))

    return f
end

### Gif ###
function trace_gif(
    t_to_gridmap,
    positions,
    observations;
    fig_xsize=800,
    additional_text="",

    draw_path=true,

    # If this is true, the agent is displayed as a discrete square
    # in the map, rather than a continuous position.
    # Its coordinates are discrete grid coordinates, not continuous coordinates.
    display_agent_as_cell=false,

    show_lines_to_walls=false,
    show_obs=true,

    ray_angles=nothing, # Defaults to LinRange(-π/2, 3π/2, num_angles)

    framerate=2,
)
    maxT = length(positions) - 1
    @assert length(observations) == maxT + 1

    t = Observable(1)

    ### Make plots ###
    f, ax = setup_figure_from_map(@lift(t_to_gridmap($t)), fig_xsize)

    if !display_agent_as_cell
        # If the agent is not displayed as a grid cell, we should
        # display it as a point in continuous space.
        Makie.scatter!(ax, @lift([Point2(positions[$t + 1]...)]), color=:red)
    end

    if show_obs
        plot_obs!(ax, @lift(positions[$t + 1]), @lift(observations[$t + 1]), ray_angles; is_continuous=!display_agent_as_cell, show_lines_to_walls)
    end

    if draw_path
        Makie.lines!(
            ax, @lift([Point2(positions[i]...) for i in 1:($t+1)]),
            color=@lift(1:($t+1))
        )
        Makie.scatter!(
            ax, @lift([Point2(positions[i]...) for i in 1:($t+1)]),
            color=@lift(1:($t+1))
        )
    end

    ### Display information ###
    l = Makie.Label(f[2, :], lift(t) do t
        str = "time: $t"
        isempty(additional_text) ? str : (str * "\n" * additional_text)
    end)
    l.tellheight=true; l.tellwidth=false

    ### Animate to Gif ###
    gif_filename = Makie.record(f, "__trace.gif", 1:maxT; framerate) do frame
        t[] = frame
    end
    
    return gif_filename
end


end # module