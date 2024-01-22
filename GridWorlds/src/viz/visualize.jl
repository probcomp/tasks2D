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
        pf = plot_path!(ax, path; alpha, colormap=[:white, :green])
    end

    gt = plot_path!(ax, gt_path; marker=:x, colormap=[:white, :purple])#:seaborn_rocket_gradient)

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

function pf_result_gif(
    args...;
    n_frames,
    framerate,
    n_panels=1, # 1 -> inference + gt overlaid; 2 -> separate; 3 -> separate, plus overlaid
    filetypes=["gif"],
    filename="__pf_result",
    kwargs...
)
    # get_figure = n_panels == 1 ? animateable_pf_results ? n_panels == 2 ? animateable_pf_results_2panel : animateable_pf_results_3panel
    f, fr = animateable_pf_results(args...; n_panels, kwargs...)
    gif_filename = nothing
    for filetype in filetypes
        n = Makie.record(f, filename*"."*filetype, 1:n_frames; framerate) do frame
            fr[] = frame
        end
        if filetype == "gif"
            gif_filename = n
        end
    end
    return gif_filename
end

# Returns `f, axes_for_inference_result, axes_for_gt`
function setup_animateable_pf_result_figure(gridmap, grid_xsize, labeltext, n_panels; titlesize=12)
    !(gridmap isa Observable) && (gridmap = Observable(gridmap))
    if n_panels == 1
        f, ax = setup_figure_from_map(gridmap, grid_xsize)
        return f, (ax,), ()
    elseif n_panels == 2
        xsize, ysize = gridmap[].size
        fig_ysize = 12 + Int(floor(grid_xsize * ysize / xsize)) + (isnothing(labeltext) ? 0 : 30)
        f = Makie.Figure(;size=(2 * grid_xsize, fig_ysize))
        ax1 = Makie.Axis(f[1, 1]; aspect=Makie.DataAspect(), title="True world state", text)
        ax2 = Makie.Axis(f[1, 2]; aspect=Makie.DataAspect(), title="Belief state", titlesize)
        Makie.hidedecorations!(ax1)
        Makie.hidedecorations!(ax2)
        gridworldplot!(ax1, gridmap; squarecolors=DEFAULT_SQUARE_COLORS)
        gridworldplot!(ax2, gridmap; squarecolors=DEFAULT_SQUARE_COLORS)
        return f, (ax2,), (ax1,)
    elseif n_panels == 3
        xsize, ysize = gridmap[].size
        fig_ysize = 12 + Int(floor(grid_xsize * ysize / xsize)) + (isnothing(labeltext) ? 0 : 30)
        f = Makie.Figure(;size=(2 * grid_xsize, 2 * fig_ysize))
        ax1 = Makie.Axis(f[1, 1]; aspect=Makie.DataAspect(), title="True world state", titlesize)
        ax2 = Makie.Axis(f[1, 2]; aspect=Makie.DataAspect(), title="Belief state", titlesize)
        ax3 = Makie.Axis(f[2, 2]; aspect=Makie.DataAspect(), title="True world state + belief state overlay", titlesize)
        Makie.hidedecorations!(ax1)
        Makie.hidedecorations!(ax2)
        Makie.hidedecorations!(ax3)
        gridworldplot!(ax1, gridmap; squarecolors=DEFAULT_SQUARE_COLORS)
        gridworldplot!(ax2, gridmap; squarecolors=DEFAULT_SQUARE_COLORS)
        gridworldplot!(ax3, gridmap; squarecolors=DEFAULT_SQUARE_COLORS)
        return f, (ax2, ax3), (ax1, ax3)
    end
end

function animateable_pf_results(
    gridmap,
    fr_to_gt_path, # frame -> List of (x, y) positions
    fr_to_pf_paths, # frame -> List of sequences of (x, y) positions
    fr_to_pf_logweights; # frame -> List of sequences of weights
    fig_xsize=800,
    fr_to_gt_obs=nothing,
    fr_to_particle_obss=nothing,
    show_lines_to_walls=false,
    n_panels=1,
    tail_length=nothing,
    labeltext=nothing,
    label_fontsize=12,
    big_label_text=nothing
)
    f, axes_for_inference_result, axes_for_gt = setup_animateable_pf_result_figure(gridmap, fig_xsize, labeltext, n_panels; titlesize=3/2 * label_fontsize)
    fr = Observable(1)

    gt_path = @lift(fr_to_gt_path($fr))
    pf_paths = @lift(fr_to_pf_paths($fr))
    pf_logweights = @lift(fr_to_pf_logweights($fr))

    alphas = @lift(logweights_to_alphas($pf_logweights))
    pf = nothing
    for ax in axes_for_inference_result
        for i in 1:length(pf_paths[])
            pf = plot_path!(ax, @lift($pf_paths[i]); alpha=@lift($alphas[i]), colormap=[:white, :green], tail_length)
        end
    end

    gt = nothing
    for ax in axes_for_gt
        gt = plot_path!(ax, gt_path; marker=:x, colormap=[:white, :purple], tail_length)#:seaborn_rocket_gradient)
    end

    gt_obs_viz = nothing
    if !isnothing(fr_to_gt_obs)
        for ax in axes_for_gt
            gt_obs = @lift(fr_to_gt_obs($fr))
            gt_obs_viz = plot_obs!(ax, @lift($gt_path[end]), gt_obs; is_continuous=true, show_lines_to_walls)
        end
    end

    particle_obs_viz = nothing
    if !isnothing(fr_to_particle_obss)
        particle_obss = @lift(fr_to_particle_obss($fr))
        for ax in axes_for_inference_result
            for i in 1:length(particle_obss[])
                obs = @lift($particle_obss[i])
                pos = @lift($pf_paths[i][end])
                alpha = @lift($alphas[i])
                particle_obs_viz = plot_obs!(ax, pos, obs; is_continuous=true, show_lines_to_walls, alpha, color=:lightgreen)
            end
        end
    end

    plots_to_label = [
        gt,
        pf,
        (isnothing(gt_obs_viz) ? () : (gt_obs_viz,))...,
        (isnothing(particle_obs_viz) ? () : (particle_obs_viz,))...
    ]
    labels = [
        "Ground Truth Trajectory",
        "Inferred Weighted Particles",
        (isnothing(gt_obs_viz) ? () : ("Observation unprojected w.r.t. true position",))...,
        (isnothing(particle_obs_viz) ? () : ("Observations unprojected w.r.t. inferred position",))...
    ]

    if n_panels == 1
        Makie.axislegend(ax, plots_to_label, labels, position=:rb)
    elseif n_panels == 3
        if !isnothing(big_label_text)
            g = Makie.GridLayout(f[2, 1])
            l = Makie.Label(g[1, 1], Makie.@lift(big_label_text($fr)), fontsize=2 * label_fontsize)
            l.tellheight=true
            leg = Makie.Legend(g[2, 1], plots_to_label, labels, labelsize=3/2 * label_fontsize)
        else
            leg = Makie.Legend(f[2, 1], plots_to_label, labels, labelsize=3/2 * label_fontsize)
        end
        # l.tellheight = true
        # l.tellwidth = true
    end

    if !isnothing(labeltext)
        pos = n_panels > 2 ? 3 : 2
        l = Makie.Label(f[pos, :], labeltext, fontsize=label_fontsize)
        l.tellheight=true
        # Makie.contents(f[1, 1])[1].tellheight = true
        # Makie.contents(f[1, 1])[1].tellwidth = true
    end

    Makie.rowsize!(f.layout, 1, Makie.Relative(1/2))
    Makie.rowsize!(f.layout, 2, Makie.Relative(1/2))
    Makie.colsize!(f.layout, 1, Makie.Relative(1/2))
    Makie.colsize!(f.layout, 2, Makie.Relative(1/2))

    return f, fr
end

function animateable_pf_results_2panel(
    gridmap,
    fr_to_gt_path, # frame -> List of (x, y) positions
    fr_to_pf_paths, # frame -> List of sequences of (x, y) positions
    fr_to_pf_logweights; # frame -> List of sequences of weights
    fig_xsize=800,
    fr_to_gt_obs=nothing,
    fr_to_particle_obss=nothing,
    show_lines_to_walls=false,
    tail_length=nothing,
    labeltext=nothing,
    label_fontsize=12
)
    ## Fig setup
    !(gridmap isa Observable) && (gridmap = Observable(gridmap))
    xsize, ysize = gridmap[].size
    fig_ysize = 12 + Int(floor(1/2 * fig_xsize * ysize / xsize)) + (isnothing(labeltext) ? 0 : 30)
    f = Makie.Figure(;size=(fig_xsize, fig_ysize))
    ax1 = Makie.Axis(f[1, 1], aspect=Makie.DataAspect(), title="True world state")
    ax2 = Makie.Axis(f[1, 2], aspect=Makie.DataAspect(), title="Belief state")
    Makie.hidedecorations!(ax1)
    Makie.hidedecorations!(ax2)
    gridworldplot!(ax1, gridmap; squarecolors=DEFAULT_SQUARE_COLORS)
    gridworldplot!(ax2, gridmap; squarecolors=DEFAULT_SQUARE_COLORS)

    ## Observable
    fr = Observable(1)
    gt_path = @lift(fr_to_gt_path($fr))
    pf_paths = @lift(fr_to_pf_paths($fr))
    pf_logweights = @lift(fr_to_pf_logweights($fr))
    alphas = @lift(logweights_to_alphas($pf_logweights))

    ## Plot ax1
    gt = plot_path!(ax1, gt_path; marker=:x, colormap=[:white, :purple], tail_length)#:seaborn_rocket_gradient)
    gt_obs_viz = nothing
    if !isnothing(fr_to_gt_obs)
        gt_obs = @lift(fr_to_gt_obs($fr))
        gt_obs_viz = plot_obs!(ax1, @lift($gt_path[end]), gt_obs; is_continuous=true, show_lines_to_walls)
    end

    ## Plot ax2
    pf = nothing
    for i in 1:length(pf_paths[])
        pf = plot_path!(ax2, @lift($pf_paths[i]); alpha=@lift($alphas[i]), colormap=[:white, :green], tail_length)
    end
    particle_obs_viz = nothing
    if !isnothing(fr_to_particle_obss)
        particle_obss = @lift(fr_to_particle_obss($fr))
        for i in 1:length(particle_obss[])
            obs = @lift($particle_obss[i])
            pos = @lift($pf_paths[i][end])
            alpha = @lift($alphas[i])
            particle_obs_viz = plot_obs!(ax2, pos, obs; is_continuous=true, show_lines_to_walls, alpha, color=:darkolivegreen)
        end
    end

    if !isnothing(labeltext)
        l = Makie.Label(f[2, :], labeltext, fontsize=label_fontsize)
        l.tellheight=true
    end

    ## Return
    return f, fr
end

function interactive_2panel(
    take_action, # action -> (); triggers a frame update
    gridmap, # Observable
    gt_path, # Observable
    pf_paths, # Observable on particle trajectories
    pf_logweights; # Observable of [logweights1, logweights2, ..]
    fig_xsize=800,
    gt_obs=nothing, # or Observable of [obs1, obs2, ...]
    particle_obss=nothing, # or Observable of 
    show_lines_to_walls=false,
    close_on_hitwall=false,
    did_hitwall_observable=nothing,
    close_window=nothing,
    save_fn = () -> nothing,
)
    ## Fig setup
    !(gridmap isa Observable) && (gridmap = Observable(gridmap))
    xsize, ysize = gridmap[].size
    fig_ysize = 12 + Int(floor(fig_xsize * ysize / xsize))
    f = Makie.Figure(;size=(2 * fig_xsize, fig_ysize))
    ax1 = Makie.Axis(f[1, 1], aspect=Makie.DataAspect(), title="True world state")
    ax2 = Makie.Axis(f[1, 2], aspect=Makie.DataAspect(), title="Belief state")
    Makie.hidedecorations!(ax1)
    Makie.hidedecorations!(ax2)
    gridworldplot!(ax1, gridmap; squarecolors=DEFAULT_SQUARE_COLORS)
    gridworldplot!(ax2, gridmap; squarecolors=DEFAULT_SQUARE_COLORS)

    ## Observables
    t = @lift(length($gt_path) - 1)
    alphas = @lift(logweights_to_alphas($pf_logweights))

    ## Interactivity
    function take_action_and_increment_time(a)
        begin
            # Only take the action if the displayed time
            # is the farthest time we have simulated to
            if t[] == length(gt_path[]) - 1
                take_action(a)
            end
        end
    end
    register_keyboard_listeners(f;
        keys=WASDE_TG_08_SPACE_KEYS(),
        callbacks=(;
            up = () -> take_action_and_increment_time(:up),
            down = () -> take_action_and_increment_time(:down),
            left = () -> take_action_and_increment_time(:left),
            right = () -> take_action_and_increment_time(:right),
            stay = () -> take_action_and_increment_time(:stay),
            timeup = (() -> nothing), timedown = (() -> nothing),
            animate_from_0 = (() -> nothing),
            save = () -> save_fn(),
            pause_or_resume = (() -> nothing),
        )
    )

    if close_on_hitwall && !isnothing(close_window)
        Makie.on(did_hitwall_observable) do hitwall
            if hitwall
                # save_fn(actions) # No need for this now; window closing will trigger this
                close_window(f)
            end
        end
    end

    # Save whenever the window closes.
    closed = Ref(false)
    Makie.on(Makie.events(f).window_open) do isopen
        if !closed[] && !isopen
            closed[] = true
            save_fn()
        elseif isopen
            closed[] = false
        end
    end

    ## Plot ax1
    gt = plot_path!(ax1, gt_path; marker=:x, colormap=[:white, :purple])#:seaborn_rocket_gradient)
    gt_obs_viz = nothing
    if !isnothing(gt_obs)
        gt_obs_viz = plot_obs!(ax1, @lift($gt_path[end]), gt_obs; is_continuous=true, show_lines_to_walls)
    end

    ## Plot ax2
    pf = nothing
    for i in 1:length(pf_paths[])
        pf = plot_path!(ax2, @lift($pf_paths[i]); alpha=@lift($alphas[i]), colormap=[:white, :green])
    end
    particle_obs_viz = nothing
    if !isnothing(particle_obss)
        for i in 1:length(particle_obss[])
            obs = @lift($particle_obss[i])
            pos = @lift($pf_paths[i][end])
            alpha = @lift($alphas[i])
            particle_obs_viz = plot_obs!(ax2, pos, obs; is_continuous=true, show_lines_to_walls, alpha, color=:darkolivegreen)
        end
    end

    ## Return
    return f, t
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

### Grid inference steps visualization ###

# All args are observables.
function inference_steps_animation(
    gridmap::Observable,
    inferred_paths_to_this_point::Observable,
    path_weights::Observable,
    current_obs::Observable, # Observable over current dist vector
    # Observable over a vector of grid points (each grid will be displayed)
    # (Design question - should these be by particle?  I think no?)
    grids_present::Observable,
    current_label::Observable;
    fig_xsize=800,
    title=nothing,
    tail_length=8
)
    f, ax = setup_figure_from_map(gridmap, fig_xsize)
    if !isnothing(title)
        ax.title = title
    end

    # Plot the grids
    flattened_pts = @lift(collect(Iterators.flatten($grids_present)))
    pts = @lift(map(Point2, $flattened_pts))
    Makie.scatter!(ax, pts, color=:red, alpha=0.5)

    alphas = @lift(logweights_to_alphas($path_weights))
    pf = nothing
    for i in 1:length(inferred_paths_to_this_point[])
        pf = plot_path!(
            ax, @lift($inferred_paths_to_this_point[i]);
            alpha=@lift($alphas[i]), colormap=[:white, :green],
            tail_length
        )
    end

    # Plot the current observation
    for i in 1:length(inferred_paths_to_this_point[])
        plot_obs!(
            ax, @lift($inferred_paths_to_this_point[i][end]), current_obs;
            is_continuous=true, alpha=@lift($alphas[i]), color=:darkolivegreen,
            show_lines_to_walls=false
        )
    end

    # Plot the current label
    l = Makie.Label(f[2, 1], current_label)
    # Makie.rowsize!(f.layout, 1, Makie.Relative(10))
    # Makie.rowsize!(f.layout, 2, Makie.Relative(1))
    # Makie.rowsize!(f.layout, 1, Makie.Relative(5))
    l.tellheight = true
    l.tellwidth=false
    ax.tellheight = false
    ax.tellwidth=false

    return f
end

end # module