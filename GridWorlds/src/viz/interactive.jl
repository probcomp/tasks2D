##################################
### Interactive visualizations ###
##################################

### Canonical mappings from keys to controls ###
"""
WASD -> up, left, down, right; E -> stay
"""
WASDE_KEYS() = (
    up    = [:w, :up],
    down  = [:s, :down],
    left  = [:a, :left],
    right = [:d, :right],
    stay  = [Makie.Keyboard.space, :e]
)
"""
WASD -> up, left, down, right; E -> stay;
T -> Increment time; G -> Decrement time
"""
WASDE_TG_KEYS() = (;
    timeup = [:t], timedown = [:g],
    WASDE_KEYS()...
)

"""
WASDE, TG as above
0 = animate from time 0
8 = save current trace
"""
WASDE_TG_08_KEYS() = (;
    WASDE_TG_KEYS()...,
    animate_from_0 = [:0, :m],
    save = [:8, :n]
)

WASDE_TG_08_SPACE_KEYS() = (;
    WASDE_TG_KEYS()...,
    animate_from_0 = [:0, :m],
    save = [:8, :n],
    pause_or_resume = [Makie.Keyboard.space]
)

### Register keyboard listeners ###
"""
Register keyboard listeners for a Makie.Axis.
`keys` is a dictionary mapping action names to lists of keys that trigger that action.
`callbacks` is a dictionary mapping action names to callback functions.
"""
function register_keyboard_listeners(
    ax::Makie.Axis;
    keys = WASDE_KEYS(),
    callbacks,
)
    Makie.on(Makie.events(ax).keyboardbutton) do event
        if event.action in (Makie.Keyboard.press, Makie.Keyboard.repeat)
            for (actionname, keys) in pairs(keys)
                if event.key in [keyname isa Symbol ? getfield(Makie.Keyboard, keyname) : keyname for keyname in keys]
                    callbacks[actionname]()
                end
            end
        end
    end
end
"""Register keyboard listeners for the first axis of a Makie.Figure"""
register_keyboard_listeners(f::Makie.Figure; kwargs...) =
    register_keyboard_listeners(Makie.contents(f[1, 1])[1]; kwargs...)

"""
Visualize a GridWorld and register keyboard listeners
so WASDE and arrow keys move the agent around.

`world_updater` is a function that takes a GridWorld and a direction
(specified by a symbol :up, :down, :left, :stay, or :right) and returns a new
GridWorld.
"""
function visualize_grid_with_interactive_agent(w; world_updater = move_agent, kwargs...)
    if !(w isa Makie.Observable)
        w = Makie.Observable(w)
    end
    f = visualize_grid(w; kwargs...)
    register_keyboard_listeners(f; controls = (;
        up    = () -> w[] = world_updater(w[], :up),
        down  = () -> w[] = world_updater(w[], :down),
        left  = () -> w[] = world_updater(w[], :left),
        right = () -> w[] = world_updater(w[], :right),
        stay  = () -> w[] = world_updater(w[], :stay)
    ))
    f
end

"""
Visualize a GridWorld, and ray-trace point observations.
Map WASDE to taking actions in the world.
Map T/G to incrementing/decrementing the displayed time.

`gridmap` is a GridWorld with just walls and empty squares (the map).
`pos_obs_seq` is an observable of a pair (pos_sequence, obs_sequence).
`take_action` is a callback function that takes an action as input and
    triggers an update to `pos_obs_seq`.
`plot_specs` is a sequence of specifications for horizontally displayed plots;
    each specification is a named tuple of booleans (show_map, show_agent, show_obs).
"""
function interactive_gui(
    t_to_gridmap,
    pos_obs_seq, # Observable of (pos_sequence, obs_sequence)
                 # where each obs is a list of distances
    take_action; # Callback function.  Accepts an action as input, and triggers an update to the pos_obs_seq
    plot_specs = DEFAULT_PLOT_SPECS(), # Specifications for a sequence of horizontally displayed plots
    size=(800, 800),
    additional_text="",

    # If this is true, the agent is displayed as a discrete square
    # in the map, rather than a continuous position.
    # Its coordinates are discrete grid coordinates, not continuous coordinates.
    display_agent_as_cell=false,

    show_lines_to_walls=false,

    ray_angles=nothing, # Defaults to LinRange(-π/2, 3π/2, num_angles)
    save_fn = (viz_actions -> nothing),
    framerate=2,
    close_on_hitwall=false,
    did_hitwall_observable=nothing,
    close_window=nothing,

    timing_args=nothing # (action_times_observable, speedup_factor, max_delay)
)
    t = Observable(length(pos_obs_seq[][1]) - 1)
    actions = []

    # Functions for interactivity
    function timeup()
        if t[] < length(pos_obs_seq[][1]) - 1
            t[] = t[] + 1
            push!(actions, (:timeup, t[]))
        end
    end
    function timedown()
        if t[] > 0
            t[] = t[] - 1
            push!(actions, (:timedown, t[]))
        end
    end
    function take_action_and_increment_time(a)
        begin
            # Only take the action if the displayed time
            # is the farthest time we have simulated to
            if t[] == length(pos_obs_seq[][1]) - 1
                take_action(a)
                t[] = t[] + 1
            end
        end
    end

    gridmap = @lift(t_to_gridmap($t))
    if display_agent_as_cell
        # GridWorld, with agent displayed
        w = @lift(place_agent($gridmap, $pos_obs_seq[1][$t + 1]))
    else
        w = gridmap
    end

    # Points corresponding to the observed distances
    obs_pts = @lift(
        map(Point2, collect(zip(points_from_raytracing(
            ($pos_obs_seq[1][$t + 1])..., # agentx, agenty
            $pos_obs_seq[2][$t + 1];      # observation points
            angles=ray_angles,
            is_continuous=!display_agent_as_cell
        )...)))
    )
    
    ### Make plots ###
    f = Makie.Figure(;size)

    for (i, plot) in enumerate(plot_specs)
        ax = Makie.Axis(f[1, i], aspect=Makie.DataAspect())
        Makie.hidedecorations!(ax)

        squarecolors=Dict(
            empty => :white,
            agent => plot.show_agent ? :red : :white,
            wall => plot.show_map ? :black : :white,
            strange => plot.show_map ? :purple : :white
        )
        gridworldplot!(ax, w; squarecolors)

        if !display_agent_as_cell
            # If the agent is not displayed as a grid cell, we should
            # display it as a point in continuous space.
            Makie.scatter!(ax, @lift([Point2($pos_obs_seq[1][$t + 1]...)]), color=:red)
        end

        if plot.show_obs
            if show_lines_to_walls
                linespec = @lift( collect(Iterators.flatten(
                    (Point2($pos_obs_seq[1][$t + 1]...),
                    pt,
                    Point2(NaN, NaN)) for pt in $obs_pts
                )) )
                Makie.lines!(ax, linespec, linewidth=0.5)
            end
        
            Makie.scatter!(ax, obs_pts)
        end
    end

    ### Display information ###
    l = Makie.Label(f[2, :], lift(t, pos_obs_seq) do t, pos_obs_seq
        str = "time: $t | max time simulated to: $(length(pos_obs_seq[1]) - 1)"
        isempty(additional_text) ? str : (str * "\n" * additional_text)
    end)
    l.tellheight=true; l.tellwidth=false
    
    ### Register event listeners ###
    # animate_from_zero = _animate_from_zero(t, () -> length(pos_obs_seq[][1]) - 1; framerate)
    (animate_from_zero, pause_or_resume) = _get_animation_fns(
        t, () -> length(pos_obs_seq[][1]) - 1, actions;
        framerate, timing_args
    )
    register_keyboard_listeners(f;
        keys=WASDE_TG_08_SPACE_KEYS(),
        callbacks=(;
            up = () -> take_action_and_increment_time(:up),
            down = () -> take_action_and_increment_time(:down),
            left = () -> take_action_and_increment_time(:left),
            right = () -> take_action_and_increment_time(:right),
            stay = () -> take_action_and_increment_time(:stay),
            timeup, timedown,
            animate_from_0 = animate_from_zero,
            save = () -> save_fn(actions),
            pause_or_resume = pause_or_resume
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
            save_fn(actions)
        elseif isopen
            closed[] = false
        end
    end

    # Actions is a vector which will be populated with pairs (action, time)
    # for actions in [timeup, timedown, pause, resume]
    # to track how a user interacts with the visualization.
    return (f, t, actions)
end

function _get_animation_fns(t_observable, get_current_maxtime, actions;
    framerate=2,
    timing_args=nothing
)
    is_paused = Ref(false)
    function initialize_animate(starttime)
        is_paused[] = false
        @async for t = starttime:get_current_maxtime()
            if is_paused[]
                break;
            else
                t_observable[] = t

                if isnothing(timing_args)
                    sleep(1/framerate)
                else
                    (action_times_observable, speedup_factor, max_delay) = timing_args
                    if 1 < t <= length(action_times_observable[])
                        delta_ms = (action_times_observable[][t] - action_times_observable[][t - 1]).value
                        delta_s = delta_ms / 1000
                        waittime_s = min(delta_s/speedup_factor, max_delay)
                        sleep(waittime_s)
                    end
                end
            end
        end
    end
    function animate_from_zero()
        push!(actions, (:animate_from_zero, 0))
        initialize_animate(0)
    end
    function resume()
        push!(actions, (:resume, t_observable[]))
        initialize_animate(t_observable[])
    end
    function pause()
        push!(actions, (:pause, t_observable[]))
        is_paused[] = true
    end
    function pause_or_resume()
        if is_paused[]
            resume()
        else
            pause()
        end
    end
    return (animate_from_zero, pause_or_resume)
end

function _animate_from_zero(t_observable, get_current_maxtime; framerate=2)
    """
    Animate from t=0.
    """
    function _do_animate()
        println("[Animating]")
        maxT = get_current_maxtime()
        @async for t = 0:maxT
            t_observable[] = t
            sleep(1/framerate)
        end
    end
end

##############################
### Play as the agent mode ###
##############################

function play_as_agent_gui(
    obs_seq, # Observable of obs_sequence
    take_action;
    size=(800, 800),
    additional_text = "",
    worldsize=20,
    show_lines_to_walls=false,
    ray_angles=nothing, # Defaults to LinRange(-π/2, 3π/2, num_angles)
    save_fn = (viz_actions -> nothing),
    framerate=2,
    close_on_hitwall=false,
    did_hitwall_observable=nothing,
    close_window=nothing,

    timing_args=nothing # (action_times_observable, speedup_factor, max_delay)
)
    t = Observable(length(obs_seq[]) - 1)

    actions = []

    # Functions for interactivity
    function timeup()
        if t[] < length(obs_seq[]) - 1
            t[] = t[] + 1
            push!(actions, (:timeup, t[]))
        end
    end
    function timedown()
        if t[] > 0
            t[] = t[] - 1
            push!(actions, (:timedown, t[]))
        end
    end
    function take_action_and_increment_time(a)
        begin
            # Only take the action if the displayed time
            # is the farthest time we have simulated to
            if t[] == length(obs_seq[]) - 1
                take_action(a)
                t[] = t[] + 1
            end
        end
    end

    ### Make plot ###
    f = Makie.Figure(; size)
    ax = Makie.Axis(f[1, 1], aspect=Makie.DataAspect())
    Makie.hidedecorations!(ax)

    # Plot obs
    egocentric_obs_pts = @lift(
        map(Point2, collect(zip(points_from_raytracing(
            0, 0, # egocentric coordinates
            $obs_seq[$t + 1];
            angles=ray_angles,
            is_continuous=true
        )...)))
    )

    if show_lines_to_walls
        linespec = @lift( collect(Iterators.flatten(
            (Point2(0, 0), pt, Point2(NaN, NaN)) for pt in $egocentric_obs_pts
        )) )
        Makie.lines!(ax, linespec, linewidth=0.5)
    end
    Makie.scatter!(ax, egocentric_obs_pts)

    # Plot agent as circle in the center of the map
    Makie.scatter!(ax, [0], [0], color=:red)

    # Makie.lines!(ax, Makie.lift(zero_pt_nan, obs_pt_xs), Makie.lift(zero_pt_nan, obs_pt_ys))

    ### Display information ###
    l = Makie.Label(f[2, :], lift(t, obs_seq) do t, obs_seq
        str = "time: $t | max time simulated to: $(length(obs_seq) - 1)"
        isempty(additional_text) ? str : (str * "\n" * additional_text)
    end)
    l.tellheight=true; l.tellwidth=false
    
    Makie.xlims!(ax, (-worldsize, worldsize))
    Makie.ylims!(ax, (-worldsize, worldsize))

    ### Register event listeners ###
    (animate_from_zero, pause_or_resume) = _get_animation_fns(
        t, () -> length(obs_seq[]) - 1, actions;
        framerate, timing_args
    )
    register_keyboard_listeners(f;
        keys=WASDE_TG_08_SPACE_KEYS(),
        callbacks=(;
            up = () -> take_action_and_increment_time(:up),
            down = () -> take_action_and_increment_time(:down),
            left = () -> take_action_and_increment_time(:left),
            right = () -> take_action_and_increment_time(:right),
            stay = () -> take_action_and_increment_time(:stay),
            timeup, timedown,
            animate_from_0 = animate_from_zero,
            save = () -> save_fn(actions),
            pause_or_resume = pause_or_resume
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
            save_fn(actions)
        elseif isopen
            closed[] = false
        end
    end

    return (f, t, actions)
end
