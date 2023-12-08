"""
    GridWorlds.Viz

Gridworld visualization module.
"""
module Viz

import Makie
using Makie: Vec2, Point2, Rect2, @lift, lift, Observable
using ..GridWorlds: GridWorld, place_agent, move_agent, points_from_raytracing
using ..GridWorlds: empty, wall, agent

export gridworldplot, visualize_grid, visualize_interactive_grid, interactive_gui, display_pf_localization!

"""Makie plotting recipe for plotting a GridWorld"""
Makie.@recipe(GridWorldPlot) do scene
    Makie.Attributes(
        squarecolors=Dict(
            empty => :white,
            wall => :black,
            agent => :red
        )
    )
end
function Makie.plot!(g::GridWorldPlot{<:Tuple{<:GridWorld}})
    w = g[1] # get the gridworld

    # plot the squares
    rects = [Rect2((x - 1, y - 1), (1, 1)) for x in 1:size(w[])[1] for y in 1:size(w[])[2]]
    colors = @lift([$(g.squarecolors)[$w[x, y]] for x in 1:size(w[])[1] for y in 1:size(w[])[2]])
    Makie.poly!(g, rects, color=colors)

    g
end

"""
Visualize a GridWorld.
"""
function visualize_grid(w; size=(400, 400), kwargs...)
    f = Makie.Figure(; size)
    ax = Makie.Axis(f[1, 1], aspect=Makie.DataAspect())
    Makie.hidedecorations!(ax)
    gridworldplot!(ax, w; kwargs...)
    f
end

##################################
### Interactive visualizations ###
##################################

### Canonical mappings from keys to controls ###
"""
WASD -> up, left, down, right; E -> stay
"""
WASDE_KEYS() = (
    up=[:w, :up],
    down=[:s, :down],
    left=[:a, :left],
    right=[:d, :right],
    stay=[Makie.Keyboard.space, :e]
)
"""
WASD -> up, left, down, right; E -> stay;
T -> Increment time; G -> Decrement time
"""
WASDE_TG_KEYS() = (;
    timeup=[:t], timedown=[:g],
    WASDE_KEYS()...
)

### Register keyboard listeners ###
"""
Register keyboard listeners for a Makie.Axis.
`keys` is a dictionary mapping action names to lists of keys that trigger that action.
`callbacks` is a dictionary mapping action names to callback functions.
"""
function register_keyboard_listeners(
    ax::Makie.Axis;
    keys=WASDE_KEYS(),
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
function visualize_grid_with_interactive_agent(w; world_updater=move_agent, kwargs...)
    if !(w isa Makie.Observable)
        w = Makie.Observable(w)
    end
    f = visualize_grid(w; kwargs...)
    register_keyboard_listeners(f; controls=(;
        up=() -> w[] = world_updater(w[], :up),
        down=() -> w[] = world_updater(w[], :down),
        left=() -> w[] = world_updater(w[], :left),
        right=() -> w[] = world_updater(w[], :right),
        stay=() -> w[] = world_updater(w[], :stay)
    ))
    f
end

DEFAULT_PLOT_SPECS() = [(; show_map=true, show_agent=true, show_obs=true)]
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
    gridmap::GridWorld,
    pos_obs_seq, # Observable of (pos_sequence, obs_sequence)
    take_action; # Callback function.  Accepts an action as input, and triggers an update to the pos_obs_seq
    plot_specs=DEFAULT_PLOT_SPECS(), # Specifications for a sequence of horizontally displayed plots
    size=(800, 800),
    additional_text="",
)
    t = Observable(length(pos_obs_seq[][1]) - 1)

    # Functions for interactivity
    function timeup()
        if t[] < length(pos_obs_seq[][1]) - 1
            t[] = t[] + 1
        end
    end
    function timedown()
        if t[] > 0
            t[] = t[] - 1
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

    # GridWorld, with agent displayed
    w = @lift(place_agent(gridmap, $pos_obs_seq[1][$t+1]))

    # Points corresponding to the observed distances
    obs_pts = @lift(
        map(Point2, collect(zip(points_from_raytracing(
            ($pos_obs_seq[1][$t+1])..., # agentx, agenty
            $pos_obs_seq[2][$t+1]      # observation points
        )...)))
    )

    ### Make plots ###
    f = Makie.Figure(; size)

    for (i, plot) in enumerate(plot_specs)
        ax = Makie.Axis(f[1, i], aspect=Makie.DataAspect())
        Makie.hidedecorations!(ax)

        squarecolors = Dict(
            empty => :white,
            agent => plot.show_agent ? :red : :white,
            wall => plot.show_map ? :black : :white
        )
        gridworldplot!(ax, w; squarecolors)

        if plot.show_obs
            Makie.scatter!(ax, obs_pts)
        end
    end

    ### Display information ###
    l = Makie.Label(f[2, :], lift(t, pos_obs_seq) do t, pos_obs_seq
        str = "time: $t | max time simulated to: $(length(pos_obs_seq[1]) - 1)"
        isempty(additional_text) ? str : (str * "\n" * additional_text)
    end)
    l.tellheight = true
    l.tellwidth = false

    ### Register event listeners ###
    register_keyboard_listeners(f;
        keys=WASDE_TG_KEYS(),
        callbacks=(;
            up=() -> take_action_and_increment_time(:up),
            down=() -> take_action_and_increment_time(:down),
            left=() -> take_action_and_increment_time(:left),
            right=() -> take_action_and_increment_time(:right),
            stay=() -> take_action_and_increment_time(:stay),
            timeup, timedown
        )
    )

    return (f, t)
end

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
            ($particles[1][$t+1], $particles[2][$t+1])
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
        Rect2(Vec2([x .- 1, y .- 1]), Vec2([1.0, 1.0]))
        for (x, y) in $particles[2]
    ])
    Makie.poly!(ax, boxes; color=colors)
end

## Visualizations for internal maps
function display_pf_state(t::Makie.Observable, particles;
    plot_specs=DEFAULT_PLOT_SPECS(), size=(800, 800),
)
    figure = Makie.Figure(; size)
    for (i, plot) in enumerate(plot_specs)
        ax = Makie.Axis(figure[1, i], aspect=Makie.DataAspect())
        Makie.hidedecorations!(ax)
    end

    display_pf_state!(figure, t, particles; plot_specs)
    figure
end

function display_pf_state!(f::Makie.Figure, t::Makie.Observable, particles;
    plot_specs=DEFAULT_PLOT_SPECS()
)
    for (i, plot) in enumerate(plot_specs)
        if plot.show_map
            ax = Makie.contents(f[1, i])[1]
            display_pf_state!(ax, t, particles)
        end
    end
end

"""
`particles` is an observable of (weights_seq, state_seq).
weights_seq[t] gives the particle weights for the belief state at time t.
state_seq[t] gives the particle positions and map for the belief state at time t.
"""
function display_pf_state!(ax::Makie.Axis, t, particles)
    # Draw all the maps
    num_particles = length(particles[][1][1])
    for i in 1:num_particles
        world = @lift($particles[2][$t+1][i].world)
        weight = @lift($particles[1][$t+1][i])
        # keep agent/empty square transparent so we can see the underlying map
        squarecolors = @lift(Dict(
            empty => Makie.RGBA(0, 0, 0, 0),
            agent => Makie.RGBA(0, 0, 0, 0),
            wall => Makie.RGBA(0, 0, 0, sqrt($weight)),
        ))
        gridworldplot!(ax, world; squarecolors)
    end
    # Draw the agent's position
    display_pf_localization!(ax,
        @lift(
            ($particles[1][$t+1], map(state -> state.pos, $particles[2][$t+1]))
        )
    )

end


end