import Makie

"""Makie plotting recipe for plotting a GridWorld"""
Makie.@recipe(GridWorldPlot) do scene
    Makie.Attributes(
        squarecolors = Dict(
            empty => :white,
            wall => :black,
            agent => :red
        )
    )
end
function Makie.plot!(g::GridWorldPlot{<:Tuple{<:GridWorld}})
    w = g[1] # get the gridworld

    # plot the squares
    rects = [Makie.Rect2( (x - 1, y - 1), (1, 1) ) for x in 1:w[].size[1] for y in 1:w[].size[2]]
    colors = Makie.@lift([$(g.squarecolors)[$w[x, y]] for x in 1:w[].size[1] for y in 1:w[].size[2]])
    Makie.poly!(g, rects, color = colors)

    g
end

"""
Visualize a GridWorld.
"""
function visualize_grid(w; resolution=(400, 400), kwargs...)
    f = Makie.Figure(;resolution)
    ax = Makie.Axis(f[1, 1], aspect=Makie.DataAspect())
    Makie.hidedecorations!(ax)
    gridworldplot!(ax, w; kwargs...)
    f
end

"""
Visualize a GridWorld and register keyboard listeners
so WASD and arrow keys move the agent around.

`world_updater` is a function that takes a GridWorld and a direction
(specified by a symbol :up, :down, :left, :stay, or :right) and returns a new
GridWorld.
"""
function visualize_interactive_grid(w; world_updater = moveagent, kwargs...)
    if !(w isa Makie.Observable)
        w = Makie.Observable(w)
    end
    f = visualize_grid(w; kwargs...)
    register_keyboard_listeners(f;
        up    = () -> w[] = world_updater(w[], :up),
        down  = () -> w[] = world_updater(w[], :down),
        left  = () -> w[] = world_updater(w[], :left),
        right = () -> w[] = world_updater(w[], :right),
        stay  = () -> w[] = world_updater(w[], :stay)
    )
    f
end


### Keyboard control ###
register_keyboard_listeners(f::Makie.Figure; kwargs...) =
    register_keyboard_listeners(Makie.contents(f[1, 1])[1]; kwargs...)

WASDE_KEYS() = (
    up    = [:w, :up],
    down  = [:s, :down],
    left  = [:a, :left],
    right = [:d, :right],
    stay  = [Makie.Keyboard.space, :e]
)
WASDE_TG_KEYS() = (;
    timeup = [:t], timedown = [:g],
    WASDE_KEYS()...
)

function register_keyboard_listeners(
    ax::Makie.Axis;
    keys = WASDE_KEYS(),
    callbacks=nothing,
    kwargs...
)
    if isnothing(callbacks)
        callbacks = kwargs
    end

    Makie.on(Makie.events(ax).keyboardbutton) do event
        if event.action in (Makie.Keyboard.press, Makie.Keyboard.repeat)
            for (actionname, keys) in pairs(keys)
                if event.key in [keyname isa Symbol ? getfield(Makie.Keyboard, keyname) : keyname for keyname in keys]
                    callbacks[actionname]()
                end
            end
        end
    end

        # ups, downs, lefts, rights = (
        #     [getfield(Makie.Keyboard, x) for x in xs]
        #     for xs in [
        #             [:w, :up], [:s, :down], [:a, :left], [:d, :right]
        #         ]
        # )

        

        # if event.action in (Makie.Keyboard.press, Makie.Keyboard.repeat)
        #     if event.key in ups
        #         callbap()
        #     elseif event.key in downs
        #         down()
        #     elseif event.key in lefts
        #         left()
        #     elseif event.key in rights
        #         right()
        #     elseif event.key in [:e, Makie.Keyboard.space]
        #         stay()
        #     end
        # end
    # end
end

function interactive_mode_gui(world, pos_obs_sequence_observable, take_action; show_agent=true, show_map=true)
    t = Observable(length(pos_obs_sequence_observable[][1]) - 1)

    # Functions for interactivity
    function timeup()
        if t[] < length(pos_obs_sequence_observable[][1]) - 1
            t[] = t[] + 1
        end
    end
    function timedown()
        if t[] > 0
            t[] = t[] - 1
        end
    end
    function take_action_and_increment_time(a)
        @async begin
            take_action(a)
            t[] = t[] + 1
        end
    end

    f = visualize_grid(
        Makie.@lift(moveagent(world, $pos_obs_sequence_observable[1][$t + 1]));
        resolution=(800, 800),
        squarecolors=Dict(
            empty => :white,
            agent => show_agent ? :red : :white,
            wall => show_map ? :black : :white
        )
    )
    obs_pts = Makie.@lift(
        map(Makie.Point2,
            zip(points_from_raytracing(
                # Position
                ($pos_obs_sequence_observable[1][$t + 1])...,
                # Observation pts
                $pos_obs_sequence_observable[2][$t + 1]
            )...)
        )
    )
    Makie.scatter!(obs_pts)
    
    register_keyboard_listeners(f;
        keys=WASDE_TG_KEYS(),
        callbacks=(;
            up = () -> take_action_and_increment_time(:up),
            down = () -> take_action_and_increment_time(:down),
            left = () -> take_action_and_increment_time(:left),
            right = () -> take_action_and_increment_time(:right),
            stay = () -> take_action_and_increment_time(:stay),
            timeup, timedown
        )
    )
    
    tostr(x) = "$x"
    
    l = Makie.Label(f[2, 1], Makie.@lift("time shown: "*tostr($t)*" | max time simulated to: "*tostr(-1 + length($pos_obs_sequence_observable[1]))))
    l.tellheight=true; l.tellwidth=false
    
    return (f, t)
end

function display_pf_localization!(f::Makie.Figure, t::Makie.Observable, pf_states)
    ax = Makie.contents(f[1, 1])[1]
    display_pf_localization!(ax, Makie.@lift($pf_states[$t + 1]))
end
using GenParticleFilters: get_traces, get_norm_weights
function display_pf_localization!(ax::Makie.Axis, pf_state)
    posaddr(t) = GenPOMDPs.state_addr(t, :pos)
    obsaddr(t) = GenPOMDPs.obs_addr(t, :obs)

    trs = Makie.lift(get_traces, pf_state)
    weights = Makie.lift(get_norm_weights, pf_state)
    colors = Makie.lift(weights) do ws
        [Makie.RGBA(0, 1, 0, sqrt(w)) for w in ws]
    end
    boxes = Makie.lift(trs) do trs
        [
            let (x, y) = tr[posaddr(get_args(tr)[1])]
                Makie.Rect2(
                    Makie.Vec2([x.-1, y.-1]),
                    Makie.Vec2([1., 1.])
                )
            end
            for tr in trs
        ]
    end
    Makie.poly!(ax, boxes; color=colors)
end