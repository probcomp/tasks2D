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