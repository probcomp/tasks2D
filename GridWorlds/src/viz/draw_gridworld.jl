
DEFAULT_SQUARE_COLORS = Dict(
    empty => Makie.RGBA(1, 1, 1, 0),
    agent => :red,
    wall => :black,
    strange => :purple
)

"""Makie plotting recipe for plotting a GridWorld"""
Makie.@recipe(GridWorldPlot) do scene
    Makie.Attributes(
        squarecolors=DEFAULT_SQUARE_COLORS
    )
end
function Makie.plot!(g::GridWorldPlot{<:Tuple{<:GridWorld}})
    w = g[1] # get the gridworld

    # plot the squares
    rects = [Rect2( (x - 1, y - 1), (1, 1) ) for x in 1:size(w[])[1] for y in 1:size(w[])[2]]
    colors = @lift([$(g.squarecolors)[$w[x, y]] for x in 1:size(w[])[1] for y in 1:size(w[])[2]])
    Makie.poly!(g, rects, color = colors)

    g
end

"""
Visualize a GridWorld.
"""
function visualize_grid(w; size=(400, 400), kwargs...)
    f = Makie.Figure(;size)
    ax = Makie.Axis(f[1, 1], aspect=Makie.DataAspect())
    Makie.hidedecorations!(ax)
    gridworldplot!(ax, w; kwargs...)
    f
end
