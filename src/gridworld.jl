using FunctionalCollections

@enum GridSquare empty wall agent

"""A world state for a 2D Grid environment."""
abstract type GridWorld end

"""Interface for GridWorld."""
Base.size(::GridWorld) = error("Not implemented.")
Base.getindex(::GridWorld, x, y) = error("Not implemented.")
agentpos(::GridWorld) = error("Not implemented.")
replace(::GridWorld, x, y, square) = error("Not implemented.")

Base.keys(w::GridWorld) = CartesianIndices(size(w))
Base.getindex(w::GridWorld, i::CartesianIndex) = w[i.I...]

"""
Move the agent from its current position to (x, y).
(If no agent is in w, this will add the agent to (x, y).)
"""
function moveagent(w::GridWorld, (x, y)::Tuple)
    if !isnothing(agentpos(w))
        w = replace(w, agentpos(w)..., empty)
    end
    w = replace(w, x, y, agent)
    return w
end

### Motion up/down/left/right ###

"""
Return the position which would result by moving :up / :down / :left / :right / :stay.
(If heed_walls is false, allow the agent to step onto wall squares..)
"""
function newpos(w::GridWorld, pos, a::Symbol; heed_walls=true)

    (x, y) = pos
    proposed_pos = (
        a == :up    ? (x, y + 1) :
        a == :down  ? (x, y - 1) :
        a == :left  ? (x - 1, y) :
        a == :right ? (x + 1, y) :
        a == :stay  ?   (x, y)   :
        error("Unrecognized action: $a [should be in [:up, :down, :left, :right, :stay]]")
    )
    truncated_position = (
        min(max(proposed_pos[1], 1), size(w)[1]),
        min(max(proposed_pos[2], 1), size(w)[2])
    )

    if heed_walls && w[truncated_position...] == wall
        return pos
    end

    return truncated_position
end

"""
Move the agent up, down, left, or right by one square.
If `heed_walls` is true, then the agent will not move into a wall
(and a move which would have moved the agent into a wall will
leave the agent in its current position).
"""
function moveagent(w::GridWorld, direction::Symbol; heed_walls=true)
    pos = newpos(w, agentpos(w), direction; heed_walls)
    return moveagent(w, pos)
end

"""Return a set of all empty squares in `w`."""
function empty_cells(w::GridWorld)
    return Set([
        (x, y)
        for x=1:size(w)[1]
        for y=1:size(w)[2]
        if w[x, y] == empty
    ])
end

"""Choose a random empty square in `w`, and replace it with `s`."""
function add_to_random_empty_square(w::GridWorld, s::GridSquare)
    empty_squares = [
        (x, y)
        for x=1:size(w)[1]
        for y=1:size(w)[2]
        if w[x, y] == empty
    ]
    (x, y) = empty_squares[rand(1:length(empty_squares))]
    return replace(w, x, y, s)
end

"""Functional implementation of a GridWorld."""
struct FGridWorld <: GridWorld
    squares::PersistentVector{PersistentVector{GridSquare}}
    agentpos::Union{Nothing, Tuple{Int, Int}}
    size::Tuple{Int, Int}
end
function FGridWorld(g::Matrix{GridSquare}, pos::Union{Nothing, Tuple{Int, Int}}, size::Tuple{Int, Int})
    FGridWorld(
        pvec([pvec([g[x, y] for y=1:size[2]]) for x=1:size[1]]),
        pos,
        size
    )
end
Base.size(w::FGridWorld) = w.size
Base.getindex(w::FGridWorld, x, y) = w.squares[x][y]
agentpos(w::FGridWorld) = w.agentpos
function replace(w::FGridWorld, x::Int, y::Int, s)
    agent_pos =
      s == agent            ? (x, y)  :
      agentpos(w) == (x, y) ? nothing : 
                              agentpos(w)

    return FGridWorld(
        assoc(w.squares, x, assoc(w.squares[x], y, s)),
        agent_pos,
        w.size
    )
end