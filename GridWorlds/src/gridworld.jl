
###########################
### GridWorld Interface ### 
###########################

"""
Value which can occupy a cell in a gridworld.

A `strange` cell causes strange behavior from
the observation model.
"""
@enum GridCell empty wall strange agent

"""A world state for a 2D Grid environment."""
abstract type GridWorld end

"""The dimensions of the gridworld."""
Base.size(::GridWorld) = error("Not implemented.")
"""The value of cell (x, y)."""
Base.getindex(::GridWorld, x, y) = error("Not implemented.")
"""(x, y) position of the agent, or nothing if there is no agent."""
agentpos(::GridWorld) = error("Not implemented.")
"""Replace the value at (x, y) with cell`."""
replace(::GridWorld, (x, y), cell) = error("Not implemented.")


###########################
### GridWorld Functions ###
###########################

function Base.:(==)(a::GridWorld, b::GridWorld)
    return (
        keys(a) == keys(b) &&
        all(a[i] == b[i] for i in keys(a))
    )
end
function Base.hash(w::GridWorld, h::UInt)
    return hash((keys(w), [w[i] for i in keys(w)]), h)
end

Base.keys(w::GridWorld) = CartesianIndices(size(w))
Base.getindex(w::GridWorld, i::CartesianIndex) = w[i.I...]

"""
Move the agent from its current position to (x, y).
(If no agent is in w, this will add the agent to (x, y).)
"""
function place_agent(w::GridWorld, (x, y)::Tuple)
    if !isnothing(agentpos(w))
        w = replace(w, agentpos(w), empty)
    end
    w = replace(w, (x, y), agent)
    return w
end

"""A set of all empty cells in `w`."""
function empty_cells(w::GridWorld)
    return Set([
        (x, y)
        for x=1:size(w)[1]
        for y=1:size(w)[2]
        if w[x, y] == empty
    ])
end

"""
Return the position which would result by moving :up / :down / :left / :right / :stay.
(If heed_walls is false, allow the agent to step onto wall squares.)
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
function move_agent(w::GridWorld, direction::Symbol; heed_walls=true)
    pos = newpos(w, agentpos(w), direction; heed_walls)
    return place_agent(w, pos)
end
