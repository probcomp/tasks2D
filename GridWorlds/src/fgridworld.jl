using FunctionalCollections

"""Functional implementation of a GridWorld."""
struct FGridWorld <: GridWorld
    cells::PersistentVector{PersistentVector{GridCell}}
    agentpos::Union{Nothing, Tuple{Int, Int}}
    size::Tuple{Int, Int}
end
function FGridWorld(g::Matrix{GridCell}, pos::Union{Nothing, Tuple{Int, Int}}, size::Tuple{Int, Int})
    FGridWorld(
        pvec([pvec([g[x, y] for y=1:size[2]]) for x=1:size[1]]),
        pos,
        size
    )
end
Base.size(w::FGridWorld) = w.size
Base.getindex(w::FGridWorld, x, y) = w.cells[x][y]
agentpos(w::FGridWorld) = w.agentpos
function replace(w::FGridWorld, (x, y)::Tuple{Int, Int}, s)
    agent_pos =
      s == agent            ? (x, y)  :
      agentpos(w) == (x, y) ? nothing : 
                              agentpos(w)

    return FGridWorld(
        assoc(w.cells, x, assoc(w.cells[x], y, s)),
        agent_pos,
        w.size
    )
end