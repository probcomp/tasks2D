using VoxelRayTracers # For lineworld -> gridworld

"""
st  - finite state machine state (direction of wall we're following)
nbs - neighbors of the agent on grid which are filled (set of dirs)
returns (action, next_st)

dirs = [:L, :R, :U, :D]
"""
function _wall_follow(st, nbs)
    isnothing(st) && :L ∉ nbs ? (:L, st) :
    isnothing(st) && :L ∈ nbs ? (:D, :L) :
    #
    st == :L && :L ∉ nbs ? (:L, :U) :
    st == :L && :D ∉ nbs ? (:D, :L) :
    st == :L && :R ∉ nbs ? (:R, :D) :
    st == :L             ? (:U, :R) :
    #
    st == :R && :R ∉ nbs ? (:R, :D) :
    st == :R && :U ∉ nbs ? (:U, :R) :
    st == :R && :L ∉ nbs ? (:L, :U) :
    st == :R             ? (:D, :L) :
    #
    st == :D && :D ∉ nbs ? (:D, :L) :
    st == :D && :R ∉ nbs ? (:R, :D) :
    st == :D && :U ∉ nbs ? (:U, :R) :
    st == :D             ? (:L, :U) :
    #
    st == :U && :U ∉ nbs ? (:U, :R) :
    st == :U && :L ∉ nbs ? (:L, :U) :
    st == :U && :D ∉ nbs ? (:D, :L) :
    st == :U             ? (:R, :D) :
    
    error("Unrecognized st/nbs pair.")
end

initial_wall_follow_state() = Any[nothing]
function wall_follow_from_pos(pos, st)
    (x, y) = pos

    # Find which sides of the agent have walls
    # (the agent's "neighbors")
    nbs = Set()
    δ = 1.25*ϵ
    for (a, newpos) in (
        (:U, [x, y + δ]), (:D, [x, y - δ]), (:L, [x - δ, y]), (:R, [x + δ, y])
    )
        if handle_wall_intersection([x, y], newpos, PARAMS.map) != newpos
            push!(nbs, a)
        end
    end
    
    (a, st) = _wall_follow(st, nbs)
    
    act = (;U=:up,L=:left,D=:down,R=:right)[a]
    
    return (act, st)

end
function wall_follow(sts, pos)
    (a, st) = wall_follow_from_pos(pos, sts[end])
    sts = vcat(sts, [st])
    
    # Prevent cycles
    if length(sts)>8 && sts[end-3:end] == sts[end-7:end-4] && length(Set(sts[end-3:end])) == 4
        sts[end] = nothing
    end
    
    return (a, sts)
end
