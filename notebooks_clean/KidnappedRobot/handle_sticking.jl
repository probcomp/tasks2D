function handle_sticking(prev_pf_state, prev_action, pf_state, action; ϵ=.25)
    if !isnothing(prev_pf_state)

        # Get a position from the current and previous belief state.
        pf_T = T(pf_state.traces[1])
        prev_pf_T = T(prev_pf_state.traces[1])
        prev_pos = prev_pf_state.traces[1][GenPOMDPs.state_addr(prev_pf_T)]
        pos = pf_state.traces[1][GenPOMDPs.state_addr(pf_T)]
        
        # If we are taking the same action as before, and when we tried
        # taking this action last time we didn't move,
        # we are stuck.  Try to get unstuck
        # by taking an action orthogonal to this sticking action.
        if taxi_dist(pos, prev_pos) < ϵ/10 && action == prev_action
            if prev_action in (:left, :right)
                action = rand() < .5 ? :up : :down
            else
                action = rand() < .5 ? :left : :right
            end
        end
    end

    return action
end