######################
# Action controllers #
######################
@gen function _random_controller(st, obs)
    aidx ~ categorical([0.25, 0.25, 0.25, 0.25])
    action = [:up, :right, :left, :down][aidx]
    return (action, nothing)
end
random_controller = GenPOMDPs.Controller(_random_controller, nothing)

@gen function _meandering_controller(current_direction, obs)
    change_dir ~ bernoulli(0.2)
    if change_dir || isnothing(current_direction)
        aidx ~ categorical([0.25, 0.25, 0.25, 0.25])
        action = [:up, :right, :left, :down][aidx]
    else
        action = current_direction
    end
    return (action, action)
end
meandering_controller = GenPOMDPs.Controller(_meandering_controller, nothing)

@gen function _meandering_wallavoiding_controller(current_direction, obs)
    # d, r, u, l
    directional_dist_inds = [Int(floor(i)) for i in LinRange(1, length(obs) + 1, 5)[1:4]]
    relevant_dists = obs[directional_dist_inds]
    near_walls = Dict()
    near_walls[:down] = relevant_dists[1] < 1
    near_walls[:right] = relevant_dists[2] < 1
    near_walls[:up] = relevant_dists[3] < 1
    near_walls[:left] = relevant_dists[4] < 1

    if near_walls[:down]
        action = :up
    elseif near_walls[:right]
        action = :left
    elseif near_walls[:up]
        action = :down
    elseif near_walls[:left]
        action = :right
    else
        action = nothing
    end

    if isnothing(action)
        change_dir ~ bernoulli(0.2)
        if change_dir || isnothing(current_direction)
            aidx ~ categorical([0.25, 0.25, 0.25, 0.25])
            action = [:up, :right, :left, :down][aidx]
        else
            action = current_direction
        end
    end

    i = 0
    while near_walls[action]
        # println("near_walls[$action]")
        if i > 20
            break
        end
        i += 1
        aidx = {(:aidx, i)} ~ categorical([0.25, 0.25, 0.25, 0.25])
        action = [:up, :right, :left, :down][aidx]
    end
    # println("near walls: $(near_walls[action])")

    return (action, action)
end
meandering_wallavoiding_controller = GenPOMDPs.Controller(_meandering_wallavoiding_controller, nothing)