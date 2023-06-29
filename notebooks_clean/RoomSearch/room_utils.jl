function rect_to_segs((b, l), (w, h))
    return L.Segment.(Vector{Float64}[
        [b, l, b + w, l],
        [b + w, l, b + w, l + h],
        [b + w, l + h, b, l + h],
        [b, l + h, b, l]
    ])
end

# Code for the agent's reasoning about room room_placements

target_object_placed(room_placements) = any(values(room_placements))
get_target_room(room_placements) = findfirst(room_placements)

function in_room(pos, room)
    (xmin, ymin), (w, h) = room
    xmax, ymax = xmin + w, ymin + h
    x, y = pos
    return xmin <= x <= xmax && ymin <= y <= ymax
end
function room_containing(pos, params)
    for room in params.all_rooms
        if in_room(pos, room)
            return room
        end
    end
    error("Position $pos is not in any room")
end
function room_center(room)
    (xmin, ymin), (w, h) = room
    return (xmin + w/2, ymin + h/2)
end
function taxi_dist_to_room_center(pos, room)
    (x, y) = pos
    (cx, cy) = room_center(room)
    return abs(x - cx) + abs(y - cy)
end
function nearest_unexplored_room(pos, params, room_placements)
    nearest_room = nothing
    nearest_dist = Inf
    for room in params.viable_rooms
        if haskey(room_placements, room)
            continue
        end
        dist = taxi_dist_to_room_center(pos, room)
        if dist < nearest_dist
            nearest_room = room
            nearest_dist = dist
        end
    end
    return nearest_room
end

function object_pos(room_placements, target_relative_to_pos)
    target_room = get_target_room(room_placements)
    if isnothing(target_room)
        return nothing
    end
    (xmin, ymin), (w, h) = target_room
    (x, y) = target_relative_to_pos
    return (xmin + x, ymin + y)
end

# Code for getting the segments corresponding to the target object
function segments_for_object_in(target_room, target_pose_relative_to_room; side_length=.8)
    (l, b) = target_room.bl
    (δx, δy) = target_pose_relative_to_room
    
    # Target object is a square
    return rect_to_segs((l + δx, b + δy), (side_length, side_length))
end

object_pos