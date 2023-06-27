function _construct_hotel_env(n_rooms)
    pts = Vector{Float64}[]
    pt = [0, 0]
    for _=1:n_rooms
        newpts = build_room(pt)
        pts = vcat(pts, newpts)
        pt = newpts[end]
    end
    n = n_rooms
    # @assert pt == [0, 3n] "$pt != $([0, 5n])"
    pts = vcat(pts, [
        [0, 5n], [7, 5n], [7, 0], [6, 0]
    ])

    return pts
end
function build_room(st_pt)
    [
        [6, 0], [6, 4], [4, 4], [4, 2], [4, 4], [6, 4], [6, 0], [4, 0], [4, 1], [4, 0], [0, 0], [0, 5]
    ] .+ [st_pt]
end

function construct_hotel_env(n_rooms)
    pts = _construct_hotel_env(n_rooms)
    segs = Geo.segments(pts)

    bb = ([0, 0], [7, 5*n_rooms])

    return (segs, bb)
end