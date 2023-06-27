function box_segments(goal, sidelength=1.)
    (x, y) = goal
    δ = sidelength/2
    return [ # square
        L.Segment([x - δ, y - δ, x + δ, y - δ]),
        L.Segment([x + δ, y - δ, x + δ, y + δ]),
        L.Segment([x + δ, y + δ, x - δ, y + δ]),
        L.Segment([x - δ, y + δ, x - δ, y - δ])
    ]
end