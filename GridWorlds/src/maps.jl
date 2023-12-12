"""
Load maps from datasets, or from manually created specifications.
"""

using FileIO

###########################################
### Manually written map specifications ###
###########################################

_spec_1 = """
wwwwwwwwwwwwwwwwwww
w      w
w      w
w  ww  wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
w  ww  wwwwwwwwwwwwww           w      w      w
w  w                            w      w      w
w  w                            w      w      w
w  wwwwwwwwwwwwwwwwwww          w      www  www
w  ww  wwwwwwwwwwwwwww          w      www  www
w  w                                          w
w  w                                          w
w  wwwwwwwwwwwwwwwwwww          w             w
w  ww  wwwwwwwwwwwwwww          w             w
w  w                            w             w
w  w                            w             w
w  wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww   w
w                      wwwwwwwwwwwwwwwwwwww   w
w                      wwwwwwwwwwwwwwwwwwww   w
w                      w       wwwwwwwwwwww   w
wwwwwww   wwwwww       w       wwwww          w
www         www       w       wwwww          w
www         www               wwwwwwwwwwwwwwww
wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
"""

MAP_SPECS() = [
    """
    wwwwwwwwwwwww
    w     w     w
    w     w     w
    w     w     w
    w     w     w
    wwww  wwww  wwwwwww
    w                 w
    wwwwwwwwwwwwwwwwwww
    """,
    _spec_1
] # We can add more maps here as needed

"""
Load the `i`th map from the built-in library of custom maps.
"""
load_custom_map(i) = mapstr_to_gridworld(MAP_SPECS()[i])

function mapstr_to_gridworld(mapstr)
    lines = split(mapstr, '\n')
    is_filled = [
        [c == 'w' for c in line]
        for line in split(mapstr, '\n')
    ]
    xsize, ysize = length(lines), maximum(length.(lines))
    matrix = fill(true, ysize, xsize)
    for (i, line) in enumerate(is_filled)
        for (j, square) in enumerate(line)
            matrix[j, i] = square
        end
    end
    return boolmatrix_to_grid(flip_y_axis(matrix), ysize)
end
function flip_y_axis(matrix)
    return matrix[:, end:-1:1]
end

##################################
### HouseExpo PNG -> GridWorld ###
##################################

"""
Load the `i`th HouseExpo environment as a GridWorld
with `xsize` grid squares in the x direction.
"""
function load_houseexpo_gridworld(xsize, i)
    img = load_is_filled_matrix(i)
    return boolmatrix_to_grid(img, xsize)
end

function load_png(i)
    pngpath = "../data/HouseExpoPngSmall/"
    filename = readdir(pngpath)[i]
    img = load(joinpath(pngpath, filename))
    return img
end

is_filled_matrix(img) = map(x -> x < 0.9, img)
function load_is_filled_matrix(args...)
    img = load_png(args...)
    return is_filled_matrix(img)
end

################################
### Convert PNG to GridWorld ###
################################

"""
i = "Image" boolean matrix; each pixel is a 1 if it is filled
and 0 if it is empty.
px, py = pixel dimensions
gx, gy = grid dimensions
"""
function boolmatrix_to_grid(i::Matrix, (gx, gy)::Tuple)
    (px, py) = size(i)
    pix_per_grid = px/gx
    minpx(gx) = 1 + (gx - 1) * pix_per_grid |> floor |> Int
    maxpx(gx) = gx * pix_per_grid |> ceil |> Int
    is_filled(gx, gy) = any(
        i[px, py]
            for px = minpx(gx):maxpx(gx)
            for py = minpx(gy):maxpx(gy)
    )
    
    return FGridWorld(
        [
            is_filled(x, y) ? wall : empty
            for x=1:gx,
                y=1:gy
        ],
        nothing,
        (gx, gy)
    )
end
function boolmatrix_to_grid(i::Matrix, gx::Int)
    px, py = size(i)
    gy = Int(ceil(py/px * gx))
    return boolmatrix_to_grid(i, (gx, gy)::Tuple)
end