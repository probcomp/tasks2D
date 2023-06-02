"""
Load HouseExpo data, and convert HouseExpo PNG -> GridWorld
"""

using FileIO

"""
Load the `i`th HouseExpo environment as a GridWorld
with `xsize` grid squares in the x direction.
"""
function load_gridworld(xsize, i)
    img = load_is_filled_matrix(i)
    return boolmatrix_to_grid(img, xsize)
end

function load_png(i)
    pngpath = "../HouseExpoPng/"
    filename = readdir(pngpath)[i]
    img = load(joinpath(pngpath, filename))
    return img
end

is_filled_matrix(img) = map(x -> x < 0.9, img)
function load_is_filled_matrix(args...)
    img = load_png(args...)
    return is_filled_matrix(img)
end

### Convert PNG to GridWorld ###

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