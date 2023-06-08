const _cuda = Ref(false)
function __init__()
    _cuda[] = CUDA.functional()
end;
DeviceArray(x) = _cuda[] ? CuArray(x) : Array(x);

export _cuda, DeviceArray

"""
    griddims = cuda_grid(datadims::Tuple{Vararg{Int}},
                         blockdims::Tuple{Vararg{Int}})

Given data dimensions `datadims` and number of threads
in each dimension `blockdims` returns the respective
grid dimensions `griddims` such that

    griddims[i] = ceil(Int, datadims[i]/blockdims[i])

"""
function cuda_grid(datadims::Tuple{Vararg{Int}}, blockdims::Tuple{Vararg{Int}})
    griddims = ceil.(Int, datadims./blockdims)
    return griddims
end