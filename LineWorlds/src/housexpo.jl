import JSON
using StatsBase: mean
using .Geometry: bounding_box, Segment, segments

function load_env(fname="../../data/task_inputs/example_1.json", path_idx=3)
    d = JSON.parsefile(fname)

    verts    = Vector{Vector{Float64}}(d["verts"]);
    clutter  = Vector{Vector{Vector{Float64}}}(d["clutter_verts"]);
    paths  = Vector{Vector{Vector{Float64}}}(d["paths"]);

    _segs   = segments(verts);
    _boxes  = vcat(segments.(clutter)...);
    _bb     = bounding_box(_segs)
    _center = mean(_bb);

    # Choose path
    xs = paths[path_idx]

    # Unpack path into 
    # poses and controls
    _dxs  = xs[2:end] - xs[1:end-1]
    _hds  = angle.(_dxs)
    _dhds = _hds[2:end] - _hds[1:end-1];
    _xs = xs[1:end-2];

    _ps = [Pose(x,hd) for (x,hd) in zip(_xs, _hds)];
    _us = [Control(dx,dhd) for (dx,dhd) in zip(_dxs, _dhds)]
    _T  = length(_xs);

    return (; _segs, _boxes, _bb, _center, _xs, _hds, _ps, _dxs, _dhds, _us, _T)
end

function load_env_sparse(fname::String)
    d = JSON.parsefile(fname)

    verts    = Vector{Vector{Float64}}(d["verts"]);

    _segs   = segments(verts);
    _bb     = bounding_box(_segs)
    _center = mean(_bb);

    return (; _segs, _bb, _center)
end
function load_env_sparse(foldername::String, idx::Int)
    # get a list of all the files in the folder:
    files = sort(readdir(foldername))

    # construct fname
    fname = joinpath(foldername, files[idx])

    return load_env_sparse(fname)
end