function get_sensor_args(pos, params, segs)
    p = L.Pose(pos, params.obs.orientation)
    w, s_noise, outlier, outlier_vol, zmax = params.obs.sensor_args
    _as = L.create_angles(params.obs.fov, params.obs.n_rays)
    zs = L.cast([p], segs; num_a=params.obs.n_rays, zmax)
    ỹ  = L.get_2d_mixture_components(zs, _as, w; fill_val_z=zmax)[1,:,:,:]

    return (ỹ, s_noise, outlier, outlier_vol)
end
