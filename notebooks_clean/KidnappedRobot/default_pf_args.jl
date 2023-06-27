function default_pf_args(PARAMS; n_particles)
    # Do some math to figure out the bounding box center,
    # and a grid schedule which will overlay roughly 400 grid points
    # over the whole grid.  We will use
    # this grid schedule for the coarsest level of the coarse-to-fine
    # proposals at the initialization step.
    avg(x, y) = (x+y)/2
    _bb = PARAMS.bounding_box
    center = [avg(_bb[1][1], _bb[2][1]), avg(_bb[1][2], _bb[2][2])]
    Δy = _bb[2][2] - _bb[1][2]
    Δx = _bb[2][1] - _bb[1][1]
    y_to_x = Δy/Δx
    step = sqrt(Δx^2 * y_to_x / 400)
    @assert Δy/step * Δx/step ≈ 400
    nstepsy = Int(floor(Δy/step))
    nstepsx = Int(floor(Δx/step));
    @assert nstepsy*nstepsx<500

    # 3 iters of coarse-to-fine
    update_grid_args = (;
        tau = 1., pmin = 1e-6, n_iters = 3,
        init_grid_args = (; k = [9, 9], r = [PARAMS.step.Δ/6, PARAMS.step.Δ/6]),
    )

    # 6 iters of coarse to fine, since we have to start at a much
    # coarser level to scan over the whole environment (not just the regions
    # where the agent could have stepped in the last timestep)
    t0_grid_args = (;
        update_grid_args..., tau=1., n_iters=6,
        init_grid_args = (; k=[nstepsx, nstepsy], r = [step, step]), initial_pos = center
    )

    resampling_args = (; n_particles, n_groups=1, ess_threshold=0.)

    return (update_grid_args, t0_grid_args, resampling_args)
end

function overwrite_params(params; p_kidnapped, step, sensor_args)
    return (;
        params...,
        p_kidnapped,
        step = (;
            params.step...,
            step...
        ),
        obs = (;
            params.obs...,
            sensor_args = (;
                params.obs.sensor_args...,
                sensor_args...
            )
        )
    )
end