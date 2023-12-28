import LineWorlds
L = LineWorlds
using GenSMCP3
import GenTraceKernelDSL
mvuniform = L.ProductDistribution(uniform);

GenSMCP3.@kernel function fwd_iterated_grid_init(
    tr, trajectory_model, params, obs, t0_grid_args,
    pos_to_init_cm
)
    pos = t0_grid_args.initial_pos
    j, grid_args, grid_proposed_positions = nothing, nothing, nothing # make these available outside loop scope
    for i = 1:t0_grid_args.n_iters
        grid_args = L.grid_schedule(t0_grid_args.init_grid_args, i)
        grid_proposed_positions, _ = L.vector_grid(GenSMCP3.GenTraceKernelDSL.DynamicForwardDiff.value(pos), grid_args...)
        grid_final_positions = reshape(grid_proposed_positions, (:,))

        pos_choicemap_grid = [pos_to_init_cm(pos) for pos in grid_final_positions]
        full_cm_grid = [
            Gen.merge(
                GenPOMDPs.nest_choicemap(pos_choicemap_grid[j], GenPOMDPs.state_addr(0)),
                GenPOMDPs.nest_choicemap(obs, GenPOMDPs.obs_addr(0))
            )
            for j in 1:length(grid_final_positions)
        ]
        wts = [Gen.generate(trajectory_model, (0, [], params), cm)[2] for cm in full_cm_grid]
        wts = L.normalize_exp(wts)
        j = {(:j, i)} ~ categorical(wts)
        pos = grid_final_positions[j]
    end
    vs = grid_proposed_positions
    x′  = {:pos}  ~ mvuniform(vs[j][1:2] -  grid_args.r[1:2]/2, vs[j][1:2] + grid_args.r[1:2]/2)

    return (
        GenPOMDPs.nest_choicemap(pos_to_init_cm(x′), GenPOMDPs.state_addr(0)),
        EmptyChoiceMap() # Backward proposal is deterministic.
    )
end
GenSMCP3.@kernel function bwd_iterated_grid_init(tr, t0_grid_args, state_to_pos)
    fwd_ch = choicemap()
    fwd_ch[:pos] = state_to_pos(tr[GenPOMDPs.state_addr(0)])

    pos = t0_grid_args.initial_pos
    for i=t0_grid_args.n_iters:1
        args = L.grid_schedule(t0_grid_args.init_grid_args, i)
        j = L.grid_index(nexpos, pos, args...; linear=true)
        fwd_ch[(:j, i)] = j
    end

    return (EmptyChoiceMap(), fwd_ch)
end

function default_pf_args(PARAMS; n_particles=1)
    # Do some math to figure out the bounding box center,
    # and a grid schedule which will overlay roughly 400 grid points
    # over the whole grid.  We will use
    # this grid schedule for the coarsest level of the coarse-to-fine
    # proposals at the initialization step.
    avg(x, y) = (x+y)/2
    _bb = ((0, 0), Tuple(PARAMS.map.size))
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

    return (t0_grid_args, update_grid_args, resampling_args)
end