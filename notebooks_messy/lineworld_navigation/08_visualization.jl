
function visual_v2(tr, states, mint=nothing, maxt=nothing;
    goal=nothing, fps=1,
    title="Generated trajectory from POMDP",
    saveas=nothing,
    minweight=0.1, # always display the color of weights as though the weight is this large
    _segs=_segs, _bb=_bb,
    disappear_at=5e-4
)
    poses = [L.Pose(position, PARAMS.orientation) for position in GenPOMDPs.state_sequence(tr)];
    observations = [obss[:obs] for obss in GenPOMDPs.observation_sequence(tr)];
    gt_ptclouds = [obs.*pose for (obs, pose) in zip(observations, poses)];

    if isnothing(mint); mint = 0; end
    if isnothing(maxt); maxt = min(T(tr), length(states)-1); end

    # -----------------------
    col = palette(:default)
    ani = Animation()
    for t=mint:maxt

        ### Ground truth agent position ###
        gt_agent_plt = plot(
            size=(500,400), aspect_ratio=:equal, title="True World State", grid=false,
            xlim=(_bb[1][1]-3, _bb[2][1]+3),
            ylim=(_bb[1][2]-5, _bb[2][2]+1),
            legend=:bottomleft,
            axis=([], false)
        )

        ys = gt_ptclouds[t+1]
        p = poses[t+1]

        # Walls
        plot!(_segs, c=:black, linewidth=1, label=nothing)
        # Observations
        scatter!(ys, c=col[1], markersize=2, alpha=.7, markerstrokewidth=0, label="Observed distances from LIDAR")
        for y in ys # lines from agent to obs

            # TODO: the fact we need this second check shows something is going
            # wrong with the observations...
            if !any(isinf.(y)) && y[1] ≤ _bb[2][1] && y[2] < _bb[2][2]
                plot!([p.x, y], c=col[1], alpha=0.1, label=nothing)
            end
        end

        # Agent
        scatter!([p.x], c=:red, marker=:square, label="True agent position")

        if !isnothing(goal)
            scatter!([goal], c=:green, marker=:x, label="Goal") 
        end

        ### Agent Belief ###
        agent_belief_plt = plot(
            size=(500,400), aspect_ratio=:equal, title="Agent Beliefs", grid=false,
            xlim=(_bb[1][1]-3, _bb[2][1]+3),
            ylim=(_bb[1][2]-5, _bb[2][2]+1),
            legend=:bottomleft,
            axis=([], false)
        )

        # Walls
        plot!(_segs, c=:black, linewidth=1, label=nothing)

        # Particles...
        particles = states[t+1]
        positions = [] # Particle positions we have plotted observations relative to at on this frame
        has_labeled_obs = false
        has_labeled_pos = false
        for (w, tr) in zip(GPF.get_norm_weights(particles), GPF.get_traces(particles))
            pos = tr[GenPOMDPs.state_addr(t)]

            # If this isn't close to any other particle we've plotted observations
            # relative to, plot the observations relative to it
            if !any(Geo.norm(p - pos) < .5 for p in positions) && w > disappear_at
                pose = L.Pose(pos, PARAMS.orientation)
                relative_obs = observations[t+1].*pose

                label = has_labeled_obs ? nothing : "LIDAR dists rel. to belief"
                !isnothing(label) && (has_labeled_obs = true)
                scatter!(relative_obs, c=col[1], markersize=2, alpha=.4, markerstrokewidth=0, label=label)
                for o in relative_obs
                    if !any(isinf.(o)) && o[1] ≤ _bb[2][1] && o[2] < _bb[2][2]
                        plot!([pose.x, o], c=col[1], alpha=0.1, label=nothing)
                    end
                end

                push!(positions, pos)
            end

            label = sqrt(w) > .1 && !has_labeled_pos ? "Belief: possible agent location" : nothing
            !isnothing(label) && (has_labeled_pos = true)
            if w > disappear_at
                scatter!([pos], c=:black, marker=:square, seriesalpha=max(sqrt(w), minweight), label=label)
            end
        end # for - particle

        if !isnothing(goal)
            scatter!([goal], c=:green, marker=:x, label="Goal") 
        end

        frame(ani, plot(gt_agent_plt, agent_belief_plt))
    end # for - t

    if !isnothing(saveas)
        return gif(ani, "$saveas.gif", fps=fps) # gif(ani, fname, fps=10)
    else
        return gif(ani, fps=fps)
    end
end # function

function visualize_tr_pf(tr, states, mint=nothing, maxt=nothing; goal=nothing, fps=1,
    title="Generated trajectory from POMDP",
    saveas=nothing,
    minweight=0.1
)
    poses = [L.Pose(position, PARAMS.orientation) for position in GenPOMDPs.state_sequence(tr)];
    ptclouds = [
        obss[:obs].*pose
        for (obss, pose) in zip(
            GenPOMDPs.observation_sequence(tr),
            poses
        )
    ];

    if isnothing(mint)
        mint = 0
    end
    if isnothing(maxt)
        maxt = min(T(tr), length(states)-1)
    end
    
    # -----------------------
    col = palette(:default)
    ani = Animation()
    # states = [state]
    for t=mint:maxt
        p = poses[t + 1]
        y = ptclouds[t + 1]
    
        agent_plt = plot(
            size=(500,500), aspect_ratio=:equal, title=title, grid=false,
            xlim=(_bb[1][1]-3, _bb[2][1]+3),
            ylim=(_bb[1][2]-3, _bb[2][2]+3),
            legend=:bottomleft,
        )
        plot!(_segs, c=:black, linewidth=1, label=nothing)
#         plot!([p], c=:red, r=1.0, linewidth=2, label=nothing)
        scatter!(Random.shuffle(y[1:2:end]), c=col[1], markersize=4, alpha=.7, markerstrokewidth=1, label="Obs in pose coords `_ys[t].*_ps[t]`")
    
        particles = states[t + 1]
        labeled=false
        for (w, tr) in zip(
                GenParticleFilters.get_norm_weights(particles),
                GenParticleFilters.get_traces(particles)
            )
            pos = tr[GenPOMDPs.state_addr(t)]
            if !labeled && sqrt(w) > 0.1
                scatter!([pos], c=:black, seriesalpha=max(sqrt(w), minweight), label="Particle")
                labeled=true
            else
                scatter!([pos], c=:black, seriesalpha=max(sqrt(w), minweight), label=nothing)
            end
        end
        
        if !isnothing(goal)
           scatter!([goal], c=:green, marker=:x, label="Goal") 
        end
        
        scatter!([p.x], c=:red, marker=:x, label="True agent position")
    
        annotate!(_bb[1][1]-1, _bb[2][2]+1, text("t=$t", :black, :right, 12))

        frame(ani, agent_plt)
    end
    if !isnothing(saveas)
        return gif(ani, "$saveas.gif", fps=fps) # gif(ani, fname, fps=10)
    else
        return gif(ani, fps=fps)
    end
end