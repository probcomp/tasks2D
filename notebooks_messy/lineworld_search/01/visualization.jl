using Plots

function trace_to_path_image(tr;
    kidnapped_at=[],
    _bb=bounding_box,
    walls=walls
)
    params = get_args(tr)[2]
    target_room = tr[state_addr(0, :target_room)]
    goalobj = segments_for_object_in(target_room, params.target_pose_relative_to_room)

    mypl = plot(size=(500,500), aspect_ratio=:equal, title="Agent's path", grid=false,
        xlim=(_bb[1][1]-3, _bb[2][1]+3),
        ylim=(_bb[1][2]-3, _bb[2][2]+3),
        legend=:bottomleft,
    )
    plot!(walls, c=:black, label="Walls")
    if !isnothing(goalobj)
        plot!(goalobj, c=:green, label="Goal")
    end

    positions = map(x->x[1], GenPOMDPs.state_sequence(tr))
    if isempty(kidnapped_at)
        plot!(positions, c=:red, label="Agent path")
    else
        t1 = first(kidnapped_at)
        plot!(positions[1:t1], c=:red, label="Agent path")
        for t in kidnapped_at[2:end]
            plot!(positions[t1+1:t], c=:red)
            t1 = t
        end
        if t1 + 1 == length(positions)
            plot!([positions[end], positions[end]+[.1,.1]], c=:red, label=nothing)
        else
            plot!(positions[t1+1:end], c=:red, label=nothing)
        end
    end

    return mypl
end

function trace_to_gif(tr, args...; kwargs...)
    # Extract the PF states from tr
    pf_states = [
        controller_state[1]
        for (action, controller_state) in GenPOMDPs.control_sequence(tr)
    ]

    return trace_and_pfstate_to_gif(tr, pf_states, args...; kwargs...)
end

function trace_and_pfstate_to_gif(tr, states, mint=nothing, maxt=nothing;
    fps=1,
    kidnapped_at=[],
    N_BUFFER_STEPS=10,
    N_KIDNAP_STEPS=20,
    saveas=nothing,
    title=nothing,
    min_display_weight=0.1, # always display the color of weights as though the weight is this large
    walls=walls, _bb=bounding_box,
    disappear_at=5e-4
)
    params = get_args(tr)[2]
    target_room = tr[state_addr(0, :target_room)]
    goalobj = segments_for_object_in(target_room, params.target_pose_relative_to_room)

    poses = [L.Pose(position, PARAMS.obs.orientation) for (position, _) in GenPOMDPs.state_sequence(tr)];
    observations = [obss[:obs] for obss in GenPOMDPs.observation_sequence(tr)];
    gt_ptclouds = [obs.*pose for (obs, pose) in zip(observations, poses)];

    if isnothing(mint); mint = 0; end
    if isnothing(maxt); maxt = min(T(tr), length(states)-1); end

    # -----------------------
    col = palette(:default)
    ani = Animation()
    for t=mint:maxt
        # if t in kidnapped_at
            
        #     gt_agent_plt = plot(
        #         size=(500,400), aspect_ratio=:equal, title="True World State", grid=false,
        #         xlim=(_bb[1][1]-3, _bb[2][1]+3),
        #         ylim=(_bb[1][2]-5, _bb[2][2]+1),
        #         legend=:bottomleft,
        #         axis=([], false)
        #     )
        #     agent_belief_plt = plot(
        #         size=(500,400), aspect_ratio=:equal, title="Agent Beliefs", grid=false,
        #         xlim=(_bb[1][1]-3, _bb[2][1]+3),
        #         ylim=(_bb[1][2]-5, _bb[2][2]+1),
        #         legend=:bottomleft,
        #         axis=([], false)
        #     )
    
        #     final_plot = isnothing(title) ? plot(gt_agent_plt, agent_belief_plt) : plot(gt_agent_plt, agent_belief_plt; title)
        #     frame(ani, final_plot)    
        # end

        for step in ((t in kidnapped_at) ? (1:(N_KIDNAP_STEPS + 2*N_BUFFER_STEPS)) : (1:1))

        ### Ground truth agent position ###
        gt_title =(t in kidnapped_at) ? "" : "True World State"
        gt_agent_plt = plot(
            size=(500,400), aspect_ratio=:equal, title=gt_title, grid=false,
            xlim=(_bb[1][1]-3, _bb[2][1]+3),
            ylim=(_bb[1][2]-5, _bb[2][2]+1),
            legend=:bottomleft,
            axis=([], false)
        )

        ys = gt_ptclouds[t+1]
        p = poses[t+1]

        # Walls
        plot!(walls, c=:black, linewidth=1, label=nothing)

        if t in kidnapped_at
            if step <= N_BUFFER_STEPS
                scatter!([poses[t].x], c=:red, marker=:square, label="Agent position")
                nothing
            elseif N_BUFFER_STEPS < step <= N_KIDNAP_STEPS + N_BUFFER_STEPS
                st = step - N_BUFFER_STEPS
                change_in_pos = poses[t + 1].x - poses[t].x
                pos = poses[t].x + st/N_KIDNAP_STEPS * change_in_pos
                scatter!([pos], c=:red, marker=:square, label="Agent [being moved manually]")
            else
                scatter!([poses[t+1].x], c=:red, marker=:square, label="Agent position")
            end
        else
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
        end # else - not a kidnapping step

        if !isnothing(goalobj)
            plot!(goalobj, c=:green, label="Goal") 
        end

        ### Agent Belief ###
        blf_title = (t in kidnapped_at) ? (
            if step <= N_BUFFER_STEPS
                "[Robot powering down...]"
            elseif N_BUFFER_STEPS < step <= (N_BUFFER_STEPS + N_KIDNAP_STEPS)
                "[Robot powered off]"
            else
                "[Robot booting up...]"
            end
        ) : "Agent Belief"
        agent_belief_plt = plot(
            size=(500,400), aspect_ratio=:equal, title=blf_title, grid=false,
            xlim=(_bb[1][1]-3, _bb[2][1]+3),
            ylim=(_bb[1][2]-5, _bb[2][2]+1),
            legend=:bottomleft,
            axis=([], false)
        )

        #=
            To add:
            [x] Both particles [location]
            [] "Object in room" [visualize green box at location of object, in the room]
            [] "Object not in room" [visualize red outline of room without object]
        =#

        # Walls
        plot!(walls, c=:black, linewidth=1, label=nothing)

        particles = states[t+1]
        positions = [] # Particle positions we have plotted observations relative to at on this frame
        has_labeled_obs = false
        has_labeled_pos = false
        has_labeled_notinroom = false
        has_labeled_inroom = false
        if t in kidnapped_at
            particles = states[t]
            for (w, tr) in zip(GPF.get_norm_weights(particles), GPF.get_traces(particles))
                pf_t = get_args(tr)[1]
                pos = tr[GenPOMDPs.state_addr(pf_t, :pos)]
                                label = sqrt(w) > .1 && !has_labeled_pos ? "Belief: possible agent location" : nothing
                !isnothing(label) && (has_labeled_pos = true)
                if w > disappear_at
                    scatter!([pos], c=:black, marker=:square, seriesalpha=max(sqrt(w), min_display_weight), label=label)
                end
            end
        else
            for (w, tr) in zip(GPF.get_norm_weights(particles), GPF.get_traces(particles))
                # println("T(tr) = $(T(tr)) ; GenPOMDPs.state_addr(t) = $(GenPOMDPs.state_addr(t))")
                pf_t = get_args(tr)[1]
                pos = tr[GenPOMDPs.state_addr(pf_t, :pos)]

                # If this isn't close to any other particle we've plotted observations
                # relative to, plot the observations relative to it
                if !any(Geo.norm(p - pos) < .5 for p in positions) && w > disappear_at
                    pose = L.Pose(pos, PARAMS.obs.orientation)
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
                    scatter!([pos], c=:black, marker=:square, seriesalpha=max(sqrt(w), min_display_weight), label=label)
                end

                ### Visualize belief about object location ###
                params = get_args(tr)[3]
                room_occupancies = GenPOMDPs.state_sequence(tr)[end][2]
                for (room, occ) in pairs(room_occupancies)
                    if occ
                        segs = segments_for_object_in(room, params.target_pose_relative_to_room)
                        lbl = has_labeled_inroom ? nothing : "Belief: object in room"
                        plot!(segs, c=:green, linestyle=:dash, linealpha=max(sqrt(w), min_display_weight), linewidth=2, label=lbl)
                        has_labeled_inroom = true
                    else
                        segs = rect_to_segs(room...)
                        lbl = has_labeled_notinroom ? nothing : "Belief: object not in room"
                        plot!(segs, c=:blue, linestyle=:dash, linealpha=max(sqrt(w), min_display_weight), linewidth=2, label=lbl)
                        has_labeled_notinroom = true
                    end
                end

            end # for - particle
        end

        # if !isnothing(goalobj)
        #     plot!(goalobj, c=:green, label="Goal") 
        # end

        final_plot = isnothing(title) ? plot(gt_agent_plt, agent_belief_plt) : plot(gt_agent_plt, agent_belief_plt; title)
        frame(ani, final_plot)
    
    end # for - step
    
    end # for - t

    if !isnothing(saveas)
        return gif(ani, "$saveas.gif", fps=fps) # gif(ani, fname, fps=10)
    else
        return gif(ani, fps=fps)
    end
end # function