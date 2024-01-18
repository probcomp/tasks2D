function pf_result_gif(tr, pfstates;
    framerate=4, n_frames=length(pfstates) - 1, n_panels=3,
    tail_length=8, labeltext="", label_fontsize=16,
    filename="tmp", show_lines_to_walls=true    
)
    gif_filename = GridWorlds.Viz.pf_result_gif(
        get_args(tr)[end].params.map,
        fr -> [st.pos for st in GenPOMDPs.state_sequence(tr)[1:fr]],
        fr -> [
            [st.pos for st in GenPOMDPs.state_sequence(trace)]
            for trace in get_traces(pfstates[fr])
        ],
        fr -> get_log_weights(pfstates[fr]);
        fr_to_gt_obs = fr -> GenPOMDPs.observation_retval_sequence(tr)[fr],
        fr_to_particle_obss = fr -> [
            GenPOMDPs.observation_retval_sequence(trace)[end]
            for trace in get_traces(pfstates[fr])
        ],
        framerate, n_frames, n_panels, tail_length,
        labeltext, label_fontsize, filename, show_lines_to_walls
    )
    println("Saved gif to $gif_filename")
    display("image/gif", read(gif_filename))
end
function pf_result_gif(ipomdp_tr::Gen.Trace; kwargs...)
    pf_states = [st.pfstate for st in GenPOMDPs.controllerstate_sequence(ipomdp_tr)]
    return pf_result_gif(ipomdp_tr, pf_states; kwargs...)
end

function trace_gif(trace::Gen.Trace;
    framerate=4,    
    kwargs...
)
    gif_filename = GridWorlds.Viz.trace_gif(
        t -> get_args(trace)[end].map,
        [st.pos for st in GenPOMDPs.state_sequence(trace)],
        GenPOMDPs.observation_retval_sequence(trace);
        framerate, kwargs...
    )
    println("Saved gif to $gif_filename")
    display("image/gif", read(gif_filename))
end