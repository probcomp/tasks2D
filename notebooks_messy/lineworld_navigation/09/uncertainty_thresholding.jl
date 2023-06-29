function there_is_ambiguity(pf_state, threshold=5e-5)
    tr1 = (GPF.sample_unweighted_traces(pf_state, 1)[1])
    pos = currentpos(tr1)
    
    p_different = 0.
    for (tr, p_particle) in zip(GPF.get_traces(pf_state), GPF.get_norm_weights(pf_state))
        if Geo.norm(currentpos(tr) - pos) > .5
            p_different += p_particle
        end
    end
    
    return p_different > threshold
end