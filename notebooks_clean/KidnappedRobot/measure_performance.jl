function near_goal(tr)
    return taxi_dist(GOAL, tr[GenPOMDPs.state_addr(get_args(tr)[1])]) < 1.
end
function T_near_goal(tr)
    T = get_args(tr)[1]
    while T > 0
        T -= 1
        if taxi_dist(GOAL, tr[GenPOMDPs.state_addr(T)]) > 1.
            return T + 1
        end
    end
    return 500
end
function take_measurement(model)
    tr = Gen.generate(model, (0, PARAMS), 
        choicemap((GenPOMDPs.state_addr(0, :pos), INITIAL_POS))
    )[1]
    
    elapsed = 0
    T = 0
    while  T < 500
        T += 10
        t1 = time()
        tr = Gen.update(tr, (T, PARAMS), (UnknownChange(), NoChange()), EmptyChoiceMap())[1]
        t2 = time()

        if near_goal(tr)
            wasted_steps = T - T_near_goal(tr)
            elapsed += (t2 - t1) * (10 - wasted_steps)/10
            break
        else
            elapsed += (t2 - t1)
        end
    end
    return (T_near_goal(tr), elapsed/T_near_goal(tr))
end

function take_measurement_KR(model)
    tr = Gen.generate(model, (0, PARAMS), 
        choicemap((GenPOMDPs.state_addr(0, :pos), INITIAL_POS))
    )[1]
    
    elapsed = 0
    t1 = time()
    tr = Gen.update(tr, (40, PARAMS), (UnknownChange(), NoChange()), EmptyChoiceMap())[1]
    t2 = time()

    elapsed += t2 - t1

    tr, _ = Gen.update(tr, (41, PARAMS), (UnknownChange(), NoChange()),
        choicemap((GenPOMDPs.state_addr(41, :is_kidnapped), true), (GenPOMDPs.state_addr(41, :pos), [5., 16.]))
    );

    T = 41

    while  T < 501
        T += 10
        t1 = time()
        tr = Gen.update(tr, (T, PARAMS), (UnknownChange(), NoChange()), EmptyChoiceMap())[1]
        t2 = time()

        if near_goal(tr)
            wasted_steps = T - T_near_goal(tr)
            elapsed += (t2 - t1) * (10 - wasted_steps)/10
            break
        else
            elapsed += (t2 - t1)
        end
    end
    return (T_near_goal(tr), elapsed/T_near_goal(tr))
end