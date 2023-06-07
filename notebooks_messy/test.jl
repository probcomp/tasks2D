import GLMakie
using Revise
using Gen
includet("../src/gridworld.jl")
includet("../src/houseexpo.jl")
includet("../src/visualize.jl")
includet("../src/raytrace.jl")
includet("../src/distributions.jl")

w = load_gridworld(24, 7);

# Observation model
@gen function observe_noisy_distances(w::GridWorld, n_rays, σ)
    dists = ray_trace_distances(w, n_rays)
    obs ~ broadcasted_normal(dists, σ)
    return obs
end
# Position prior
# Accepts as input a gridworld without the agent in it
@gen function uniform_agent_pos(w::GridWorld)
    pos ~ uniform_from_set(empty_cells(w))
    return pos
end
@dist list_categorical(list, probs) = list[categorical(probs)]
# action `a` should be a symbol in [:up, :down, :left, :right, :stay]
@gen function motion_model(w::GridWorld, xy, a::Symbol, σ, return_logprobs=false)
    (x2, y2) = newpos(w, xy, a)
    possible_positions = reshape(Tuple.(keys(w)), (:,))
    logprobs = [
        w[x, y] == empty ? logpdf(broadcasted_normal, [x, y], [x2, y2], σ) : -Inf
        for (x, y) in possible_positions
    ]
    normalized_logprobs = logprobs .- logsumexp(logprobs)
    pos ~ list_categorical(possible_positions, exp.(normalized_logprobs))
    if return_logprobs
        return (pos, normalized_logprobs)
    else
        return pos
    end
end
@gen function generate_trajectory(T, w0::GridWorld, actions, params)
    σmotion, n_obs_pts, σobs = params.σmotion, params.n_obs_pts, params.σobs
    
    pos = {:t => 0 => :pos} ~ uniform_agent_pos(w0)
    obs = {:t => 0 => :obs} ~ observe_noisy_distances(moveagent(w, pos), n_obs_pts, σobs)
    for t=1:T
        pos = {:t => t => :pos} ~ motion_model(w0, pos, actions[t], σmotion)
        obs = {:t => t => :obs} ~ observe_noisy_distances(moveagent(w, pos), n_obs_pts, σobs)
    end
end


PARAMS = (σmotion = 0.1, n_obs_pts = 20, σobs = 1.)
#=
First -- I want a function to visualize a trace at a given timestep.
Then  -- I want to be able to take actions, and have the trace update.
I also want actions that backtrack the trace.

So, I need a function which accepts a trace as input.
It will create an observable `t`.
It will visualize the world state, and observations, as of time t in the trace.
It will set up an action listener, so the "t" and "g" keys increment/decrement t.
    (t and g since they do not have Jupyter keybindings)
It will set up action listeners so that if t = its maximum value, then WASD & E control up/left/down/right & stay.
    Pressing one of these will extend the trace by updating it with the given action.
It will return f, and t.
=#
function play_as_agent(initial_trace)
    t = Makie.Observable(0)
    tr = Makie.Observable(initial_trace)
    
    empty_world = get_args(initial_trace)[2]
    world = Makie.@lift(moveagent(empty_world, $tr[:t => $t => :pos]))
    
    function action(a)
        println("trying to take action...")
        if t[] == get_args(tr[])[1]
            # Update the trace with the action
            @async begin
                T, w0, as, params = get_args(tr[])
                tr[] = Gen.update(
                    tr[],
                    (T + 1, w0, [as; a], params),
                    (UnknownChange(), NoChange(), UnknownChange(), NoChange()),
                    EmptyChoiceMap()
                )[1]
                t[] = t[] + 1
            end
        end
        println("took action")
    end
    function timeup()
        if t[] < get_args(tr[])[1]
            t[] = t[] + 1
        end
    end
    function timedown()
        if t[] > 0
            t[] = t[] - 1
        end
    end
    
    f = visualize_grid(world)
    obs_pts = Makie.@lift(
        map(Makie.Point2,
            zip(points_from_raytracing($tr[:t => $t => :pos]..., $tr[:t => $t => :obs])...)
        )
    )
    Makie.scatter!(obs_pts)
    
    register_keyboard_listeners(f;
        keys=WASDE_TG_KEYS(),
        callbacks=(;
            up = () -> action(:up),
            down = () -> action(:down),
            left = () -> action(:left),
            right = () -> action(:right),
            stay = () -> action(:stay),
            timeup, timedown 
        )
    )
    
    (f, t, tr)
end
function play_as_agent(w::GridWorld, params)
    play_as_agent(simulate(generate_trajectory, (0, w, [], params)))
end

(f, t, tr) = play_as_agent(w, PARAMS);
f