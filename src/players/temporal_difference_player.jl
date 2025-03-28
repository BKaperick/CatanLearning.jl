include("../learning/feature_computation.jl")
include("../learning/production_model.jl")
include("learning_player_base.jl")
include("../reinforcement.jl")
using Catan: do_post_action_step, choose_next_action

function get_transitions(action_sets::Vector{AbstractActionSet})::Vector{MarkovTransition}
    transitions = Vector{MarkovTransition}([])
    for (i,set) in enumerate(action_sets)
        #append!(transitions, get_transitions(set))
        if set isa ActionSet{SampledAction}
            append!(transitions, get_transitions(set))
        elseif set isa ActionSet{Action}
            append!(transitions, get_transitions(set))
        end
    end
    return transitions
end
function get_transitions(set::ActionSet{SampledAction})::Vector{MarkovTransition}
    transition = MarkovTransition(Vector{MarkovState}([]), set)
    # One transition for each set of sampled actions
    for act in set.actions
        push!(transition.states, MarkovState(act.features))
    end
    return [transition]
end
function get_transitions(set::ActionSet{Action})::Vector{MarkovTransition}
    transitions = []
    # One transition per deterministic action
    for action in set.actions
        transition = MarkovTransition(MarkovState(action.features), ActionSet(action))
        push!(transitions, transition)
    end
    return transitions
end

function Catan.do_post_action_step(board::Board, player::TemporalDifferencePlayer)
    next_features = compute_features(board, player.player)
    next_state = MarkovState(next_features)
    finish_temporal_difference_step!(player.process, player.current_state, next_state::MarkovState)
end


function Catan.choose_next_action(board::Board, players::Vector{PlayerPublicView}, player::TemporalDifferencePlayer, actions::Set{Symbol})::Tuple
    best_action_index = 0
    best_action_proba = -1
    machine = player.machine

    current_features = compute_features(board, player.player)
    current_state = MarkovState(current_features)
    current_quantity = get_state_optimizing_quantity(player.process, player.policy, current_state)
    player.current_state = current_state

    # TODO we have already computed win_proba, so we could pass it here and then 
    # we wouldn't need to worry in reinforcement-learning code about it
    action_sets = get_legal_action_sets(board, players, player.player, actions)
    analyze_actions!(board, players, player, action_sets)
    reachable_transitions = get_transitions(action_sets)::Vector{MarkovTransition}

    # TODO we just return the best reachable state based on the underlying model,
    # but we need to think about how temporal_difference_player should take probabilistic actions
    
    if length(reachable_transitions) == 0
        return (nothing, nothing)
    end
    
    next_state_quantity, index, transition = sample_from_policy(player.process, player.policy, current_state, 
                                           reachable_transitions)

    # Two cases: 
    # 1. transition is deterministic, so there is only one action in the set
    # 2. transition is stochastic, so there are multiple actions, but they're all the same func!
    action_func = transition.action_set.actions[1].func!
    
    # Only do an action if it will improve his optimized quantity
    if next_state_quantity > current_quantity
        return (nothing, action_func) # TODO
    end

    return (nothing, nothing)
end
