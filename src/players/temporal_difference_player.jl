include("../reinforcement.jl")
using Catan: do_post_action_step, choose_next_action

function get_transitions(process::MarkovRewardProcess, model::DecisionModel, action_sets::Vector{AbstractActionSet})::Vector{MarkovTransition}
    transitions = Vector{MarkovTransition}([])
    for (i,set) in enumerate(action_sets)
        #append!(transitions, get_transitions(set))
        if set isa ActionSet{SampledAction}
            append!(transitions, get_transitions(process, model, set))
        elseif set isa ActionSet{Action}
            append!(transitions, get_transitions(process, model, set))
        end
    end
    return transitions
end
function get_transitions(process::MarkovRewardProcess, model::DecisionModel, set::ActionSet{SampledAction})::Vector{MarkovTransition}
    states = Vector{MarkovState}([])
    # One transition for each set of sampled actions
    for act in set.actions
        push!(states, MarkovState(process, act.features, model))
    end
    return [MarkovTransition(states, set)]
end
function get_transitions(process::MarkovRewardProcess, model::DecisionModel, set::ActionSet{Action})::Vector{MarkovTransition}
    transitions = []
    # One transition per deterministic action
    for action in set.actions
        transition = MarkovTransition([MarkovState(process, action.features, model)], ActionSet(action))
        push!(transitions, transition)
    end
    return transitions
end

function Catan.do_post_action_step(board::Board, player::MarkovPlayer)
    next_features = compute_features(board, player.player)
    next_state = MarkovState(player.process, next_features, player.machine)

    @assert next_state.reward !== nothing
    finish_temporal_difference_step!(player.process, player.current_state, next_state::MarkovState)
end


function Catan.choose_next_action(board::Board, players::AbstractVector{PlayerPublicView}, player::MarkovPlayer, actions::Set{PreAction})::ChosenAction
    best_action_index = 0
    best_action_proba = -1
    decision_model = player.machine::DecisionModel

    current_features = compute_features(board, player.player)
    current_state = MarkovState(player.process, current_features, decision_model)
    #current_quantity = 0
    #current_state.reward = 0
    current_quantity = get_state_optimizing_quantity(player.process, player.policy, current_state)
    player.current_state = current_state

    # TODO we have already computed win_proba, so we could pass it here and then 
    # we wouldn't need to worry in reinforcement-learning code about it
    action_sets = get_legal_action_sets(board, players, player.player, actions)
    analyze_actions!(board, players, player, action_sets, 0)
    reachable_transitions = get_transitions(player.process, decision_model, action_sets)::Vector{MarkovTransition}

    # TODO we just return the best reachable state based on the underlying model,
    # but we need to think about how temporal_difference_player should take probabilistic actions
    
    if length(reachable_transitions) == 0
        return ChosenAction(:DoNothing)
    end
    
    next_state_quantity, index, transition = sample_from_policy(player.process, player.policy, current_state, 
                                           reachable_transitions)

    # Two cases: 
    # 1. transition is deterministic, so there is only one action in the set
    # 2. transition is stochastic, so there are multiple actions, but they're all the same func!
    #action_func = transition.action_set.actions[1].func!
    best_action = transition.action_set.actions[1]
    
    # Only do an action if it will improve his optimized quantity
    if next_state_quantity > current_quantity
        return ChosenAction(best_action.name, best_action.args...)
        #return action_func # TODO
    end

    return ChosenAction(:DoNothing)
end
