# CatanLearning
Code for machine learning players training and playing via the [Catan.jl](https://github.com/BKaperick/Catan.jl) game engine.  It implements the reinforcement-learned players and their training code.

## Modeling strategy

Our current approach is via Temporal Difference learning.  We model the game as a Markov Reward Process, in which the scripted player refines their initial policy by exploring the state space via repeated self-play.

The reward function for player $i$, $R_i(s)$ is a function of the current game state, $s$ ($s$ comprises the board, the players, and any information that is not hidden to $i$) being used is a weighted linear combination:

$R_i(s) = \alpha M_i(s) + \beta P_i(s)$

where $P_i(s)$ is the number of victory points controlled by player $i$ in state $s$, and $M(s)$ is the output of a Random Forest classification model we have trained on ~500k games of a random player playing against itself.  The idea is that this rudimentary model has some insight that will give us a good initial policy, combined with the natural objective of maximizing victory points (as this is explicitly the win condition in Settlers of Catan -- the first player to achieve 10 victory points wins).

The value (that is, the reward + an exponentially-decaying summation of future state rewards branching from this state) $V_i(s)$ is updated via tabular Temporal Difference learning, which bootstraps our initial value guess in each state and gradually refines its value estimates of each game state as more states are explored (via self-play).

Our current approach is limited by the large state space of Catan, and so the next step is to implement some ideas to approximate the value of unseen states based on its proximity to already-explored states, as it is quite reasonable to expect that there is some continuity in the feature space of the state value function.  For examplee, if we consider state $s$ and state $s'$ which are identical except player $i$ having one extra sheep in his hand, we expect $V_i(s)$ and $V_i(s')$ to be close.  So even if we haven't visited $s'$ in our training data, the model should be able to infer this from its training data.

## State space exploration

Based on experiments doing TD learning on 4 players simultaneously, we reach ~2000 new states per game played.  With our set of 32 integer-valued features, we roughly estimate $32^5$ total feature values exist in realistic game positions, so that means we should need to play approximately 17,000 games to fully explore the state space.

### Exploring state space with TD(0)

Running `julia --project ./src/explore_temporal_difference_values.jl` starts running tournaments of 4 `TemporalDifferencePlayer` players against each other, and recording the estimated values of each hashed game state.
After approximately 9000 games, across 900 maps (10 games per randomly-generated map), we have explored ~20 million, or $2.0\times 10^7$i states at least once.  Given our above estimate of $32^5$ total states, we have explorted ~47% of the total state space.

![Value estimates](https://github.com/BKaperick/CatanLearning.jl/blob/master/data/sorted_value_estimates.png)

#### Testing Player

We test the temporal difference learned player against random players, `DefaultRobotPlayer`.

Results, where `no_winner` occurs if we reach 5000 turns without anyone achieving 10 victory points.

|no winner | TD player | Random player 1| Random player 2| Random player 3
-|-|-|-|-|-
TD player first|21|37|148|145|149
TD player last|27|40|135|150|148

So the TD player is performing badly.  Worse than a random player.  To debug this, we implement a simple validation test, `test_feature_perturbations` which constructs a feature vector, and then perturbs one feature at a time, in a direction that should be evidently correlated with victory.  For example, `:SettlementCount`, we expect the underlying model output, the state reward, and the state value to increase if we change nothing else except increasing the number of settlements a player has.

Starting from a null feature vector:
Random Forest model fails on features `:SettlementCount` and `:CountWood` with +2 and +3 perturbations.  All succeed with +1.
Reward fails on features `:SumStoneDiceWeight` and `:CountYearOfPlenty`.  All succeed with +1.
Value succeeds on all.

Interesting (and slightly suspicious) that only two fail on each of RF model and the state Reward, and that it's not the same problematic features.


# Benchmarks
Player type | mean game time
-|-
Catan.DefaultRobotPlayer | 332 ms
