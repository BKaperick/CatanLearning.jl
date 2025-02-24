# CatanLearning
Code for machine learning players training and playing via the [Catan.jl](https://github.com/BKaperick/Catan.jl) game engine.  It implements the reinforcement-learned players and their training code.

## Modeling strategy

Our current approach is via Temporal Difference learning.  We model the game as a Markov Reward Process, in which the scripted player refines their initial policy by exploring the state space via repeated self-play.

The reward function for player $i$, $R_i(s)$ is a function of the current game state, $s$ ($s$ comprises the board, the players, and any information that is not hidden to $i$) being used is a weighted linear combination:

$R_i(s) = \alpha M_i(s) + \beta P_i(s)$

where $P_i(s)$ is the number of victory points controlled by player $i$ in state $s$, and $M(s)$ is the output of a Random Forest classification model we have trained on ~500k games of a random player playing against itself.  The idea is that this rudimentary model has some insight that will give us a good initial policy, combined with the natural objective of maximizing victory points (as this is explicitly the win condition in Settlers of Catan -- the first player to achieve 10 victory points wins).

The value (that is, the reward + an exponentially-decaying summation of future state rewards branching from this state) $V_i(s)$ is updated via tabular Temporal Difference learning, which bootstraps our initial value guess in each state and gradually refines its value estimates of each game state as more states are explored (via self-play).

Our current approach is limited by the large state space of Catan, and so the next step is to implement some ideas to approximate the value of unseen states based on its proximity to already-explored states, as it is quite reasonable to expect that there is some continuity in the feature space of the state value function.  For examplee, if we consider state $s$ and state $s'$ which are identical except player $i$ having one extra sheep in his hand, we expect $V_i(s)$ and $V_i(s')$ to be close.  So even if we haven't visited $s'$ in our training data, the model should be able to infer this from its training data.
