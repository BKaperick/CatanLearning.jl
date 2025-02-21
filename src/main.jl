using Catan
import Catan

mutable struct NewRobotPlayer <: RobotPlayer
    player::Player
end

NewRobotPlayer(team::Symbol) = NewRobotPlayer(Player(team))

#players = read_players_from_config(ARGS[1])
players = Vector{Catan.PlayerType}([
    DefaultRobotPlayer(:Cyan),
    NewRobotPlayer(:Red),
    DefaultRobotPlayer(:Yellow),
    DefaultRobotPlayer(:Green)
   ])
#Catan.run(ARGS, players)
Catan.run(players)
