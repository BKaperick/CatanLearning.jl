# Catan.jl - a Settlers of Catan game engine
# CatanLearning.jl - 

The project uses the full Julia Catan engine, Catan.jl:
* [Catan.jl source code](https://github.com/BKaperick/Catan.jl/tree/master)
* [Catan.jl Docs](https://bkaperick.github.io/Catan.jl/stable/)

## How to run the game
To launch a new game, set up a configuration file with the desired player types, and any gameplay configuration.

For example,
```toml filename="Configuration.toml"
# Changing any configs you want (defaults in ./DefaultConfiguration.toml)
PRINT_BOARD = true

# Setting up the types of players.  
# One human (blue) against three scripted players of type `DefaultRobotPlayer`
[PlayerSettings]
[PlayerSettings.blue]
TYPE = "HumanPlayer"
[PlayerSettings.cyan]
TYPE = "DefaultRobotPlayer"
[PlayerSettings.green]
TYPE = "DefaultRobotPlayer"
[PlayerSettings.yellow]
TYPE = "DefaultRobotPlayer"
```

If this file is saved as `Configuration.toml`, then we can run one game with the following script:
```julia
using Catan
using CatanLearning
winners = CatanLearning.run_tournament("Configuration.toml")
```
