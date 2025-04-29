using Random
using CSV
using MLJ
using DataFrames
import DataFramesMeta as DFM
using DelimitedFiles
#include("../io.jl")

"""
:SettlementCount => 0.0,
:CityCount => 0.0,
:RoadCount => 0.0,
:MaxRoadLength => 0.0,
:SumWoodDiceWeight => 0.0,
:SumBrickDiceWeight => 0.0,
:SumPastureDiceWeight => 0.0,
:SumStoneDiceWeight => 0.0,
:SumGrainDiceWeight => 0.0
:PortWood => 0.0,
:PortBrick => 0.0,
:PortPasture => 0.0,
:PortStone => 0.0,
:PortGrain => 0.0
:CountWood => 0.0,
:CountBrick => 0.0,
:CountPasture => 0.0,
:CountStone => 0.0,
:CountGrain => 0.0
:CountKnight => 0.0,
:CountMonopoly => 0.0,
:CountYearOfPlenty => 0.0,
:CountRoadBuilding => 0.0,
:CountVictoryPoint => 0.0
# :IsNotLoss => 0.0
"""


struct Feature
    name::Symbol
    type::Type
    #TODO add scitype as well for ML pipeline
    min::Float64
    max::Float64
end
feature_library = Dict{Symbol, Feature}()

function register_feature(name::Symbol, type::Type, min, max)
    feature_library[name] = Feature(name, type, min, max)
end

# Helper functions start with `get_`, and feature computers take (board, player) and start with `compute_`.

register_feature(:CountSettlement,Int8,0,5)
compute_count_settlement = (board, player) -> get_building_count(board, :Settlement, player.team)
register_feature(:CountTotalSettlement,Int8,0,9)
compute_count_total_settlement = (board, player) -> get_building_count(board, :Settlement, player.team) + get_building_count(board, :City, player.team)

register_feature(:CountCity, Int8, 0, 4)
compute_count_city = (board, player) -> get_building_count(board, :City, player.team)

register_feature(:CountRoad, Int8, 0, 14)
compute_count_road = (board, player) -> get_road_count(board, player.team)

register_feature(:MaxRoadLength, Int8, 0, 14)
compute_max_road_length = (board, player) -> BoardApi.get_max_road_length(board, player.team)

register_feature(:SumWoodDiceWeight, Int16, 4, 186)
compute_sum_wood_dice_weight = (board, player) -> get_sum_resource_dice_weight(board, player.team, :Wood)
register_feature(:SumBrickDiceWeight, Int16, 4, 186)
compute_sum_brick_dice_weight = (board, player) -> get_sum_resource_dice_weight(board, player.team, :Brick)
register_feature(:SumPastureDiceWeight, Int16, 4, 186)
compute_sum_pasture_dice_weight = (board, player) -> get_sum_resource_dice_weight(board, player.team, :Pasture)
register_feature(:SumStoneDiceWeight, Int16, 4, 186)
compute_sum_stone_dice_weight = (board, player) -> get_sum_resource_dice_weight(board, player.team, :Stone)
register_feature(:SumGrainDiceWeight, Int16, 4, 186)
compute_sum_grain_dice_weight = (board, player) -> get_sum_resource_dice_weight(board, player.team, :Grain)

register_feature(:CountPortWood, Int8, 0, 1)
compute_count_port_wood = (board, player) -> get_resource_port_count(board, player.team, :Wood)
register_feature(:CountPortBrick, Int8, 0, 1)
compute_count_port_brick = (board, player) -> get_resource_port_count(board, player.team, :Brick)
register_feature(:CountPortPasture, Int8, 0, 1)
compute_count_port_pasture = (board, player) -> get_resource_port_count(board, player.team, :Pasture)
register_feature(:CountPortStone, Int8, 0, 1)
compute_count_port_stone = (board, player) -> get_resource_port_count(board, player.team, :Stone)
register_feature(:CountPortGrain, Int8, 0, 1)
compute_count_port_grain = (board, player) -> get_resource_port_count(board, player.team, :Grain)

register_feature(:CountHandWood, Int8, 0, 20)
compute_count_hand_wood = (board, player) -> get_resource_hand_count(player, :Brick)
register_feature(:CountHandBrick, Int8, 0, 20)
compute_count_hand_brick = (board, player) -> get_resource_hand_count(player, :Brick)
register_feature(:CountHandPasture, Int8, 0, 20)
compute_count_hand_pasture = (board, player) -> get_resource_hand_count(player, :Pasture)
register_feature(:CountHandStone, Int8, 0, 20)
compute_count_hand_stone = (board, player) -> get_resource_hand_count(player, :Stone)
register_feature(:CountHandGrain, Int8, 0, 20)
compute_count_hand_grain = (board, player) -> get_resource_hand_count(player, :Grain)

register_feature(:CountTotalWood, Int8, 0, 20)
compute_count_total_wood = (board, player) -> get_resource_total_count(board, player, :Brick)
register_feature(:CountTotalBrick, Int8, 0, 20)
compute_count_total_brick = (board, player) -> get_resource_total_count(board, player, :Brick)
register_feature(:CountTotalPasture, Int8, 0, 20)
compute_count_total_pasture = (board, player) -> get_resource_total_count(board, player, :Pasture)
register_feature(:CountTotalStone, Int8, 0, 20)
compute_count_total_stone = (board, player) -> get_resource_total_count(board, player, :Stone)
register_feature(:CountTotalGrain, Int8, 0, 20)
compute_count_total_grain = (board, player) -> get_resource_total_count(board, player, :Grain)

register_feature(:CountDevCardsUsedKnight, Int8, 0, 14)
compute_count_devcards_used_knight = (board, player) -> get_devcards_used_count(player, :Knight)
register_feature(:CountDevCardsUsedMonopoly, Int8, 0, 2)
compute_count_devcards_used_monopoly = (board, player) -> get_devcards_used_count(player, :Monopoly)
register_feature(:CountDevCardsUsedYearOfPlenty, Int8, 0, 2)
compute_count_devcards_used_year_of_plenty = (board, player) -> get_devcards_used_count(player, :YearOfPlenty)
register_feature(:CountDevCardsUsedRoadBuilding, Int8, 0, 2)
compute_count_devcards_used_road_building = (board, player) -> get_devcards_used_count(player, :RoadBuilding)

register_feature(:CountDevCardsKnight, Int8, 0, 14)
compute_count_devcards_owned_knight = (board, player) -> get_devcards_owned_count(player, :Knight)
register_feature(:CountDevCardsMonopoly, Int8, 0, 2)
compute_count_devcards_owned_monopoly = (board, player) -> get_devcards_owned_count(player, :Monopoly)
register_feature(:CountDevCardsYearOfPlenty, Int8, 0, 2)
compute_count_devcards_owned_year_of_plenty = (board, player) -> get_devcards_owned_count(player, :YearOfPlenty)
register_feature(:CountDevCardsRoadBuilding, Int8, 0, 2)
compute_count_devcards_owned_road_building = (board, player) -> get_devcards_owned_count(player, :RoadBuilding)
register_feature(:CountDevCardsVictoryPoint, Int8, 0, 5)
compute_count_devcards_owned_victory_point = (board, player) -> get_devcards_owned_count(player, :VictoryPoint)
register_feature(:HasLargestArmy, Bool, 0, 1)
compute_has_largest_army = (board, player) -> board.largest_army == player.team
register_feature(:HasLongestRoad, Bool, 0, 1)
compute_has_longest_road = (board, player) -> board.longest_road == player.team
register_feature(:CountVictoryPoint, Int8, 0, 10)
compute_count_victory_points = (board, player) -> Catan.GameRunner.get_total_vp_count(board, player)
register_feature(:CountVisibleVictoryPoint, Int8, 0, 10)
compute_count_public_victory_points = (board, player) -> Catan.BoardApi.get_public_vp_count(board, player.team)
register_feature(:WonGame, Bool, 0, 1)
compute_won_game = (board, player) -> Catan.GameRunner.get_total_vp_count(board, player) >= board.configs["GameSettings"]["VICTORY_POINT_GOAL"]
register_feature(:HasMostPoints, Bool, 0, 1)
compute_has_most_points = (game, board, player) -> get_has_most_points(game, board, player)
register_feature(:NumberOfTurns, Int32, 0, 10_000)
compute_number_of_turns = (game, board, player) -> game.turn_num

"""
    `compute_features(board, player::Player)::Vector{Pair{Symbol, Float64}}`

Compute features from public info, and used for inference.  Note, this *cannot* include `Game`, because that
would leak private info about other players.
"""
function compute_features(board, player::Player)::Vector{Pair{Symbol, Float64}}
    raw_features = [
        :CountSettlement => compute_count_settlement(board, player),
        :CountTotalSettlement => compute_count_total_settlement(board, player),
        :CountCity => compute_count_city(board, player),
        :CountRoad => compute_count_road(board, player),
        :MaxRoadLength => compute_max_road_length(board, player),

        :SumWoodDiceWeight => compute_sum_wood_dice_weight(board, player),
        :SumBrickDiceWeight => compute_sum_brick_dice_weight(board, player),
        :SumPastureDiceWeight => compute_sum_pasture_dice_weight(board, player),
        :SumStoneDiceWeight => compute_sum_stone_dice_weight(board, player),
        :SumGrainDiceWeight => compute_sum_grain_dice_weight(board, player),
        :CountPortWood => compute_count_port_wood(board, player),
        :CountPortBrick => compute_count_port_brick(board, player),
        :CountPortPasture => compute_count_port_pasture(board, player),
        :CountPortStone => compute_count_port_stone(board, player),
        :CountPortGrain => compute_count_port_grain(board, player),

        :CountHandWood => compute_count_hand_wood(board, player),
        :CountHandBrick => compute_count_hand_brick(board, player),
        :CountHandPasture => compute_count_hand_pasture(board, player),
        :CountHandStone => compute_count_hand_stone(board, player),
        :CountHandGrain => compute_count_hand_grain(board, player),
        
        :CountTotalWood => compute_count_total_wood(board, player),
        :CountTotalBrick => compute_count_total_brick(board, player),
        :CountTotalPasture => compute_count_total_pasture(board, player),
        :CountTotalStone => compute_count_total_stone(board, player),
        :CountTotalGrain => compute_count_total_grain(board, player),
        
        :CountDevCardsKnight => compute_count_devcards_owned_knight(board, player),
        :CountDevCardsMonopoly => compute_count_devcards_owned_monopoly(board, player),
        :CountDevCardsYearOfPlenty => compute_count_devcards_owned_year_of_plenty(board, player),
        :CountDevCardsRoadBuilding => compute_count_devcards_owned_road_building(board, player),
        :CountDevCardsVictoryPoint => compute_count_devcards_owned_victory_point(board, player),
        :HasLargestArmy => compute_has_largest_army(board, player),
        :HasLongestRoad => compute_has_longest_road(board, player),
        :CountVictoryPoint => compute_count_victory_points(board, player)
       ]
    
end

function compute_public_features(board, player)::Vector{Pair{Symbol, Float64}}
    return [
        :CountSettlement => compute_count_settlement(board, player),
        :CountTotalSettlement => compute_count_total_settlement(board, player),
        :CountCity => compute_count_city(board, player),
        :CountRoad => compute_count_road(board, player),
        :MaxRoadLength => compute_max_road_length(board, player),

        :SumWoodDiceWeight => compute_sum_wood_dice_weight(board, player),
        :SumBrickDiceWeight => compute_sum_brick_dice_weight(board, player),
        :SumPastureDiceWeight => compute_sum_pasture_dice_weight(board, player),
        :SumStoneDiceWeight => compute_sum_stone_dice_weight(board, player),
        :SumGrainDiceWeight => compute_sum_grain_dice_weight(board, player),
        :CountPortWood => compute_count_port_wood(board, player),
        :CountPortBrick => compute_count_port_brick(board, player),
        :CountPortPasture => compute_count_port_pasture(board, player),
        :CountPortStone => compute_count_port_stone(board, player),
        :CountPortGrain => compute_count_port_grain(board, player),

        #:CountHandWood => compute_count_hand_wood(board, player),
        #:CountHandBrick => compute_count_hand_brick(board, player),
        #:CountHandPasture => compute_count_hand_pasture(board, player),
        #:CountHandStone => compute_count_hand_stone(board, player),
        #:CountHandGrain => compute_count_hand_grain(board, player),
        
        #:CountTotalWood => compute_count_total_wood(board, player),
        #:CountTotalBrick => compute_count_total_brick(board, player),
        #:CountTotalPasture => compute_count_total_pasture(board, player),
        #:CountTotalStone => compute_count_total_stone(board, player),
        #:CountTotalGrain => compute_count_total_grain(board, player),
        
        :CountDevCardsUsedKnight => compute_count_devcards_used_knight(board, player),
        :CountDevCardsUsedMonopoly => compute_count_devcards_used_monopoly(board, player),
        :CountDevCardsUsedYearOfPlenty => compute_count_devcards_used_year_of_plenty(board, player),
        :CountDevCardsUsedRoadBuilding => compute_count_devcards_used_road_building(board, player),
        :HasLargestArmy => compute_has_largest_army(board, player),
        :HasLongestRoad => compute_has_longest_road(board, player),
        :CountVisibleVictoryPoint => compute_count_public_victory_points(board, player)
       ]
end

function compute_labels(game, board, player::Player)::Vector{Pair{Symbol, Float64}}
    return [
        # :CountVictoryPoint => compute_count_victory_points(board, player),
        :HasMostPoints => compute_has_most_points(game, board, player),
        #:NumberOfTurns => compute_number_of_turns(game, board, player),
        :WonGame => compute_won_game(board, player)
       ]
end
function compute_features_and_labels(game, board, player::Player)::Vector{Pair{Symbol, Float64}}
    return vcat(compute_features(board, player), compute_labels(game, board, player))
end
function compute_public_features_and_labels(game, board, player::Player)::Vector{Pair{Symbol, Float64}}
    return vcat(compute_public_features(board, player), compute_labels(game, board, player))
end

function get_building_count(board, building_type, team)
    out = 0
    for building in board.buildings
        if building.team == team && building.type == building_type
            out += 1
        end
    end
    return out
end

function get_road_count(board, team)
    out = 0
    for building in board.roads
        if building.team == team
            out += 1
        end
    end
    return out
end

"""
    `get_sum_resource_dice_weight(board, player.team, resource)::Int`

The sum of dice weight (0-5 increasing probability of rolling number, 7 is 0, while 2 and 12 are 1.)
MAX => 3*(2(4*5) + (2*5 + 3*4)) = 186
MIN => 2*2 = 4
"""
function get_sum_resource_dice_weight(board, team, resource)::Int
    total_weight = 0
    for (c,b) in board.coord_to_building
        if b.team == team
            for tile in Catan.COORD_TO_TILES[c]
                if board.tile_to_resource[tile] == resource
                    weight = Catan.DICEVALUE_TO_PROBA_WEIGHT[board.tile_to_dicevalue[tile]]
                    if b.type == :City
                        weight *= 2
                    end

                    total_weight += weight
                end
            end
        end
    end
    return total_weight
end
function get_resource_hand_count(player, resource)::Int
    return haskey(player.resources, resource) ? player.resources[resource] : 0
end

"""
    `get_resource_total_count(board, player, resource)::Int`

Get the count of all resources in hand as well as all the resources already spent
"""
function get_resource_total_count(board, player, resource)::Int
    hand_count = get_resource_hand_count(player, resource)
    building_count = 0
    for b in board.buildings
        if b.team == player.team && haskey(Catan.COSTS[b.type], resource)
            building_count += Catan.COSTS[b.type][resource]
        end
    end

    road_count = 0
    for b in board.roads
        if b.team == player.team && haskey(Catan.COSTS[:Road], resource)
            building_count += Catan.COSTS[:Road][resource]
        end
    end
    
    devcard_count = 0
    for c in player.devcards
        if haskey(Catan.COSTS[:DevelopmentCard], resource)
            devcard_count += Catan.COSTS[:DevelopmentCard][resource]
        end
    end
    for c in player.devcards_used
        if haskey(Catan.COSTS[:DevelopmentCard], resource)
            devcard_count += Catan.COSTS[:DevelopmentCard][resource]
        end
    end
    
    return hand_count + building_count + road_count + devcard_count        
end

function get_resource_port_count(board, team, resource)::Int
    count = 0
    for (c,p) in board.coord_to_port
        if p == resource && haskey(board.coord_to_building, c) && board.coord_to_building[c].team == team
            count += 1
        end
    end
    return count
end

function get_devcards_owned_count(player, devcard)::Int
    count = 0
    for (card,cnt) in player.devcards
        if card == devcard
            count += cnt
        end
    end
    count += get_devcards_used_count(player, devcard)
    return count
end

function get_devcards_used_count(player, devcard)::Int
    count = 0
    for (card,cnt) in player.devcards_used
        if card == devcard
            count += cnt
        end
    end
    return count
end

function get_has_most_points(game, board, player::Player)::Bool
    points = Catan.GameRunner.get_total_vp_count(board, player)
    for p in game.players
        if Catan.GameRunner.get_total_vp_count(board, p.player) > points
            return false
        end
    end
    return true
end

"""
    `predict_public_model(machine::Machine, board::Board, player::PlayerPublicView)`

Either a trained simple model that uses publicly-available info (defined in `compute_public_features`)
and use that, or we use the naive model, i.e., we just do a linear scaling of public VP count 
"""
function predict_public_model(machine::Machine, board::Board, player::PlayerPublicView)
    return predict_model(machine, compute_public_features(board, player))
end

function predict_model(machine::Machine, board::Board, player::PlayerType)
    return predict_model(machine, compute_features(board, player.player))
end
function predict_model(machine::Machine, features)
    X_new = DataFrame(features)
    CatanLearning.coerce_feature_types!(X_new)
    CatanLearning.filter_bad_features!(X_new)
    pred = MLJ.predict(machine, X_new)
    return pdf(pred[1], 1)
end
