using Random
using CSV
using MLJ
using DataFrames
import DataFramesMeta as DFM
using DelimitedFiles
include("../io.jl")

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
    min::Float64
    max::Float64
end
feature_library = Dict{Symbol, Feature}()

function register_feature(name::Symbol, type::Type, min, max)
    feature_library[name] = Feature(name, type, min, max)
end

# Helper functions start with `get_`, and feature computers take (board, player) and start with `compute_`.

register_feature(:SettlementCount,Int8,0,5)
compute_count_settlement = (board, player) -> get_building_count(board, :Settlement, player.team)

register_feature(:CityCount, Int8, 0, 4)
compute_count_city = (board, player) -> get_building_count(board, :City, player.team)

register_feature(:RoadCount, Int8, 0, 14)
compute_count_road = (board, player) -> get_road_count(board, player.team)

register_feature(:MaxRoadLength, Int8, 0, 14)
compute_max_road_length = (board, player) -> get_max_road_length(board, player.team)

register_feature(:SumWoodDiceWeight, Int8, 4, 186)
compute_sum_wood_dice_weight = (board, player) -> get_sum_resource_dice_weight(board, player.team, :Wood)
register_feature(:SumBrickDiceWeight, Int8, 4, 186)
compute_sum_brick_dice_weight = (board, player) -> get_sum_resource_dice_weight(board, player.team, :Brick)
register_feature(:SumPastureDiceWeight, Int8, 4, 186)
compute_sum_pasture_dice_weight = (board, player) -> get_sum_resource_dice_weight(board, player.team, :Pasture)
register_feature(:SumStoneDiceWeight, Int8, 4, 186)
compute_sum_stone_dice_weight = (board, player) -> get_sum_resource_dice_weight(board, player.team, :Stone)
register_feature(:SumGrainDiceWeight, Int8, 4, 186)
compute_sum_grain_dice_weight = (board, player) -> get_sum_resource_dice_weight(board, player.team, :Grain)

register_feature(:PortWood, Int8, 0, 1)
compute_count_port_wood = (board, player) -> get_resource_port_count(board, player.team, :Wood)
register_feature(:PortBrick, Int8, 0, 1)
compute_count_port_brick = (board, player) -> get_resource_port_count(board, player.team, :Brick)
register_feature(:PortPasture, Int8, 0, 1)
compute_count_port_pasture = (board, player) -> get_resource_port_count(board, player.team, :Pasture)
register_feature(:PortStone, Int8, 0, 1)
compute_count_port_stone = (board, player) -> get_resource_port_count(board, player.team, :Stone)
register_feature(:PortGrain, Int8, 0, 1)
compute_count_port_grain = (board, player) -> get_resource_port_count(board, player.team, :Grain)

register_feature(:CountWood, Int8, 0, 20)
compute_count_hand_wood = (board, player) -> get_resource_hand_count(player, :Brick)
register_feature(:CountBrick, Int8, 0, 20)
compute_count_hand_brick = (board, player) -> get_resource_hand_count(player, :Brick)
register_feature(:CountPasture, Int8, 0, 20)
compute_count_hand_pasture = (board, player) -> get_resource_hand_count(player, :Pasture)
register_feature(:CountStone, Int8, 0, 20)
compute_count_hand_stone = (board, player) -> get_resource_hand_count(player, :Stone)
register_feature(:CountGrain, Int8, 0, 20)
compute_count_hand_grain = (board, player) -> get_resource_hand_count(player, :Grain)

register_feature(:CountDevCardsKnight, Int8, 0, Catan.DEVCARD_COUNTS[:Knight])
compute_count_devcards_owned_knight = (board, player) -> get_devcards_owned_count(player, :Knight)
register_feature(:CountDevCardsMonopoly, Int8, 0, Catan.DEVCARD_COUNTS[:Monopoly])
compute_count_devcards_owned_monopoly = (board, player) -> get_devcards_owned_count(player, :Monopoly)
register_feature(:CountDevCardsYearOfPlenty, Int8, 0, Catan.DEVCARD_COUNTS[:YearOfPlenty])
compute_count_devcards_owned_year_of_plenty = (board, player) -> get_devcards_owned_count(player, :YearOfPlenty)
register_feature(:CountDevCardsRoadBuilding, Int8, 0, Catan.DEVCARD_COUNTS[:RoadBuilding])
compute_count_devcards_owned_road_building = (board, player) -> get_devcards_owned_count(player, :RoadBuilding)
register_feature(:CountDevCardsVictoryPoint, Int8, 0, Catan.DEVCARD_COUNTS[:VictoryPoint])
compute_count_devcards_owned_victory_point = (board, player) -> get_devcards_owned_count(player, :VictoryPoint)
register_feature(:HasLargestArmy, Bool, 0, 1)
compute_has_largest_army = (board, player) -> board.largest_army == player.team
register_feature(:HasLongestRoad, Bool, 0, 1)
compute_has_longest_road = (board, player) -> board.longest_road == player.team
register_feature(:CountDevCardsVictoryPoint, Int8, 0, Catan.DEVCARD_COUNTS[:VictoryPoint])
compute_has_largest_army
register_feature(:CountVictoryPoint, Int8, 0, 10)
compute_count_victory_points = (board, player) -> Catan.GameRunner.get_total_vp_count(board, player)
#compute_is_not_loss = (board, player) -> 

function compute_features(board, player)::Vector{Pair{Symbol, Float64}}
    return [
        :CountSettlement => compute_count_settlement(board, player),
        :CountCity => compute_count_city(board, player),
        :CountRoad => compute_count_road(board, player),

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
        :CountDevCardsKnight => compute_count_devcards_owned_knight(board, player),
        :CountDevCardsMonopoly => compute_count_devcards_owned_monopoly(board, player),
        :CountDevCardsYearOfPlenty => compute_count_devcards_owned_year_of_plenty(board, player),
        :CountDevCardsRoadBuilding => compute_count_devcards_owned_road_building(board, player),
        :CountDevCardsVictoryPoint => compute_count_devcards_owned_victory_point(board, player),

        :HasLargestArmy => compute_has_largest_army(board, player),
        :HasLongestRoad => compute_has_longest_road(board, player),

        :CountVictoryPoint => compute_count_victory_points(board, player)
        # :IsNotLoss => compute_is_not_loss(board, player)
       ]
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
    for (card,cnt) in player.devcards_used
        if card == devcard
            count += cnt
        end
    end
    return count
end

function predict_model(machine::Machine, board::Board, player::PlayerType)
    features = [x for x in compute_features(board, player.player)]
    return predict_model(machine, features)
end

function predict_model(machine::Machine, features::Dict{Symbol, Float64})
    return predict_model(machine, collect(features))
end
function predict_model(machine::Machine, features::Vector{Pair{Symbol, Float64}})
    header = get_csv_friendly.(first.(features))
    feature_vals = last.(features)
    pred = _predict_model_feature_vec(machine, feature_vals, header)

    # Returns the win probability (proba weight on category for label `1` indicating win)
    return pdf(pred[1], 1)
end

function _predict_model_feature_vec(machine::Machine, feature_vals::Vector{T}, header::Vector{String}) where T <: Number
    data = reshape(feature_vals, 1, length(feature_vals))
    #header = names(machine.data[1])
    df = DataFrame(data, vec(header), makeunique=true)
    return Base.invokelatest(MLJ.predict, machine, df)
end
