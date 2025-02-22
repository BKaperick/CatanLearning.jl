function test_evolving_robot_game(neverend)
    team_and_playertype = [
                          (:blue, EmpathRobotPlayer),
                          (:cyan, MutatedEmpathRobotPlayer),
                          (:green, EmpathRobotPlayer),
                          (:red, MutatedEmpathRobotPlayer)
            ]
    players = Catan.setup_players(team_and_playertype)
    Catan.test_automated_game(neverend, players)
end

function empath_player()
    player = EmpathRobotPlayer(:red)
    board = read_map(SAMPLE_MAP)
    p = predict_model(player.machine, board, player)
    return player, board, p
end

test_evolving_robot_game(neverend)

