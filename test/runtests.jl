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

test_evolving_robot_game(neverend)
