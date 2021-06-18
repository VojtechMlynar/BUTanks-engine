"""BUTanks engine - sample outline script RAI 2021"""

import engine


WIDTH, HEIGHT = 1000, 1000  # [px]
MAP_FILENAME = "map1.png"  # ../assets/maps (preset path)
TARGET_CAPTURE_TIME = 5  # [s]
NUM_OF_ROUNDS = 4
TANK_SCALE = 1


def main():
    # Custom config (optional)
    config = engine.GameConfig()
    config.MAP_BACKGROUND_COLOR = (100, 100, 100)

    # Create engine.Game class instance
    game = engine.Game(MAP_FILENAME, (WIDTH, HEIGHT), NUM_OF_ROUNDS,
                       game_config=config)

    # Create spawn lists
    team1_sl = [(100, 100, 0)]
    team2_sl = [(WIDTH-100, HEIGHT-100, 180)]

    # Main loop
    while not game.quit_flag:
        game.init_round(team_1_spawn_list=team1_sl,
                        team_2_spawn_list=team2_sl,
                        target_capture_time=TARGET_CAPTURE_TIME,
                        tank_scale=TANK_SCALE)

        # # Debug mode example:
        game.render_antennas_flag = True
        game.manual_input_flag = True
        game.team_2_list[0].manual_control_flag = True

        # Main round loop
        while game.round_run_flag:
            game.get_delta_time()
            game.check_and_handle_events()
            # -------------------------------------------------------
            # INPUT: (game.inputAI())
            # -------------------------------------------------------
            game.update()
            game.draw_background()
            # Draw under tanks here

            game.draw_tanks()
            # Draw on top of tanks here

            game.update_frame()
            game.check_state()
            # -------------------------------------------------------
            #  OUTPUT: (observations)
            # -------------------------------------------------------
            # print("FPS: ", (1/(game.last_millis/1000)))


if __name__ == "__main__":
    main()
