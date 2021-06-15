"""BUTanks engine sample main script RAI 2021"""

import engine


WIDTH, HEIGHT = 1000, 1000
MAP_FILENAME = "map1.png"
NUM_OF_ROUNDS = 4


def main():
    game = engine.Game(MAP_FILENAME, (WIDTH,HEIGHT), NUM_OF_ROUNDS)
    
    # init()
    t1 = [(100,100,0)]
    t2 = [(WIDTH-100,HEIGHT-100,180)]

    while not game.quit_flag:
        game.init_round(team_1_spawn_list=t1,
                        team_2_spawn_list=t2,
                        target_capture_time=5,
                        tank_scale=1)
        # Debug:
        game.render_antennas_flag = True  
        game.manual_input_flag = True
        game.team_1_list[0].manual_control_flag = True

        while game.round_run_flag:
            game.get_delta_time()
            game.check_and_handle_events()
            # -------------------------------------------------------
            # INPUT game.inputAI()
            # -------------------------------------------------------
            game.update()
            game.draw_background()
            # Place to draw under tanks
            game.draw_tanks()
            # Plac to draw on top of tanks
            game.update_frame()
            game.check_state()
            # -------------------------------------------------------
            #  OUTPUT
            # -------------------------------------------------------
            # print("FPS: ", (1/(game.last_millis/1000)))

if __name__ == "__main__":
    main()