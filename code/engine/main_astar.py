"""BUTanks engine with basic influence map and A* pathfinding

Influence map is updated each T_UPDATE. It takes into account positions of
enemy tanks and capture area positions. In theory, agent should try to avoid
dangerous paths.

Note: Floodfill algorithm probably isn't working properly, might need further
debugging

requires: pygame, pyastar2d (might require to install manually - just
    copy Github repo and follow instructions)

Author: Vojtech Mlynar
Date: 18.06.2021
"""

import engine
import math
import time
import numpy as np
import tanks_utility as tu

def AI_get_weights(arena: tu.ArenaMasks, tank: engine.Tank, enemy: engine.Tank):
    """Get influence map of game - lower value means more attractive
    
        arena -- game arena object handle
        tank -- friendly tank object handle
        enemy -- enemy tank object handle
    """
    ENEMY_WEIGHT = 10000
    CAPTURE_WEIGHT = -300
    COVER_WEIGHT = -20 # cover is unreacheable, parametr is left for the future
    
    scale = arena.res_scale
    baseline = arena.obstacles_bin.copy() 
    dilated = arena.obstacles_dil
    capture_area = arena.capture_area_mask
    x_e, y_e = round(enemy.x/scale[0]), round(enemy.y/scale[1])
    #approximation of "danger area" caused by enemies
    enemy_AOE = baseline.copy()
    enemy_AOE = floodfill(enemy_AOE, (x_e, y_e), (x_e, y_e), 100, ENEMY_WEIGHT,
                          max_depth = 70)
    # Make obstacles impassable
    baseline[np.where(dilated != 0)] = np.inf

    weights = baseline \
              + enemy_AOE \
              + (dilated)*COVER_WEIGHT \
              + capture_area*CAPTURE_WEIGHT
    # Shift to make lowest value equal 1
    weights += -1*np.amin(weights) + 1
    return weights


def floodfill(matrix, start, orig, radius, weight, max_depth = 50):
    """Recursively floodfill matrix stopping on obstacles
    
        matrix -- numpy array to work on
        start -- starting coordinates [x0,y0] (changes through calls)
        orig -- coordinates of origin (does not change through calls)
        raidus -- euclidean radius of how far to fill array
        weight -- number to fill matrix with
        max_depth -- (default 50) limit to how deep can recursions go  
    """
    dist = math.sqrt((start[0]-orig[0])**2 + (start[1]-orig[1])**2)
    if (dist < radius) & (max_depth > 0):
        x, y = start[0], start[1]
        matrix[x,y] = weight
        if(matrix[x-1, y] == 0):
            matrix = floodfill(matrix, (x-1, y), orig, radius, weight, max_depth-1)
        if(matrix[x+1, y] == 0):
            matrix = floodfill(matrix, (x+1, y), orig, radius, weight, max_depth-1)        
        if(matrix[x, y-1] == 0):
            matrix = floodfill(matrix, (x, y-1), orig, radius, weight, max_depth-1)
        if(matrix[x, y+1] == 0):
            matrix = floodfill(matrix, (x, y+1), orig, radius, weight, max_depth-1)   
    return matrix
    

# MAIN LOOP

WIDTH, HEIGHT = 1000, 1000 # Windows size in pixels
MAP_FILENAME = "map3.png" # Map to play (from ../code/assets/maps)
NUM_OF_ROUNDS = 4 # How many rounds do you want to play
T_UPDATE = 1 # How often will AI search new path

def main():
    game = engine.Game(MAP_FILENAME, (WIDTH,HEIGHT), NUM_OF_ROUNDS)
    
    # init()
    t1 = [(100,850,180)]
    t2 = [(WIDTH-500,900,180)]
    # ------------
    # USER INIT

    tic = time.perf_counter()
    path = None 
    masks = tu.ArenaMasks(game.arena)  

    # ------------
    while not game.quit_flag:
        game.init_round(team_1_spawn_list=t1,
                        team_2_spawn_list=t2,
                        target_capture_time=5,
                        tank_scale=1)
        # Debug:
        game.render_antennas_flag = True  
        game.manual_input_flag = True
        game.team_1_list[0].manual_control_flag = True

        red_controller = tu.AIController(masks,
                                         game.team_2_list[0],
                                         game.team_1_list[0],
                                         toothless=False)

        while game.round_run_flag:
            game.get_delta_time()
            game.check_and_handle_events()
            # -------------------------------------------------------
            # INPUT game.inputAI()
            # Push control inputs to the tank object
            ctrls = red_controller.controls_output()
            # -------------------------------------------------------
            game.update()
            game.draw_background()
            # Place to draw under tanks
            # -------------------------
            game.draw_tanks()
            # Place to draw on top of tanks
            path = red_controller.waypoints
            if path is not None:
                txy = np.array([game.team_2_list[0].x, game.team_2_list[0].y]
                               ,ndmin = 2)
                path_draw = np.concatenate((txy, path), axis=0)
                red_controller.draw_path(game, path_draw, (0,255,0))
            # ---------------------------------------------
            game.update_frame()
            game.check_state()
            # -------------------------------------------------------
            #  OUTPUT
            toc = time.perf_counter()
            time_from_update = toc-tic

            if time_from_update > T_UPDATE:
                AI_red_matrix = AI_get_weights(masks, 
                                               game.team_2_list[0],
                                               game.team_1_list[0])
                # Choose most valuable point
                candidates = np.where(AI_red_matrix == np.amin(AI_red_matrix))
                if candidates is not None:
                    # And find cheapest way with A*
                    target = (candidates[0][0], candidates[1][0])
                    path = red_controller.plan_path_astar(AI_red_matrix,
                                                         target)
                    # Set controller
                    red_controller.set_waypoints(path[1:,:])
                tic = time.perf_counter()
            # -------------------------------------------------------
            # print("FPS: ", (1/(game.last_millis/1000)))

if __name__ == "__main__":
    main()