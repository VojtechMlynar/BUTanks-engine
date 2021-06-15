import engine
import math
import time
import numpy as np
import pyastar2d
import pygame
import tanks_utility as tu

def AI_get_weights(arena: engine.Arena, tank: engine.Tank, enemy: engine.Tank):
    """AI_get_weights docstring"""
    ENEMY_WEIGHT = 100
    CAPTURE_WEIGHT = -300
    COVER_WEIGHT = -20
    
    scale = arena.res_scale
    baseline = arena.obstacles_bin.copy() 
    dilated = arena.obstacles_dil#.copy() 
    capture_area = arena.capture_area_mask#.copy() # TODO
    x_e, y_e = round(enemy.x/scale), round(enemy.y/scale)
    enemy_AOE = baseline.copy()
    enemy_AOE = floodfill(enemy_AOE, (x_e, y_e), (x_e, y_e), 25, ENEMY_WEIGHT,
                          max_depth = 50)

    baseline[np.where(dilated != 0)] = np.inf

    weights = baseline \
              + enemy_AOE \
              + (dilated)*COVER_WEIGHT \
              + capture_area*CAPTURE_WEIGHT
    weights += -1*np.amin(weights) + 1
    return weights

def AI_get_LOS_matrix(arena):
    terrain = arena.obstacles_bin
    size = terrain.shape
    LOS_mat = np.zeros(size)

    for i in range(0,size[0]):
        for j in range(0,i):
            pass


def floodfill(matrix, start, orig, radius, weight, max_depth = 50):
    """Floodfill docstring"""
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

def AI_plan_path(arena: engine.Arena, tank: engine.Tank,
                 weights, target, pullstring = True):
    """Docstring """

    X0 = round(tank.x/arena.res_scale)
    Y0 = round(tank.y/arena.res_scale)

    path = pyastar2d.astar_path(weights,
        (X0, Y0), (target[0], target[1]), allow_diagonal = False)

    if path is not None:
        path_scaled = path*arena.res_scale + math.floor(arena.res_scale/2)
        if pullstring is True:
            path_optimal = tu.string_pulling(path_scaled, arena.dilated_scaled)
            return path_optimal
        else:
            return path_scaled
    else:
        return np.array([X0, Y0]) 

def AI_draw_path(game: engine.Game, path, colour):
    for i in range(np.size(path,0)-1):
        pygame.draw.line(game.WINDOW, colour, 
            (path[i,]), (path[i+1,]), 2)
    


# MAIN LOOP
def main():
    game = engine.Game(1)
    # init()
    map_size = (1000,1000)
    t1 = [(100,100,0)]
    t2 = [(map_size[0]-100,map_size[1]-100,180)]

    
    while not game.quit_flag:
        game.init_round(team_1_spawn_list=t1,
                        team_2_spawn_list=t2,
                        map_filename="map1.png",
                        target_capture_time=5,
                        window_size=map_size)
        # Debug:
        game.render_antennas_flag = True  
        game.manual_input_flag = True
        game.team_1_list[0].manual_control_flag = True

        tic = time.perf_counter()
        path = None

        while game.round_run_flag:
            game.get_delta_time()
            game.check_and_handle_events()
            # -------------------------------------------------------
            # INPUT game.inputAI()
            # -------------------------------------------------------
            game.update()
            game.draw()
            game.check_state()
            # -------------------------------------------------------
            #  OUTPUT
            toc = time.perf_counter()
            time_from_update = toc-tic

            if time_from_update > 1:
                AI_red_matrix = AI_get_weights(game.arena, 
                    game.team_2_list[0], game.team_1_list[0])
                candidates = np.where(AI_red_matrix == np.amin(AI_red_matrix))

                if candidates is not None:
                    target = (candidates[0][0], candidates[1][0])
                    path = AI_plan_path(game.arena, game.team_2_list[0],
                                    AI_red_matrix, target)
                tic = time.perf_counter()

            if path is not None:
                AI_draw_path(game, path, (0,255,0))
                pygame.display.update()

            # -------------------------------------------------------
            # print("FPS: ", (1/(game.last_millis/1000)))

if __name__ == "__main__":
    main()