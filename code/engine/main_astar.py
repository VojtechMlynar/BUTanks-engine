
import engine
import math
import numpy as np

def AI_get_weights(arena: engine.Arena, tank: engine.Tank, enemy: engine.Tank):
    ENEMY_WEIGHT = 100
    CAPTURE_WEIGHT = -100
    COVER_WEIGHT = -20
    
    scale = arena.res_scale
    baseline = arena.obstacles_bin.copy() 
    dilated = arena.obstacles_dil#.copy() 
    capture_area = arena.capture_area_mask#.copy() # TODO
    x_e, y_e = round(enemy.x/scale), round(enemy.y/scale)
    enemy_AOE = baseline.copy()
    enemy_AOE = floodfill(enemy_AOE, (x_e, y_e), (x_e, y_e), 25, ENEMY_WEIGHT)

    baseline[np.where(baseline != 0)] = np.inf

    weights = baseline \
              + enemy_AOE \
              + (dilated)*COVER_WEIGHT \
              + capture_area*CAPTURE_WEIGHT
    weights += -1*np.amin(weights) + 1

    return weights


def floodfill(matrix, start, orig, radius, weight, diag = False):
    dist = math.sqrt((start[0]-orig[0])**2 + (start[1]-orig[1])**2)
    if dist < radius:
        x, y = start[0], start[1]
        matrix[x,y] = weight
        if(matrix[x-1, y] == 0):
            matrix = floodfill(matrix, (x-1, y), orig, radius, weight)
        if(matrix[x+1, y] == 0):
            matrix = floodfill(matrix, (x+1, y), orig, radius, weight)        
        if(matrix[x, y-1] == 0):
            matrix = floodfill(matrix, (x, y-1), orig, radius, weight)
        if(matrix[x, y+1] == 0):
            matrix = floodfill(matrix, (x, y+1), orig, radius, weight)
        if diag is True:
            if(matrix[x-1, y-1] == 0):
                matrix = floodfill(matrix, (x-1, y-1), orig, radius, weight)
            if(matrix[x+1, y-1] == 0):
                matrix = floodfill(matrix, (x+1, y-1), orig, radius, weight)        
            if(matrix[x+1, y+1] == 0):
                matrix = floodfill(matrix, (x+1, y+1), orig, radius, weight)
            if(matrix[x-1, y+1] == 0):
                matrix = floodfill(matrix, (x-1, y+1), orig, radius, weight)     
    
    return matrix


# MAIN LOOP
def main():
    game = engine.Game()
    # init()
    game.init_round()

    while game.run:
        game.get_delta_time()
        game.check_for_events()
        # -------------------------------------------------------
        # AI INPUT (intern)
        # -------------------------------------------------------
        game.update()
        game.draw()
        game.check_state()
        # -------------------------------------------------------
        #  AI outpput
        AI_get_weights(game.arena, game.team_1_list[0], game.team_2_list[0])
        # -------------------------------------------------------
        #print("FPS: ", (1/(game.last_millis/1000)))
        #print("Phi:", game.team_1_list[0].phi)

if __name__ == "__main__":
    main()