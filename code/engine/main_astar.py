import engine
import math
import time
import numpy as np
import pyastar2d
import pygame
import tanks_utility as tu

class ArenaMasks():
    """Class to hold various masks of arena image """
    def __init__(self, arena: engine.Arena):
        """Extract features from arena image """
        NAV_MARGIN = 25 #[px] should be at least half width of tank sprite
        
        # Create binary mask of obstacels
        self.obstacles_bin = arena.alpha_arr.copy()
        self.obstacles_bin[np.where(self.obstacles_bin < 255)] = 0
        self.obstacles_bin[np.where(self.obstacles_bin == 255)] = 1
        self.obstacles_bin = self.obstacles_bin.astype(np.float32)
        
        # Dilate image to get safety navigation margin
        size = arena.image.get_size()
        dil = math.ceil(NAV_MARGIN / arena.res_scale[0])
        self.obstacles_dil = tu.dilate_image(self.obstacles_bin, dil, "max")
        self.dilated_scaled = tu.resize_image(self.obstacles_dil, 
                                              size[0], size[1])
        # Extract capture area
        self.capture_area_mask = arena.alpha_arr.copy()
        self.capture_area_mask[np.where(
            (self.capture_area_mask > 250))] = 0
        self.capture_area_mask[np.where(
            (self.capture_area_mask > 10))] = 1

        self.res_scale = arena.res_scale
    

def AI_get_weights(arena: ArenaMasks, tank: engine.Tank, enemy: engine.Tank):
    """Get influence map of game - lower value means more attractive
    
        arena -- game arena object handle
        tank -- friendly tank object handle
        enemy -- enemy tank object handle
    """
    ENEMY_WEIGHT = 100
    CAPTURE_WEIGHT = -300
    COVER_WEIGHT = -20
    
    scale = arena.res_scale
    baseline = arena.obstacles_bin.copy() 
    dilated = arena.obstacles_dil#.copy() 
    capture_area = arena.capture_area_mask#.copy() # TODO
    x_e, y_e = round(enemy.x/scale[0]), round(enemy.y/scale[1])
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

class AI_controller():
    """AI controller class
    
        attributes:
            waypoints -- numpy array n x 2 of waypoints
            tank -- handle for tank object

        methods:
            __init__ -- assign tank handle for controller

            set_waypoints(path) -- provide new path (numpy array) 
                                   for controller

            controls_output -- control assigned tank to move to the waypoint
                with highest priority. Returns controls vector
    """
    def __init__(self, tank: engine.Tank):
        self.waypoints = np.zeros((1,2))
        self.tank = tank
        
    def set_waypoints(self, path):
        self.waypoints = path

    def controls_output(self):
        """Path following get actions 
        
            Output: List of integers in format:
                    [0]: body rotation 1/0/-1
                    [1]: body forward/backwards movement 1/0/-1
                    [2]: turret rotation 1/0/-1
                    [3]: shoot 1/0
        """
        controls = [0, 0, 0, 0]

        phi_rad = math.radians(self.tank.phi)
        x = self.tank.x
        y = self.tank.y
        
        waypoints_left = np.size(self.waypoints,0) 

        if waypoints_left > 0:
            target = self.waypoints[0,]
            waypoint_dist = abs(target[0]-x)\
                          + abs(target[1]-y)
            if waypoint_dist < 20:
                self.waypoints = np.delete(self.waypoints, 0, 0)
            else:
                phi_tar = np.math.atan2((target[0]-x), (target[1]-y))
                phi_err =  phi_rad - phi_tar

                if (phi_err > np.pi*2) | (phi_err < 0):
                    phi_err = phi_err - 2*np.pi*(phi_err // (2*np.pi))
                
                # Turn towards the waypoint
                if (phi_err < (2*np.pi*35/36)) & (phi_err > (2*np.pi*1/36)):
                    if phi_err > np.pi:
                        controls[0] = 1
                    elif phi_err <= np.pi:
                        controls[0] = -1
                else:
                    controls[0] = 0
                
                # Move forward if approximately facing waypoint
                if (abs(phi_err) < np.pi/6) | (abs(phi_err) > np.pi*5/6):
                    controls[1] = 1
                else:
                    controls[1] = 0

        self.tank.input_AI(controls)
        return controls


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


def AI_plan_path(arena: ArenaMasks, tank: engine.Tank,
                 weights, target, pullstring = True):
    """Plan path through array with A*
    
    arena -- arena object handle
    tank -- friendly tank object handle
    weights -- numpy array float 32 of weighed grid (lowest value must be 1!)
    target -- target coordinates
    pullstring -- (default True) optimize path for non-grid movement
                    by string pulling method 
    
    """
    X0 = round(tank.x/arena.res_scale[0])
    Y0 = round(tank.y/arena.res_scale[1])

    path = pyastar2d.astar_path(weights,
        (X0, Y0), (target[0], target[1]), allow_diagonal = False)

    if path is not None:
        path_scaled = path*arena.res_scale[0] + math.floor(arena.res_scale[0]/2)
        if pullstring is True:
            path_optimal = tu.string_pulling(path_scaled, arena.dilated_scaled)
            return path_optimal
        else:
            return path_scaled
    else:
        return np.array([X0, Y0]) 

def AI_draw_path(game: engine.Game, path, colour):
    """Draw lines on path coordinates """
    for i in range(np.size(path,0)-1):
        pygame.draw.line(game.WINDOW, colour, 
            (path[i,]), (path[i+1,]), 2)
    

# MAIN LOOP

WIDTH, HEIGHT = 1000, 1000
MAP_FILENAME = "map1.png"
NUM_OF_ROUNDS = 4

def main():
    game = engine.Game(MAP_FILENAME, (WIDTH,HEIGHT), NUM_OF_ROUNDS)
    
    # init()
    t1 = [(100,100,0)]
    t2 = [(WIDTH-100,HEIGHT-100,180)]
    # ------------
    # USER INIT

    tic = time.perf_counter()
    path = None 
    masks = ArenaMasks(game.arena)  

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

        # ROUND USER INIT ---
        red_controller = AI_controller(game.team_2_list[0])
        #-----
        while game.round_run_flag:
            game.get_delta_time()
            game.check_and_handle_events()
            # -------------------------------------------------------
            # INPUT game.inputAI()
            ctrls = red_controller.controls_output()
            # -------------------------------------------------------
            game.update()
            game.draw_background()
            # Place to draw under tanks
            # ----
            game.draw_tanks()
            # Place to draw on top of tanks
            path = red_controller.waypoints
            if path is not None:
                txy = np.array([game.team_2_list[0].x, game.team_2_list[0].y]
                               ,ndmin = 2)
                path_draw = np.concatenate((txy, path), axis=0)
                AI_draw_path(game, path_draw, (0,255,0))
            # ----
            game.update_frame()
            game.check_state()
            # -------------------------------------------------------
            #  OUTPUT
            toc = time.perf_counter()
            time_from_update = toc-tic

            if time_from_update > 1:
                AI_red_matrix = AI_get_weights(masks, 
                                               game.team_2_list[0],
                                               game.team_1_list[0])

                candidates = np.where(AI_red_matrix == np.amin(AI_red_matrix))
                if candidates is not None:
                    target = (candidates[0][0], candidates[1][0])
                    path = AI_plan_path(masks, game.team_2_list[0],
                                    AI_red_matrix, target)
                    red_controller.set_waypoints(path[1:,:])
                tic = time.perf_counter()
            # -------------------------------------------------------
            # print("FPS: ", (1/(game.last_millis/1000)))

if __name__ == "__main__":
    main()