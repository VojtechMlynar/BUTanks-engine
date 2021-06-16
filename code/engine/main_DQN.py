"""BUTanks engine sample main script RAI 2021"""

import engine
import tanks_utility as tu
import numpy as np
import pyastar2d
import math
import time
import pygame
#import tensorflow as tf

class ArenaMasks():
    """Class to hold various masks of arena image """
    def __init__(self, arena: engine.Arena):
        """Extract features from arena image """
        NAV_MARGIN = 30 #[px] should be at least half width of tank sprite
        
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
        self.LOS_mask = arena.LOS_mask

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
    def __init__(self, masks: ArenaMasks,
                       tank: engine.Tank,
                       enemy: engine.Tank):
        self.waypoints = np.zeros((1,2))
        self.tank = tank
        self.enemy = enemy
        self.masks = masks
        
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
        x = round(self.tank.x)
        y = round(self.tank.y)
        x_e = round(self.enemy.x)
        y_e = round(self.enemy.y)

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
                if (phi_err < (2*np.pi*0.99)) & (phi_err > (2*np.pi*0.01)):
                    if phi_err > np.pi:
                        controls[0] = 1
                    elif phi_err <= np.pi:
                        controls[0] = -1
                else:
                    controls[0] = 0
                
                # Move forward if approximately facing waypoint
                if (abs(phi_err) < np.pi/10) | (abs(phi_err) > np.pi*19/10):
                    controls[1] = 1
                else:
                    controls[1] = 0

            # Check if is stuck in a wall
            distcs = self.tank.ant_distances
            close = self.tank.h*0.6
            id = np.array([0,1,9])
            if(np.min(distcs[id]) < close ):
                controls[1] = -1

        # Turret controls
        tur_rad = math.radians(self.tank.phi + self.tank.phi_rel)
        phi_tar = np.math.atan2((x_e-x), (y_e-y))
        phi_err = tur_rad - phi_tar #+ np.pi
        if (phi_err > np.pi*2) | (phi_err < 0):
            phi_err = phi_err - 2*np.pi*(phi_err // (2*np.pi))
        if (phi_err < (2*np.pi*0.99)) & (phi_err > (2*np.pi*0.01)):
            if phi_err > np.pi:
                controls[2] = 1
            elif phi_err <= np.pi:
                controls[2] = -1
        else:
            if tu.has_LOS(x, y, x_e, y_e, self.masks.LOS_mask, 10) is True:
                controls[3] = 1
            else:
                controls[3] = 0

        self.tank.input_AI(controls)
        return controls

def AI_plan_path(arena: ArenaMasks, tank: engine.Tank,
                 weights, target, pullstring = True):
    """Plan path through array with A*
    
    arena -- arena object handle
    tank -- friendly tank object handle
    weights -- numpy array float32 of weighed grid (lowest value must be 1!)
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

# ------------------------
# MAIN LOOP
# ------------------------
WIDTH, HEIGHT = 1000, 1000
MAP_FILENAME = "map2.png"
NUM_OF_ROUNDS = 4

def main():
    # Init
    game = engine.Game(MAP_FILENAME, (WIDTH, HEIGHT), NUM_OF_ROUNDS)
    t1 = [(100,500,90)]
    t2 = [(WIDTH-100, HEIGHT-500, 270)]
    # USER INIT
    tic = time.perf_counter()
    
    masks = ArenaMasks(game.arena)
    AI_red_matrix = masks.obstacles_dil
    AI_red_matrix[np.where(AI_red_matrix > 0)] = np.inf
    AI_red_matrix += 1
    AI_red_matrix = AI_red_matrix.astype(np.float32)

    # END USER INIT
    while not game.quit_flag:
        game.init_round(team_1_spawn_list=t1,
                        team_2_spawn_list=t2,
                        target_capture_time=5,
                        tank_scale=1)
        # Debug
        game.render_antennas_flag = True  
        game.manual_input_flag = True
        game.team_1_list[0].manual_control_flag = True
        # USER ROUND INIT
        red_controller = AI_controller(masks, 
                                       game.team_2_list[0],
                                       game.team_1_list[0])
        path = t2[0:1] 
        # END USER ROUND INIT
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
            path = red_controller.waypoints
            if path is not None:
                txy = np.array([game.team_2_list[0].x, game.team_2_list[0].y]
                               ,ndmin = 2)
                path_draw = np.concatenate((txy, path), axis=0)
                AI_draw_path(game, path_draw, (0,255,0))
            # end
            game.draw_tanks()
            # Place to draw on top of tanks
            game.update_frame()
            game.check_state()
            # -------------------------------------------------------
            #  OUTPUT
            toc = time.perf_counter()
            time_from_update = toc-tic


            if time_from_update > 1:
                candidates = np.array([50,50],ndmin=1)
                if candidates is not None:
                    target = candidates
                    path = AI_plan_path(masks, game.team_2_list[0],
                                    AI_red_matrix, target)
                    red_controller.set_waypoints(path[1:,:])
                tic = time.perf_counter()
            # -------------------------------------------------------
            # print("FPS: ", (1/(game.last_millis/1000)))


if __name__ == "__main__":
    main()
