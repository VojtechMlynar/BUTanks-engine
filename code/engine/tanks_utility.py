"""
BUTanks engine utility module

contains various function used in the engine, mainly image manipulation
"""
import pyastar2d
import pygame
import math
import numpy as np

def string_pulling(path, environment):
    """Straighten path from grid to account for non-grid movement
    
    path -- numpy array n x 2 of [x,y] path waypoints
    environment -- numpy array of movement blocking terrain

    returns m x 2 numpy array of optimized path
    """
    STRIDE=5 # Adjust this parameter to improve performance, or avoid errors

    path = path.astype(int)

    start = 0 # starting node
    target = path[-1,]

    n = np.size(path,0)
    k = 1
    opt_path = np.array(path[0,], ndmin=2)
    while k < n:
        if has_LOS(path[start,0], path[start,1], 
                path[k,0], path[k,1], environment,stride=STRIDE) is False:
            opt_path = np.append(opt_path, np.atleast_2d(path[k-1]), 0)
            start = k
        k += 1
    
    opt_path = np.append(opt_path, np.atleast_2d(path[-1,]), 0)

    return opt_path    

def has_LOS(x0, y0, xt, yt, environment, stride = 1):
    """Check if [x0, y0] has line of sight to [xt, yt]
    
    Check pixel by pixel whether environment block LOS between two coordinates
    x0, y0 -- first pixel (integer)
    xt, yt -- target pixel (integer)
    environment -- numpy array of LOS blocking terrain
                -- 0 means non blocking, other values blocking 

    return True/False whether there is/is not LOS between points            
    """
    if(environment[x0,y0] != 0):
        return False
        
    if (xt-x0) == 0:
        kinv = 0
        if (yt-y0) == 0:
            return True
    else:    
        k = (yt-y0)/(xt-x0)
        if k != 0:
            kinv = 1/k

    passed = True
    
    if abs(x0-xt) >= abs(y0-yt):
        for x in range(x0, xt+1, stride*np.sign(xt-x0)):
            y = round(y0 + k*(x-x0))
            if environment[x,y] != 0:
                passed = False
                break
    else:
        for y in range(y0, yt+1, stride*np.sign(yt-y0)):
            x = round(x0 + kinv*(y-y0))
            if environment[x,y] != 0:
                passed = False
                break  
    
    return passed

def cast_line(x0, y0, phi, env, stride = 15):
    """Project line from point at an angle until it reaches obstacle 
    
    x0, y0 -- starting point
    phi -- line angle
    env -- numpy array of environment
    stride -- function checks only each nth pixel and then backsteps to find
              intersection point. Higher value ~ better performance, but can
              skip corners, or entire walls when set too great.

    returns euclidean distance to the obstacle and that point's coordinates 
    """

    if phi > 2*np.pi:
        phi = phi - 2*np.pi*(phi // (2*np.pi))
    elif phi < 0:
        phi = phi - 2*np.pi*(phi // (2*np.pi))

    x = round(x0)
    y = round(y0) 

    i = 0
    k = np.math.tan(phi)

    if abs(k) > 1: # Y axis is primary
        while (env[x,y] == 0):
            if (phi > 0) & (phi < np.pi):
                y += stride
            else:
                y -= stride
            
            x = round(x0 + 1/k*(y-y0))
        # Backstepping
        while (env[x, y] != 0):
            if (phi > 0) & (phi < np.pi):
                y -= 1
            else:
                y += 1
            x = round(x0 + 1/k*(y-y0))

    else: # X axis is primary
        while (env[round(x),round(y)] == 0):
            if (phi > np.pi*3/2) | (phi < np.pi/2):
                x += stride
            else:
                x -= stride 
            y = round(y0 + k*(x-x0))
        # Backstepping
        while (env[round(x),round(y)] != 0):
            if (phi > np.pi*3/2) | (phi < np.pi/2):
                x -= 1
            else:
                x += 1 
            y = round(y0 + k*(x-x0))

    dist = np.sqrt((x-x0)**2 + (y-y0)**2)
    return dist, x, y

def is_close(a,b,tolerance):
    """Check if abs(a-b) is within tolerance """
    if abs(a-b) < tolerance:
        return True
    else:
        return False

def dilate_image(im_src, dilation_level = 1, method = "min"):
    """Use min or max filter to dilate/erode image
    
    Uses min/max filter on kernel of specified size. Kernel size is 
    dilation_level*2 + 1. Replaces middle value with min/max value from
    the kernel area. Works best on binary (highly contrast) images

    Arguments:
    im_src -- numpy array of pixel values
    dilation_level -- radius of kernel (default = 1)
    method -- "min" or "max" filter (default "min")
    """

    ker_size = 2*dilation_level + 1
    skip = dilation_level

    # pad array with constants
    im_pad = np.pad(array=im_src, pad_width=skip, mode='constant')
    pad_shape = im_pad.shape
    im_copy = im_src.copy()

    # iterate through array and apply filter
    for r in range(skip, pad_shape[0]-skip-1):
        for c in range(skip, pad_shape[1]-skip-1):
            if method == "min":
                im_copy[r-skip, c-skip] = np.min(
                    im_pad[(r-skip):(r+skip+1), (c-skip):(c+skip+1)])
            elif method == "max":
                im_copy[r-skip, c-skip] = np.max(
                    im_pad[(r-skip):(r+skip+1), (c-skip):(c+skip+1)])
    
    return im_copy

def resize_image(im_src, new_w, new_h):
    """Resize input image to specified dimensions
    
    im_src -- numpy array of pixel values
    new_w -- desired image width
    new_h -- desired image height
    """

    original_image = im_src.copy()
    width, height = new_w, new_h
    resize_im = np.zeros(shape=(width,height))
    
    for W in range(width):
        for H in range(height):
            new_width = int( W * original_image.shape[0] / width )
            new_height = int( H * original_image.shape[1] / height )
            resize_im[W][H] = original_image[new_width][new_height]
    
    return resize_im

class ArenaMasks():
    """Class to hold various masks of arena image """
    def __init__(self, arena):
        """Extract features from arena image
        
        arena -- game.Arena object handle
        """
        NAV_MARGIN = 30 #[px] should be at least half width of tank sprite
        
        # Create binary mask of obstacels
        self.obstacles_bin = arena.alpha_arr.copy()
        self.obstacles_bin[np.where(self.obstacles_bin < 255)] = 0
        self.obstacles_bin[np.where(self.obstacles_bin == 255)] = 1
        self.obstacles_bin = self.obstacles_bin.astype(np.float32)
        self.size = self.obstacles_bin.shape

        # Dilate image to get safety navigation margin
        size = arena.image.get_size()
        self.size_big = size
        dil = math.ceil(NAV_MARGIN / arena.res_scale[0])
        self.obstacles_dil = dilate_image(self.obstacles_bin, dil, "max")
        self.dilated_scaled = resize_image(self.obstacles_dil, 
                                              size[0], size[1])
        # Extract capture area
        self.capture_area_mask = arena.alpha_arr.copy()
        self.capture_area_mask[np.where(
            (self.capture_area_mask > 250))] = 0
        self.capture_area_mask[np.where(
            (self.capture_area_mask > 10))] = 1

        self.res_scale = arena.res_scale
        self.LOS_mask = arena.LOS_mask

class AIController():
    """AI controller class
    
        attributes:
            waypoints -- numpy array n x 2 of waypoints
            tank -- handle for tank object
            enemy -- handle for adversary tank object
            toothless -- parameter to determine whether tank will/won't
                         shoot at the enemy tank
        methods:
            __init__ -- assign tank handle for controller

            set_waypoints(path) -- provide new path (numpy array) 
                                   for controller

            controls_output -- control assigned tank to move to the waypoint
                with highest priority. Returns controls vector

            plan_path_astar -- Plan cheapest path through weights array with A*

            draw_path -- draw provided path
    """
    def __init__(self, masks: ArenaMasks,
                       tank,
                       enemy,
                       toothless=False):
        """Init AI_controller Class
        
            arguments:
                masks -- ArenaMasks object handle
                tank -- engine.Tank object handle (self)
                enemy -- engine.Tank object handle (adversarial tank)
                toothless -- bool: True-> will shoot at the enemy
                                   False-> will not shoot at the enemy
        """
        self.waypoints = np.array([tank.x, tank.y],ndmin=2)
        self.tank = tank
        self.enemy = enemy
        self.masks = masks
        self.toothless = toothless
        
    def set_waypoints(self, path):
        """Set waypoints attribute macro
        
        path -- numpy array of (2 x n) dimension, integer dtype
        """
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
        # saturate phi
        if (phi_err > np.pi*2) | (phi_err < 0):
            phi_err = phi_err - 2*np.pi*(phi_err // (2*np.pi))

        if (phi_err < (2*np.pi*0.99)) & (phi_err > (2*np.pi*0.01)):
            if phi_err > np.pi:
                controls[2] = 1
            elif phi_err <= np.pi:
                controls[2] = -1
        else:
            if (has_LOS(x, y, x_e, y_e, self.masks.LOS_mask, 10) is True) \
                    & (self.toothless is False):
                controls[3] = 1 # shoot
            else:
                controls[3] = 0 # dont shoot

        self.tank.input_AI(controls) # apply control to tank class
        return controls

    def plan_path_astar(self, weights, target, pullstring = True):
        """Plan path through array with A*
        
        arena -- arena object handle
        weights -- numpy array float32 of weighed grid (lowest value must be 1!)
        target -- target coordinates
        pullstring -- (default True) optimize path for non-grid movement
                        by string pulling method 
        """
        arena = self.masks
        tank = self.tank
        X0 = round(tank.x/arena.res_scale[0])
        Y0 = round(tank.y/arena.res_scale[1])

        if (X0<0) | (Y0<0) | (X0>arena.size[0]) | (Y0>arena.size[1]):
            return np.array([X0, Y0],ndmin=2)
            
        path = pyastar2d.astar_path(weights,
            (X0, Y0), (target[0], target[1]), allow_diagonal = False)

        if (path is not None):
            path_scaled = path*arena.res_scale[0]\
                        + math.floor(arena.res_scale[0]/2)
            if pullstring is True:
                path_optimal = string_pulling(path_scaled,
                                              arena.dilated_scaled)
                return path_optimal
            else:
                return path_scaled
        else:
            return np.array([X0, Y0],ndmin=2) 

    def draw_path(game, path, colour):
        """Draw lines on path coordinates
        
            game -- engine.Game class object handle
            path -- numpy array of waypoints (n x 2)
            colour -- tuple of line colour
        """
        for i in range(np.size(path,0)-1):
            pygame.draw.line(game.WINDOW, colour, 
                (path[i,]), (path[i+1,]), 2)
