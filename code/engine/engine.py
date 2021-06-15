""" BUTanks engine v0.0.8 - dev_DB branch

Build on Python 3.9.1 and Pygame 2.0.1.
Python 3.6+ required.

TODO:
[x] Suitable for more than 1v1
[x] Understand fixed dt (for learning alghorithms)
[ ] Display game stats
[ ] Game states
[ ] I/O and its link with AI
[=] Comments (PARTIALLY)
[=] Clean (PARTIALLY)
""" 

import os
import math
import itertools
import numpy as np
import pygame
import sys
import tanks_utility as tu
from pathlib import Path

# MAP
MAP_FILENAME = "map1.png"
WIDTH, HEIGHT = 1000, 1000
WHITE = (100,100,100)  # HACK Dark mode 

# Game settings
TARGET_CAPTURE_TIME = 5   # Time that must be spent in capture area [s]
NUM_OF_ROUNDS = 4

# Game rendering
TARGET_FPS = 60           # Set very high for non graphic learning (e.g. 10000)
RENDER_ALL_FRAMES = True  # Set False for non graphic learning
TARGET_FRAME = 100        # Every n-th rendered frame  
FIXED_DT = None     # None for graphic mode otherwise reccomended value is 1/25 

# Tank settings
FORWARD_SPEED = 150      # [px/s]
BACKWARD_SPEED = 80      # [px/s]
TURN_SPEED = 250         # [deg/s]
TURRET_TURN_SPEED = 250  # [deg/s]
TSHELL_SPEED = 350       # [px/s]
GUN_COOLDOWN = 1         # [s]
MAX_HEALTH = 5

# Collision handling
MAX_WALL_COLLISION_ITER = 20

# PATHS
p = Path(__file__).parents[1]
ASSETS_DIR = os.path.join(p,"assets")
MAPS_DIR = os.path.join(ASSETS_DIR,"maps")
IMGS_DIR = os.path.join(ASSETS_DIR,"images")

# INTERNAL
NAV_MARGIN = 25

# TODO: Check comments esp: (i_frame, master_list)
class Game():
    """Game object holds important data for proper game function.

    Attributes:
            target_capture_time: int
                Target capture time in seconds.
            dt: float
                Time step between "frames" in seconds. 
                (Also known as delta time.)
            team_1_capture_time: float
                Team 1 captured time in seconds.
            team_2_capture_time: float
                Team 2 captured time in seconds.
            team_1_alive: int
                Number of Team 1 members alive.
            team_2_alive: int
                Number of Team 2 members alive.
            team_1_captured_flag: bool
                True if Team 1 captured area.
            team_2_captured_flag: bool
                True if Team 2 captured area.
            targetFPS: int
                The Frames Per Second (FPS) cap. 
                Set very high for non graphic learning.
            render_all_frames: bool
                Set False for rendering only target frames.
            target_frame: int
                Target frame number.
                Only multiplies will be rendered.
            i_frame: int
                Current frame number.
            dt_fixed: None or float
                If not None, sets fixed time step in seconds.
                Usefull for non graphic learning.
            team_1_list: list
                List of Team 1 Tank objects.
            team_2_list: list
                List of Team 2 Tank objects.
            master_list: list
                team_1_list + team_2_list created after their creation!  
    """

    def __init__(self, window_size: tuple, num_of_rounds: int = 1):
        """Initializes Game class object and pygame window."""

        self.num_of_rounds = num_of_rounds
        self.win_list = []
        self.quit_Flag = 0
        self.quit = 0

        self.i_round = 1

        # WINDOW init
        pygame.init()
        self.WIN = pygame.display.set_mode((window_size[0], window_size[1]))
        pygame.display.set_caption("BUTanks engine")
        print("BUTanks engine initializing... Have fun!\n")


    def init_round(self, team_1_spawn_list: list, team_2_spawn_list: list,
                    map_filename: str, manual_input: bool = False):


        # Settings
        self.target_capture_time = 5
                
        # Stats
        self.dt = 0
        self.team_1_capture_time = 0
        self.team_2_capture_time = 0
        self.win_team = None

        # State related
        self.team_1_captured_flag = False
        self.team_2_captured_flag = False
        self.team_1_destroyed_flag = False
        self.team_2_destroyed_flag = False

        # Rendering
        self.targetFPS = TARGET_FPS
        self.render_all_frames = RENDER_ALL_FRAMES
        self.target_frame = TARGET_FRAME
        self.i_frame = 0
        self.dt_fixed = FIXED_DT

        # Team lists
        self.team_1_list = []
        self.team_2_list = []

        self.arena = None
        self.fpsClock = None
        self.last_millis = 0
        self.round_run = False

        self.render_antenas = False
        self.manual_input = manual_input

        self.arena = Arena(map_filename)
        self.fpsClock = pygame.time.Clock()
        # Objects
        for item in team_1_spawn_list:
            self.team_1_list.append(Tank(item[0], item[1], item[2], 0, 
                                        "tank_1.png", "turret_1.png", 
                                        "tank_shell_1.png"))
        for item in team_2_spawn_list:
            self.team_2_list.append(Tank(item[0], item[1], item[2], 0,
                                        "tank_2.png", "turret_2.png", 
                                        "tank_shell_2.png"))

        self.master_list = self.team_1_list + self.team_2_list
        
        self.team_1_alive = len(team_1_spawn_list)
        self.team_2_alive = len(team_2_spawn_list)
        
        self.round_run = True
        print(f'Round {self.i_round}', 
              f'({self.team_1_alive}v{self.team_2_alive}):')

    def get_delta_time(self):
        
        if self.dt_fixed is None:
            self.dt = self.last_millis/1000
        else:
            self.dt = self.dt_fixed

    def check_for_events(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_Flag = True
                self.round_run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    if self.render_antenas:
                        self.render_antenas = False
                    else:
                        self.render_antenas = True

            if self.manual_input:
                if event.type == pygame.KEYDOWN:
                    for tank in self.master_list:
                        tank.input_manual(event.key, 1)
                if event.type == pygame.KEYUP:
                    for tank in self.master_list:
                        tank.input_manual(event.key, 0)

    def update(self):

        for tank in self.master_list:
            tank.update(self.dt, self.arena)
        # Check and resolve eventual collisions
        handleCollisions(self.arena, self)

    def draw(self):

        if (self.i_frame == self.target_frame) or (self.render_all_frames):
            self.WIN.fill(WHITE)
            pygame.sprite.RenderPlain(self.arena).draw(self.WIN)
            for tank in self.master_list:
                tank.draw(self)
            pygame.display.update()
            self.i_frame = 0
        else:
            self.i_frame += 1
    
    def check_state(self):
        """Check game conditions and eventually switch game state."""
        
        # Check if team had already captured area
        if self.team_1_capture_time >= self.target_capture_time:
            self.team_1_captured_flag = True
            print("Team 1 captured area!")
        if self.team_2_capture_time >= self.target_capture_time:
            self.team_2_captured_flag = True
            print("Team 2 captured area!")
        # Check if whole team is destroyded
        if self.team_1_alive == 0:
            self.team_1_destroyed_flag = True
            print("- Team 1 destroyed!")
        if self.team_2_alive == 0:
            self.team_2_destroyed_flag = True
            print("- Team 2 destroyed!")

        if self.team_1_captured_flag and self.team_2_captured_flag:
            self.win_team = 0  # Tie
        elif self.team_1_captured_flag:
            self.win_team = 1
        elif self.team_2_captured_flag:
            self.win_team = 2

        if self.team_1_destroyed_flag and self.team_2_destroyed_flag:
            self.win_team = 0  # Tie
        elif self.team_1_destroyed_flag:
            self.win_team = 2
        elif self.team_2_destroyed_flag:
            self.win_team = 1

        if self.win_team is not None:
            self.win_list.append(self.win_team)
            if self.win_team == 0:
                win_str = "Round ended in a tie!\n"
            else:
                win_str = f'Team {self.win_team} won!\n'
            print(win_str)
            self.round_run = False

            if self.i_round == self.num_of_rounds:
                self.quit_Flag = True
            else:
                self.i_round += 1

        if self.quit_Flag:
            pygame.quit()
            print("Final win list (0 = Tie): ", self.win_list)
            print("Quitting.\n")
            return

        self.last_millis = self.fpsClock.tick(self.targetFPS)


class Arena(pygame.sprite.Sprite):
    """Arena sprite with CaptureArea attribute.
    
    Important attribute:
        CaptureArea: CaptureArea object for capture area detection.  
    """
    
    def __init__(self, img_filename: str):
        """
        Parameters:
            img_filename: Arena image filename.

        Notes:
            Image must have non transparent walls and transparent background.    
        """

        super(Arena, self).__init__()
        self.x = 0
        self.y = 0
        self.image = pygame.image.load(os.path.join(MAPS_DIR,img_filename))
        self.image.convert_alpha()
        w, h = self.image.get_size()

        # load image and create weights array
        self.weights_scale = WIDTH/w
        self.weights = pygame.surfarray.array_alpha(self.image)
        self.weights[np.where(self.weights < 255)] = 0
        self.weights = self.weights.astype(np.float32)

        # dilate image to get safety navigation margin
        dil = math.ceil(NAV_MARGIN/self.weights_scale)
        dilated = tu.dilate_image(self.weights, dil, "max")
        self.weights_scaled = tu.resize_image(dilated, WIDTH, HEIGHT) 
        obstacles = np.where(dilated != 0)
        self.weights[obstacles[0], obstacles[1]] = np.inf
        self.weights += 1 # For astar search - lowest value is 1

        self.image = pygame.transform.scale(self.image,(WIDTH,HEIGHT))
        self.rect = self.image.get_rect()
        self.LOS_mask = pygame.surfarray.array_alpha(self.image)
        self.LOS_mask[np.where(self.LOS_mask < 255)] = 0
        self.mask = pygame.mask.from_surface(self.image, 254)
        self.switch = False
        self.CaptureArea = CaptureArea(self)


class CaptureArea(pygame.sprite.Sprite):
    """Capture area sprite for collision detection.
    
    Notes:
        Instance is automatically created via Arena init.  
        Capture area must be specified in .png by color with more than 
        50% transparency (alpha value above 125)!  
    """

    def __init__(self, arena: Arena):
        """
        Parameters:
            arena: Arena class instance.  
        """

        super(CaptureArea, self).__init__()
        self.mask = pygame.mask.from_surface(arena.image, 125)
        self.rect = self.mask.get_rect()


class TankShell(pygame.sprite.Sprite):
    """Tank shell sprite for collision detection."""

    def __init__(self, im_tank_shell: pygame.image, x0: float, y0: float,
                phi: float, v: float):
        """
        Parameters:
            im_tank_shell: Tank shell asset image.
            x0: Initial center coordinate on x axis in pixels.
            y0: Initial center coordinate on y axis in pixels.
            phi: Initial angle in degrees.
            v: Speed of tank shell in pixels per second.  
        """

        super(TankShell, self).__init__()
        self.phi = phi
        self.x = x0
        self.y = y0
        self.v = v
        self.image = pygame.transform.rotate(im_tank_shell, self.phi)
        self.rect = self.image.get_rect(center = (self.x, self.y))

    def update(self, dt: float):
        """Updates position based on time step.
        
        Also updates rectangle and mask attributes for further collision checks.

        Parameters:
            dt: Time step in seconds.
        """

        self.phi = self.phi
        self.x += self.v*math.sin(math.radians(self.phi)) *dt
        self.y += self.v*math.cos(math.radians(self.phi)) *dt
        self.rect = self.image.get_rect(center = (self.x, self.y))
        self.mask = pygame.mask.from_surface(self.image)


class Tank(pygame.sprite.Sprite):
    """Tank object (most important of all)."""

    def __init__(self, pos0_x: float, pos0_y: float, phi0: float, 
                phi_rel0: float, img_body: str, img_turret: str, 
                img_tshell: str):
        """
        Paramters:
            pos0_x: Initial center coordinate on x axis in pixels.
            pos0_y:  Initial center coordinate on y axis in pixels.
            phi0: Initial body angle in degrees.
            phi_rel0: Initial relative angle of canon in degrees.
            img_body: Tank body asset filename.
            img_turret: Tank turret asset filename.
            img_tshell: Tank shell asset filename.
        """

        super(Tank, self).__init__()
        # Load resources
        self.im_body = pygame.image.load(os.path.join(IMGS_DIR,img_body))
        self.im_body.convert()
        self.w, self.h = self.im_body.get_size()
        self.im_turret = pygame.image.load(os.path.join(IMGS_DIR,img_turret))
        self.im_turret.convert()
        self.im_tshell = pygame.image.load(os.path.join(IMGS_DIR,img_tshell))
        self.im_tshell.convert()
        self.canon_len = self.im_turret.get_height()/3
        # Initial positions
        self.x = pos0_x
        self.y = pos0_y
        self.phi = phi0
        self.phi_rel = phi_rel0
        # Properties
        self.FORWARD_SPEED = FORWARD_SPEED
        self.BACKWARD_SPEED = BACKWARD_SPEED
        self.TURN_SPEED = TURN_SPEED
        self.TURRET_TURN_SPEED = TURRET_TURN_SPEED
        self.TSHELL_SPEED = TSHELL_SPEED
        self.GUN_COOLDOWN = GUN_COOLDOWN
        # Controls
        self.phi_in = 0
        self.v_in = 0
        self.phi_rel_in = 0
        self.shoot = 0
        # Pseudo position memory
        self.last_x = self.x
        self.last_y = self.y
        self.last_phi = self.phi_in
        # Memory for draw
        self.mask = None
        self.rect = None
        self.im_body_rot = None
        self.turret_rect = None
        self.im_turret_rot = None
        # Tank shells
        self.tshell_list = []
        self.shoot_delay = self.GUN_COOLDOWN
        # Tank stats
        self.health = MAX_HEALTH
        self.capturing = False
        self.timeCaptured = 0
        self.capturedFlag = False
        self.destroyedFlag = False
        # Tank antennas
        self._ant_num = 3
        self.antennas = np.linspace(
            0, 2*np.pi - (2*np.pi/self._ant_num), self._ant_num)
        self.ant_distances = np.zeros(self._ant_num)
        self.ant_points = np.zeros((self._ant_num, 2))


    def input_manual(self, key: pygame.key, key_event: int):
        """Handles manual inputs (primarly for checking game parameters).

        Parameters:
            key (pygame.event.key): pygame key associated with event 
            key_event (int): 0 = KEYUP, 1 = KEYDOWN
        """
        
        if self.destroyedFlag:
            return
        if key_event == 1:
            if key == pygame.K_LEFT:
                self.phi_in = 1
            elif key == pygame.K_RIGHT:
                self.phi_in = -1
            elif key == pygame.K_UP:
                self.v_in = 1
            elif key == pygame.K_DOWN:
                self.v_in = -1
            elif key == pygame.K_i:
                self.phi_rel_in = 1
            elif key == pygame.K_p:
                self.phi_rel_in = -1
            elif key == pygame.K_o:
                self.shoot = 1
        else:
            if key == pygame.K_LEFT:
                self.phi_in = 0
            elif key == pygame.K_RIGHT:
                self.phi_in = 0
            elif key == pygame.K_UP:
                self.v_in = 0
            elif key == pygame.K_DOWN:
                self.v_in = 0
            elif key == pygame.K_i:
                self.phi_rel_in = 0
            elif key == pygame.K_p:
                self.phi_rel_in = 0
            elif key == pygame.K_o:
                self.shoot = 0

    def inputAI(self, inputs):
        """AI control input method 
        
        inputs -- list in format:
         [0] -- carriage rotation 1/0/-1
         [1] -- carriage forward/backwards movement 1/0/-1
         [2] -- turret rotation 1/0/-1
         [3] -- shoot 1/0
        """

        self.phi_in = round(inputs[0])
        self.v_in = round(inputs[1])
        self.phi_rel_in = round(inputs[2])
        self.shoot = round(inputs[3])

    def moveAssets(self):
        """Rotates and moves assets based on positions.
        
        Also updates rectangle and mask attributes for further collision checks.
        """

        # Base
        self.im_body_rot = pygame.transform.rotate(self.im_body, self.phi)
        self.rect = self.im_body_rot.get_rect(center = (self.x, self.y))
        self.mask = pygame.mask.from_surface(self.im_body_rot)
        # Turret
        self.im_turret_rot = pygame.transform.rotate(self.im_turret, 
                                                    self.phi+self.phi_rel)
        self.turret_rect = self.im_turret_rot.get_rect(center=(self.x, self.y))

    def measureDistances(self, env):
        """Measure distances from sensors to LOS blocking environment 
        
        env -- arena handle
        """
        phi_rad = math.radians(self.phi)
        for i in range(0,self._ant_num):
            self.ant_distances[i],xt ,yt = tu.cast_line(
                self.x, self.y, 
                self.antennas[i] + phi_rad, env.LOS_mask)
            self.ant_points[i] = np.array([xt, yt], ndmin=2)

    def update(self, dt, arena: Arena, mode: int = 0):
        """Update tank related assets based on input and time step.

        Parameters:
            dt: Time step in seconds.
            mode: Optional, defaults to 0.
                0: Basic update for next "frame".
                1: For tankArenaCollision (iterative inverse method).
        """

        # Save last position in case of collision
        self.last_x = self.x
        self.last_y = self.y
        self.last_phi = self.phi
        if mode == 0:
            # Handle shooting
            if self.shoot == 1:
                # Shoot new tank shell
                if self.shoot_delay >= self.GUN_COOLDOWN:
                    s_phi = self.phi + self.phi_rel
                    s_x0 = self.x + self.canon_len \
                           * math.sin(math.radians(s_phi))
                    s_y0 = self.y + self.canon_len \
                           * math.cos(math.radians(s_phi))
                    self.tshell_list.append(TankShell(self.im_tshell, s_x0, \
                                            s_y0, s_phi, self.TSHELL_SPEED))
                    self.shoot_delay = 0
            # Gun cooldown updater
            if self.shoot_delay > self.GUN_COOLDOWN:
                self.shoot_delay = self.GUN_COOLDOWN
            elif self.shoot_delay < self.GUN_COOLDOWN:
                self.shoot_delay += dt
            # Update of fired shells
            for tshell in self.tshell_list:
                tshell.update(dt)
        # Handle body movement
        if not self.v_in < 0:
            speed = self.FORWARD_SPEED
        else:
            speed = self.BACKWARD_SPEED
        turn_speed = self.TURN_SPEED
        # Invert inputs for inverse collision method
        if mode == 1:
            speed = speed*(-1)
            turn_speed = turn_speed*(-1)
        # Edit position attributes based on input
        self.x += speed*(self.v_in)*math.sin(math.radians(self.phi)) *dt
        self.y += speed*(self.v_in)*math.cos(math.radians(self.phi)) *dt
        self.phi += turn_speed*self.phi_in *dt
        if self.phi > 360:
            self.phi = self.phi - 360*(self.phi // 360)
        elif self.phi < 0:
            self.phi = self.phi + 360*(1 - (self.phi // 360))

        if mode == 0:
            self.phi_rel += self.TURRET_TURN_SPEED*self.phi_rel_in *dt
            self.measureDistances(arena)
        self.moveAssets()

    def revertToLast(self, arena:Arena):
        """Revert position attributes to last saved."""

        self.x = self.last_x
        self.y = self.last_y
        self.phi = self.last_phi
        self.measureDistances(arena)
        self.moveAssets()


    def wallCollision(self, map: Arena, dt: float):
        """Advaced handling of collision with walls.

        Parameters:
            map: Arena class object which is also a Sprite.
            dt: Time step in seconds.
        """

        dt_step = dt/MAX_WALL_COLLISION_ITER
        # Check for simultal inputs -> more complex solution required
        if (self.v_in != 0) and (self.phi_in != 0):
            caution = 1
            translate_step = 0.5
        else:
            caution = 0
        # Save collision positions
        x = self.x
        y = self.y
        phi = self.phi
        # And last positions before collision
        last_x = self.last_x
        last_y = self.last_y
        last_phi = self.last_phi

        fix = 0
        # Iterate to non colliding position using inverse steps
        for i in range(MAX_WALL_COLLISION_ITER):
            self.update(dt_step, map, 1)
            col = pygame.sprite.collide_mask(self, map)
            if col is None:
                fix = 1
                break
        # If more complex solution is required
        if caution == 1:
            # Save last solution
            x_fix = self.x
            y_fix = self.y
            phi_fix = self.phi
            # Compute distance from collision position
            fix_dist = math.sqrt((abs(x-self.x))**2 + (abs(y-self.y))**2)
            # Reset colliding positions
            self.x = x
            self.y = y
            self.phi = phi
            # Iterate to non coliding position using alternative approach
            for i in range(MAX_WALL_COLLISION_ITER):
                self.x += translate_step*math.sin(math.radians( \
                                            (self.phi_in*(-90))+(self.phi)))
                self.y += translate_step*math.cos(math.radians( \
                                            (self.phi_in*(-90))+(self.phi)))
                self.moveAssets()
                col = pygame.sprite.collide_mask(self, map)
                if col is None:
                    break 
            fixt_dist2 = math.sqrt((abs(x-self.x))**2 + (abs(y-self.y))**2)
            # Compare solutions based on distance
            if (fix == 1) and (fix_dist < fixt_dist2):
                self.x = x_fix
                self.y = y_fix
                self.phi = phi_fix
                self.moveAssets()
        # Check if collision was solved
        col = pygame.sprite.collide_mask(self, map)
        if col is not None:
            # If not, just move to last pre collision position
            self.last_x = last_x
            self.last_y = last_y
            self.last_phi = last_phi
            self.revertToLast(map)

    def destroy(self):
        """Set destroyedFlag attribute and move tank out of window."""

        self.destroyedFlag = True
        self.x = -300
        self.y = -300
        self.moveAssets()

    def draw(self, game: Game):
        """Draw tank and fired tank shells.
        
        Parameters:
            game: Game class object.
        """

        # Draw body
        game.WIN.blit(self.im_body_rot, self.rect)
        # Draw fired tank shells
        for tshell in self.tshell_list:
            game.WIN.blit(tshell.image, tshell.rect)
        game.WIN.blit(self.im_turret_rot, self.turret_rect)           

        if game.render_antenas:
            if self.destroyedFlag:
                return
            # Draw distance measuring lines
            for i in range(0,self._ant_num):
                xt = self.ant_points[i,0]
                yt = self.ant_points[i,1]
                pygame.draw.line(game.WIN, (255, 0, 0), 
                    (self.x, self.y), (xt, yt), 1)     


def handleCollisions(arena: Arena, game: Game):
    """Handle collisons between all relevant Sprites (objects).

    Parameters:
        arena: Arena class object.
        game: Game class object.
    """
    

    def tankArenaCollision(arena: Arena, tank: Tank, dt: float):
        """Handles collision between Tank and Arena.

        Parameters:
            arena: Arena class object.
            tank: Tank class object.
            dt: Time step in seconds.  
        """

        if tank.destroyedFlag:
            return
        # Check for collision
        col = pygame.sprite.collide_mask(tank, arena)
        if col is not None:
            tank.wallCollision(arena, dt)
            # Prevent wall penetration in tank to tank collision handling
            tank.last_x = tank.x
            tank.last_y = tank.y
            tank.last_phi = tank.phi

    def tankToTankCollison(tank1: Tank, tank2: Tank, arena: Arena):
        """Handles collision between two Tanks.

        Parameters:
            tank1: Tank class object.
            tank2: Tank class object.
            arena: Arena class object.
        """

        if tank1.destroyedFlag or tank2.destroyedFlag:
            return
        # Check for collision
        col = pygame.sprite.collide_mask(tank1, tank2)
        if col is not None:
            tank1.revertToLast(arena)
            tank2.revertToLast(arena)               

    def shellCollisions(arena: Arena, game: Game, tank: Tank,
                        opposing_team: int):
        """Handles collisions between tank shells of the Tank and all Tanks of 
        opposing team.

        Parameters:
            arena: Arena class object.
            game: Game class object.
            tank: Tank class object.
            opposing_team: Selects opposing team (1 or 2).
        """

        if opposing_team == 1:
            opposing_team_list = game.team_1_list
        elif opposing_team == 2:
            opposing_team_list = game.team_2_list
        for opposing_tank in opposing_team_list:
            if opposing_tank.destroyedFlag:
                continue
            # Tank shell colllisions
            shell_list = []
            for shell in tank.tshell_list:
                col = pygame.sprite.collide_mask(shell, arena)
                if col is None:
                    shell_list.append(shell)
            shell_list2 = []
            for shell in shell_list:
                col = pygame.sprite.collide_mask(shell, opposing_tank)
                if col is not None:
                    opposing_tank.health -= 1
                    if opposing_tank.health == 0:
                        opposing_tank.destroy()
                        if opposing_team == 1:
                            game.team_1_alive -= 1
                            print("-- Team 1 tank destroyed.")
                        elif opposing_team == 2:
                            game.team_2_alive -= 1
                            print("-- Team 2 tank destroyed.")
                else:
                    shell_list2.append(shell)
            tank.tshell_list = shell_list2       

    def captureAreaCheck(arena: Arena, tank: Tank, team_num: int, game: Game):
        """Checks if Tank is "touching" capture area and performs relevant
        actions.

        Parameters:
            arena: Arena class object.
            tank: Tank class object.
            team_num: Team number (1 or 2).
            game: Game class object.
        """

        if tank.destroyedFlag:
            return
        # Check touch
        col = pygame.sprite.collide_mask(arena.CaptureArea, tank)
        if col is not None:
            if tank.capturing:
                # Add time in capture area to the team (and Tank)
                tank.timeCaptured += game.dt
                if team_num == 1:
                    game.team_1_capture_time += game.dt
                elif team_num == 2:
                    game.team_2_capture_time += game.dt                    
            else:
                tank.capturing = True
        else:
            tank.capturing = False

    # Main handleCollisions() sequence:
    combinations_list = itertools.combinations(game.master_list,2)
    for tank in game.master_list:
        tankArenaCollision(game.arena, tank, game.dt)
    for comb in combinations_list:
        tankToTankCollison(comb[0], comb[1], game.arena)
    for tank in game.team_1_list:
        shellCollisions(arena, game, tank, 2)
    for tank in game.team_2_list:
        shellCollisions(arena, game, tank, 1)
    for tank1 in game.team_1_list:
        captureAreaCheck(arena, tank1, 1, game)
    for tank2 in game.team_2_list:
        captureAreaCheck(arena, tank2, 2, game)
        

# MAIN LOOP
def main():
    game = Game((WIDTH, HEIGHT), NUM_OF_ROUNDS)
    # init()
    t1 = [(100, 100, 0),
          (200, 100, 0)]
    t2 = [(WIDTH-100, HEIGHT-100, 180)]

    while not game.quit_Flag:
        game.init_round(t1, t2, MAP_FILENAME, True)
        game.render_antenas = True  # Debug
        while game.round_run:
            game.get_delta_time()
            game.check_for_events()
            # -------------------------------------------------------
            # INPUT
            # -------------------------------------------------------
            game.update()
            game.draw()
            game.check_state()
            # -------------------------------------------------------
            #  OUTPUT
            # -------------------------------------------------------
            # print("FPS: ", (1/(game.last_millis/1000)))

if __name__ == "__main__":
    main()