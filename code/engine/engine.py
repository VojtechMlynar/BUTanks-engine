"""BUTanks engine v0.1.2 HOTFIX - dev_DB branch RAI 2021

Build on Python 3.9.1 and Pygame 2.0.1.
Python 3.6+ required.

Info:
- To quit just close the window.
- Pressing 'a' key will toggle "sensor lines" visualization.

- Run this script for check of all parameters (manual controls enabled).

Manual control keybindings:
- Turn body left ---- 'left arrow' key
- Turn body right --- 'right arrow' key
- Move forward ------ 'up arrow' key 
- Move backward ----- 'down arrow' key
- Turn canon left --- 'i' key
- Shoot ------------- 'o' key
- Turn canon right -- 'p' key
"""

import os
import math
import itertools
import numpy as np
import pygame
from pathlib import Path
import tanks_utility as tu

# Game rendering
TARGET_FPS = 60  # Set very high for non graphic learning (e.g. 10000)
RENDER_ALL_FRAMES = True  # Set False to render only target frames or None
TARGET_FRAME = 100  # Target frames to render (multiplies of this num)
FIXED_DT = None  # None for graphic mode

# Convenient EXAMPLES:----------------------------------------------------------
# Real time mode:
# TARGET_FPS = 60  
# RENDER_ALL_FRAMES = True
# FIXED_DT = None
#
# Non graphic mode (suitable for learning):
# TARGET_FPS = 10000
# RENDER_ALL_FRAMES = None
# FIXED_DT = 1/25
#
# Render only each hundredth frame with fixed dt (for validity check by look):
# TARGET_FPS = 10000
# RENDER_ALL_FRAMES = False
# TARGET_FRAME = 100
# FIXED_DT = 1/25
# ------------------------------------------------------------------------------

# Game settings
MAP_BACKGROUND_COLOR = (100, 100, 100)
TARGET_CAPTURE_TIME = 5  # Time that must be spent in the capture area [s]

# Tank settings
FORWARD_SPEED = 150      # [px/s]
BACKWARD_SPEED = 80      # [px/s]
TURN_SPEED = 150         # [deg/s]
TURRET_TURN_SPEED = 150  # [deg/s]
TSHELL_SPEED = 350       # [px/s]
GUN_COOLDOWN = 1         # [s]
MAX_HEALTH = 5
NUM_OF_ANTENNAS = 10

# Collision handling
MAX_WALL_COLLISION_ITER = 20

# Map dilation
NAV_MARGIN = 25

# PATHS to files
p = Path(__file__).parents[1]
ASSETS_DIR = os.path.join(p, "assets")
MAPS_DIR = os.path.join(ASSETS_DIR, "maps")
IMGS_DIR = os.path.join(ASSETS_DIR, "images")

# Only for running this exact py:
WIDTH, HEIGHT = 1000, 1000
MAP_FILENAME = "map1.png"
NUM_OF_ROUNDS = 4


class Game:
    """Game object holds important data for proper game function.

    Attributes:
            num_of_rounds: int
                Number of rounds that will be played.

            team_1_list: list
                List of Team 1 Tank objects.

            team_2_list: list
                List of Team 2 Tank objects.

            team_1_alive: int
                Number of Team 1 members alive.

            team_2_alive: int
                Number of Team 2 members alive.

            dt: float
                Time step between "frames" in seconds.
                Not suitable for FPS check use last_millis!

            win_list: list
                List with results. 0 for tie, otherwise team number.

            round_run_flag: bool
                True if round is in progress.

            quit_flag: bool
                True if quit request is raised by end of all rounds or closing
                of window.

            team_1_captured_time: float
                Team 1 captured time in seconds.

            team_2_captured_time: float
                Team 2 captured time in seconds.

            team_1_captured_flag: bool
                True if Team 1 successfully captured area.

            team_2_captured_flag: bool
                True if Team 2 successfully captured area.

            team_1_destroyed_flag: bool
                True if whole Team 1 was destroyed.

            team_2_destroyed_flag: bool
                True if whole Team 2 was destroyed.

            target_FPS: int
                The Frames Per Second (FPS) cap.
                Set very high for non graphic mode.

            render_all_frames: bool
                Set False for rendering only target frames.
                None for non rendering.

            target_frame: int
                Target frame number.
                Only multiplies will be rendered.

            dt_fixed: None or float
                If not None, sets fixed time step in seconds.
                Usefull for non graphic learning.

            target_capture_time: int
                Target capture time in seconds.

            last_millis: float
                Milliseconds between ticks (one round loop).

            WINDOW: pygame.display
                Pygame window. Important for drawing frames, resizing, etc.

            arena: Arena
                Arena class object (specifies map - walls and capture area).

            render_antennas_flag: bool
                If true antenna lines will be rendered.

            manual_input_flag: bool
                If True all tanks input is linked to keyboard.
                For parameter testing.
    """

    def __init__(self, map_filename: str, window_size: tuple,
                 num_of_rounds: int = 1):
        """
        Parameters:
            num_of_rounds: Number of game rounds. Defaults to 1.
            map_filename: Arena image filename.
            window_size: Size of window as (width,height) in pixels.
        """

        # Declaration of all attributes (most of them changed in init_round)
        self.num_of_rounds = num_of_rounds
        self.team_1_list = []
        self.team_2_list = []
        self.master_list = []
        self.team_1_alive = 0
        self.team_2_alive = 0
        self.dt = 0
        self.win_list = []
        self.round_run_flag = False
        self.quit_flag = False
        self.team_1_captured_time = 0
        self.team_2_captured_time = 0
        self.team_1_captured_flag = False
        self.team_2_captured_flag = False
        self.team_1_destroyed_flag = False
        self.team_2_destroyed_flag = False
        self.target_FPS = TARGET_FPS
        self.render_all_frames = RENDER_ALL_FRAMES
        self.target_frame = TARGET_FRAME
        self.dt_fixed = FIXED_DT
        self.target_capture_time = None
        self.last_millis = 0
        # Window init
        pygame.init()
        self.WINDOW = pygame.display.set_mode((window_size[0], window_size[1]))
        self.arena = Arena(map_filename, window_size)
        # Dev flags
        self.render_antennas_flag = False
        self.manual_input_flag = False
        # Other attributes
        self._fps_clock = None
        self._i_frame = 0
        self._win_team = None
        self._i_round = 1
        self._win_team = None
        pygame.display.set_caption("BUTanks engine")
        print("BUTanks engine initializing... Have fun!\n")

    def init_round(self, team_1_spawn_list: list, team_2_spawn_list: list,
                   target_capture_time: float, tank_scale: float = 1):
        """Initializes game round and sets/resets all attributes needed.

        Parameters:
            team_1_spawn_list: List of tuples with spawn coordinates (x, y, phi).
            team_2_spawn_list: List of tuples with spawn coordinates (x, y, phi).
            target_capture_time: Target capture are time in seconds.
            tank_scale: Scale of all tanks. Defaults to 1 (optional).
        """

        self.team_1_list = []
        self.team_2_list = []
        self._win_team = None
        # Populate team lists
        for item in team_1_spawn_list:
            self.team_1_list.append(Tank(item[0], item[1], item[2], 0,
                                         "tank_1.png", "turret_1.png",
                                         "tank_shell_1.png", tank_scale))
        for item in team_2_spawn_list:
            self.team_2_list.append(Tank(item[0], item[1], item[2], 0,
                                         "tank_2.png", "turret_2.png",
                                         "tank_shell_2.png", tank_scale))
        self.master_list = self.team_1_list + self.team_2_list
        self.team_1_alive = len(team_1_spawn_list)
        self.team_2_alive = len(team_2_spawn_list)
        self.round_run_flag = True
        self.team_1_captured_time = 0
        self.team_2_captured_time = 0
        self.team_1_captured_flag = False
        self.team_2_captured_flag = False
        self.team_1_destroyed_flag = False
        self.team_2_destroyed_flag = False
        self.target_capture_time = target_capture_time
        self.last_millis = 0
        self.render_antennas_flag = False
        self.manual_input_flag = False
        self._i_frame = 0
        self._fps_clock = pygame.time.Clock()
        print(f'Round {self._i_round}',
              f'({self.team_1_alive}v{self.team_2_alive}):')

    def get_delta_time(self):
        """Get time step in seconds."""

        if self.dt_fixed is None:
            self.dt = self.last_millis / 1000
        else:
            self.dt = self.dt_fixed

    def check_and_handle_events(self):
        """Check for pygame events and perform appropriate actions.
        
        Keypress, window closing, etc.
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_flag = True
                self.round_run_flag = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    if self.render_antennas_flag:
                        self.render_antennas_flag = False
                    else:
                        self.render_antennas_flag = True
            if self.manual_input_flag:
                if event.type == pygame.KEYDOWN:
                    for tank in self.master_list:
                        if tank.manual_control_flag:
                            tank.input_manual(event.key, 1)
                if event.type == pygame.KEYUP:
                    for tank in self.master_list:
                        if tank.manual_control_flag:
                            tank.input_manual(event.key, 0)

    def update(self):
        """Update game objects and handle their collisions."""

        for tank in self.master_list:
            tank.update(self.dt, self.arena)
        handle_collisions(self.arena, self)

    def draw_background(self):
        """Draws background (arena) into buffered frame."""

        if (self._i_frame == self.target_frame) or self.render_all_frames:
            self.WINDOW.fill(MAP_BACKGROUND_COLOR)
            pygame.sprite.RenderPlain(self.arena).draw(self.WINDOW)

    def draw_tanks(self):
        """Draws tank related objects into buffered frame."""

        if (self._i_frame == self.target_frame) or self.render_all_frames:
            for tank in self.master_list:
                tank.draw(self)

    def update_frame(self):
        """Draws buffered frame (flip frame)."""

        if (self._i_frame == self.target_frame) or self.render_all_frames:
            pygame.display.update()
            self._i_frame = 0
        else:
            if self.render_all_frames is not None:
                self._i_frame += 1

    def check_state(self):
        """Check game conditions and eventually switch game states."""

        # Check if team had already captured area
        if self.team_1_captured_time >= self.target_capture_time:
            self.team_1_captured_flag = True
            print("Team 1 captured area!")
        if self.team_2_captured_time >= self.target_capture_time:
            self.team_2_captured_flag = True
            print("Team 2 captured area!")
        # Check if whole team is destroyed
        if self.team_1_alive == 0:
            self.team_1_destroyed_flag = True
            print("- Team 1 destroyed!")
        if self.team_2_alive == 0:
            self.team_2_destroyed_flag = True
            print("- Team 2 destroyed!")
        # Check winner based on successfully captured area
        if self.team_1_captured_flag and self.team_2_captured_flag:
            self._win_team = 0  # Tie
        elif self.team_1_captured_flag:
            self._win_team = 1
        elif self.team_2_captured_flag:
            self._win_team = 2
        # Check winner based on destroying whole opposite team
        if self.team_1_destroyed_flag and self.team_2_destroyed_flag:
            self._win_team = 0  # Tie
        elif self.team_1_destroyed_flag:
            self._win_team = 2
        elif self.team_2_destroyed_flag:
            self._win_team = 1
        # Check if any team just won
        if self._win_team is not None:
            self.win_list.append(self._win_team)
            # Write appropriate string to terminal
            if self._win_team == 0:
                win_str = "Round ended in a tie!\n"
            else:
                win_str = f'Team {self._win_team} won!\n'
            print(win_str)
            # Raise flags
            self.round_run_flag = False
            if self._i_round == self.num_of_rounds:
                self.quit_flag = True
            else:
                self._i_round += 1
        if self.quit_flag:
            pygame.quit()
            print("Final win list (0 = Tie): ", self.win_list)
            print("Quitting.")
            return
        self.last_millis = self._fps_clock.tick(self.target_FPS)


class Arena(pygame.sprite.Sprite):
    """Arena sprite with CaptureArea attribute.
    
    Important attribute:
        CaptureArea: CaptureArea object for capture area detection.

    Other attributes:
        image -- map file image

        res_scale -- tuple of how much was each axis scaled from original image
            note: Use low resolution underlying images to save resources

        alpha_arr -- numpy array of converted alpha values of the image file
            notes:-- opaque pixels (255) are considered walls
                  -- transparent pixels (0) are considered non blocking for
                     movement and line of sight (LOS)
                  -- pixels > 125, but < 255 are considered capture area,
                     dont block movement and LOS

        LOS_mask -- numpy array of window-scaled blocking terrain
                 -- 255 is blocking terrain, everything equal or less than 254
                    is set to 0

        example: You have set size parameter to (1000,1000) when creating game
            and have map image with resolution of (100,100). Hence, res_scale 
            will be (10,10), alpha_arr will have dimensions (100,100) and 
            LOS_mask (1000,1000)
    """

    def __init__(self, img_filename: str, size: tuple):
        """
        Parameters:
            img_filename: Arena image filename.
            size: Window size as (width, height) in pixels.

        Notes:
            Image must have non transparent walls and transparent background.    
        """

        super(Arena, self).__init__()
        self.x = 0
        self.y = 0
        self.image = pygame.image.load(os.path.join(MAPS_DIR, img_filename))
        self.image.convert_alpha()
        w, h = self.image.get_size()
        # Load image and create alpha array
        self.res_scale = (size[0]/w, size[1]/h)
        self.alpha_arr = pygame.surfarray.array_alpha(self.image)

        # Main procedures
        self.image = pygame.transform.scale(self.image, (size[0], size[1]))
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
    """Tank shell sprite for collision detection.
    
    Attributes:
        x: float
            Coordinate of center on x axis.
        
        y: float
            Coordinate of center on y axis.
        
        phi: float
            Angle in degrees (zero on positive y axis direction).

        v: float
            Speed in pixels per second.
    """

    def __init__(self, im_tank_shell: pygame.image, x0: float, y0: float,
                 phi: float, v: float):
        """
        Parameters:
            im_tank_shell: Tank shell asset image.
            x0: Initial center coordinate on x axis in pixels.
            y0: Initial center coordinate on y axis in pixels.
            phi: Initial angle in degrees (zero on positive y axis direction).
            v: Speed of tank shell in pixels per second.  
        """

        super(TankShell, self).__init__()
        self.phi = phi
        self.x = x0
        self.y = y0
        self.v = v
        self.image = pygame.transform.rotate(im_tank_shell, self.phi)
        self.rect = self.image.get_rect(center=(self.x, self.y))
        self.mask = pygame.mask.from_surface(self.image)

    def update(self, dt: float):
        """Updates position based on time step.
        
        Also updates rectangle and mask attributes for further collision checks.

        Parameters:
            dt: Time step in seconds.
        """

        self.phi = self.phi
        self.x += self.v * math.sin(math.radians(self.phi)) * dt
        self.y += self.v * math.cos(math.radians(self.phi)) * dt
        self.rect = self.image.get_rect(center=(self.x, self.y))
        self.mask = pygame.mask.from_surface(self.image)


class Tank(pygame.sprite.Sprite):
    """Tank object.
    
    Attributes:
        x: float
            Coordinate of center on x axis.
        
        y: float
            Coordinate of center on y axis.
        
        phi: float
            Angle of body in degrees (zero on positive y axis direction).

        phi_rel: float
            Relative angle of canon in degrees.

        health: int
            Tank health (number of shots that cloud be taken to destroy).

        captured_time: float
            Captured time in seconds.

        capturing_flag: bool
            True while capturing.
        
        captured_flag: bool
            True if successfully captured area.

        destroyed_flag: bool
            True if destroyed.

        manual_control_flag: bool
            True for manual control (debug).
    """

    def __init__(self, pos0_x: float, pos0_y: float, phi0: float,
                 phi_rel0: float, img_body: str, img_turret: str,
                 img_tshell: str, scale: float = 1):
        """
        Parameters:
            pos0_x: Initial center coordinate on x axis in pixels.
            pos0_y:  Initial center coordinate on y axis in pixels.
            phi0: Initial body angle in degrees 
            (zero on positive y axis direction).
            phi_rel0: Initial relative angle of canon in degrees.
            img_body: Tank body asset filename.
            img_turret: Tank turret asset filename.
            img_tshell: Tank shell asset filename.
        """

        super(Tank, self).__init__()
        self.scale = scale
        # Load resources
        self.im_body = pygame.image.load(os.path.join(IMGS_DIR, img_body))
        self.im_turret = pygame.image.load(os.path.join(IMGS_DIR, img_turret))
        self.im_tshell = pygame.image.load(os.path.join(IMGS_DIR, img_tshell))
        if self.scale != 1:
            w, h = self.im_body.get_size()
            self.im_body = pygame.transform.scale(self.im_body,
                                                  (int(w * self.scale),
                                                   int(h * self.scale)))
            w, h = self.im_turret.get_size()
            self.im_turret = pygame.transform.scale(self.im_turret,
                                                    (int(w * self.scale),
                                                     int(h * self.scale)))
            w, h = self.im_tshell.get_size()
            self.im_tshell = pygame.transform.scale(self.im_tshell,
                                                    (int(w * self.scale),
                                                     int(h * self.scale)))
        self.w, self.h = self.im_body.get_size()
        self.im_body.convert()
        self.im_turret.convert()
        self.im_tshell.convert()
        self.canon_len = self.im_turret.get_height() / 3
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
        if self.scale != 1:
            self.FORWARD_SPEED = FORWARD_SPEED * self.scale
            self.BACKWARD_SPEED = BACKWARD_SPEED * self.scale
            self.TSHELL_SPEED = TSHELL_SPEED * self.scale
        # Controls
        self.phi_in = 0
        self.v_in = 0
        self.phi_rel_in = 0
        self.shoot = 0
        self.manual_control_flag = False
        # Pseudo position memory
        self.last_x = self.x
        self.last_y = self.y
        self.last_phi = self.phi_in
        # Memory for draw
        self.im_body_rot = None
        self.turret_rect = None
        self.im_turret_rot = None
        # For draw and collisions
        self.mask = None
        self.rect = None
        # Tank shells
        self.tshell_list = []
        self.shoot_delay = self.GUN_COOLDOWN
        # Tank stats
        self.health = MAX_HEALTH
        self.captured_time = 0
        self.capturing_flag = False
        self.captured_flag = False
        self.destroyed_flag = False
        # Tank antennas
        self.ant_num = NUM_OF_ANTENNAS
        self.antennas = np.linspace(
            0, 2 * np.pi - (2 * np.pi / self.ant_num), self.ant_num)
        self.antennas += np.pi / 2
        self.ant_distances = np.zeros(self.ant_num)
        self.ant_points = np.zeros((self.ant_num, 2))

    def input_manual(self, key: pygame.key, key_event: int):
        """Handles manual inputs (primarily for checking game parameters).

        Parameters:
            key (pygame.event.key): pygame key associated with event 
            key_event (int): 0 = KEYUP, 1 = KEYDOWN
        """

        if self.destroyed_flag:
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

    def input_AI(self, inputs: list):
        """AI control input method 
        
        Parameters:
            inputs: List of integers in format:
                    [0]: body rotation 1/0/-1
                    [1]: body forward/backwards movement 1/0/-1
                    [2]: turret rotation 1/0/-1
                    [3]: shoot 1/0
        """

        self.phi_in = round(inputs[0])
        self.v_in = round(inputs[1])
        self.phi_rel_in = round(inputs[2])
        self.shoot = round(inputs[3])

    def move_assets(self):
        """Rotates and moves assets based on positions.
        
        Also updates rectangle and mask attributes for further collision checks.
        """

        # Base
        self.im_body_rot = pygame.transform.rotate(self.im_body, self.phi)
        self.rect = self.im_body_rot.get_rect(center=(self.x, self.y))
        self.mask = pygame.mask.from_surface(self.im_body_rot)
        # Turret
        self.im_turret_rot = pygame.transform.rotate(self.im_turret,
                                                     self.phi + self.phi_rel)
        self.turret_rect = self.im_turret_rot.get_rect(center=(self.x, self.y))

    def measure_distances(self, arena: Arena):
        """Measure distances from sensors to LOS blocking environment 
        
        Parameters:
            arena: Arena class object.
        """
        if self.phi > 360:
            self.phi = self.phi - 360 * (self.phi // 360)
        elif self.phi < 0:
            self.phi = self.phi - 360 * (self.phi // 360)
        phi_rad = math.radians(self.phi)
        for i in range(0, self.ant_num):
            self.ant_distances[i], xt, yt = tu.cast_line(
                self.x, self.y,
                self.antennas[i] - phi_rad, arena.LOS_mask)
            self.ant_points[i] = np.array([xt, yt], ndmin=2)

    def update(self, dt, arena: Arena, mode: int = 0):
        """Update tank related assets based on input and time step.

        Parameters:
            dt: Time step in seconds.
            arena: Arena class object.
            mode: Optional, defaults to 0.
                0: Basic update for next "frame".
                1: For tank_arena_collision (iterative inverse method).
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
                    s_x0 = self.x + self.canon_len * math.sin(math.radians(s_phi))
                    s_y0 = self.y + self.canon_len * math.cos(math.radians(s_phi))
                    self.tshell_list.append(TankShell(self.im_tshell, s_x0,
                                                      s_y0, s_phi,
                                                      self.TSHELL_SPEED))
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
            speed = speed * (-1)
            turn_speed = turn_speed * (-1)
        # Edit position attributes based on input
        self.x += speed * self.v_in * math.sin(math.radians(self.phi)) * dt
        self.y += speed * self.v_in * math.cos(math.radians(self.phi)) * dt
        self.phi += turn_speed * self.phi_in * dt
        if mode == 0:
            self.phi_rel += self.TURRET_TURN_SPEED * self.phi_rel_in * dt
        self.move_assets()

    def revert_to_last(self):
        """Revert position attributes to last saved."""

        self.x = self.last_x
        self.y = self.last_y
        self.phi = self.last_phi
        self.move_assets()

    def wall_collision(self, arena: Arena, dt: float):
        """Advanced handling of collision with walls.

        Parameters:
            arena: Arena class object which is also a Sprite.
            dt: Time step in seconds.
        """

        dt_step = dt / MAX_WALL_COLLISION_ITER
        translate_step = 0.5
        # Check for simultaneous inputs -> more complex solution required
        if (self.v_in != 0) and (self.phi_in != 0):
            caution = 1
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
            self.update(dt_step, arena, 1)
            col = pygame.sprite.collide_mask(self, arena)
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
            fix_dist = math.sqrt((abs(x - self.x)) ** 2 + (abs(y - self.y)) ** 2)
            # Reset colliding positions
            self.x = x
            self.y = y
            self.phi = phi
            # Iterate to non colliding position using alternative approach
            for i in range(MAX_WALL_COLLISION_ITER):
                self.x += translate_step * math.sin(math.radians(
                          (self.phi_in * (-90)) + self.phi))
                self.y += translate_step * math.cos(math.radians(
                    (self.phi_in * (-90)) + self.phi))
                self.move_assets()
                col = pygame.sprite.collide_mask(self, arena)
                if col is None:
                    break
            fixt_dist2 = math.sqrt((abs(x - self.x)) ** 2 + (abs(y - self.y)) ** 2)
            # Compare solutions based on distance
            if (fix == 1) and (fix_dist < fixt_dist2):
                self.x = x_fix
                self.y = y_fix
                self.phi = phi_fix
                self.move_assets()
        # Check if collision was solved
        col = pygame.sprite.collide_mask(self, arena)
        if col is not None:
            # If not, just move to last pre collision position
            self.last_x = last_x
            self.last_y = last_y
            self.last_phi = last_phi
            self.revert_to_last()

    def destroy(self):
        """Set destroyed_flag attribute and move tank out of render window."""

        self.destroyed_flag = True
        self.x = -300
        self.y = -300
        self.move_assets()

    def draw(self, game: Game):
        """Draw tank and fired tank shells.
        
        Parameters:
            game: Game class object.
        """

        # Draw body
        game.WINDOW.blit(self.im_body_rot, self.rect)
        # Draw fired tank shells
        for tshell in self.tshell_list:
            game.WINDOW.blit(tshell.image, tshell.rect)
        game.WINDOW.blit(self.im_turret_rot, self.turret_rect)

        if game.render_antennas_flag:
            if self.destroyed_flag:
                return
            # Draw distance measuring lines
            for i in range(0, self.ant_num):
                xt = self.ant_points[i, 0]
                yt = self.ant_points[i, 1]
                pygame.draw.line(game.WINDOW, (255, 0, 0),
                                 (self.x, self.y), (xt, yt), 1)


def handle_collisions(arena: Arena, game: Game):
    """Handle collisions between all relevant Sprites (objects).

    Parameters:
        arena: Arena class object.
        game: Game class object.
    """

    def tank_arena_collision(arena: Arena, tank: Tank, dt: float):
        """Handles collision between Tank and Arena.

        Parameters:
            arena: Arena class object.
            tank: Tank class object.
            dt: Time step in seconds.  
        """

        if tank.destroyed_flag:
            return
        # Check for collision
        col = pygame.sprite.collide_mask(tank, arena)
        if col is not None:
            tank.wall_collision(arena, dt)
            # Prevent wall penetration in tank to tank collision handling
            tank.last_x = tank.x
            tank.last_y = tank.y
            tank.last_phi = tank.phi

    def tank_tank_collision(tank1: Tank, tank2: Tank, arena: Arena):
        """Handles collision between two Tanks.

        Parameters:
            tank1: Tank class object.
            tank2: Tank class object.
            arena: Arena class object.
        """

        if tank1.destroyed_flag or tank2.destroyed_flag:
            return
        # Check for collision
        col = pygame.sprite.collide_mask(tank1, tank2)
        if col is not None:
            tank1.revert_to_last()
            tank2.revert_to_last()

    def tshell_collisions(arena: Arena, game: Game, tank: Tank,
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
        else:
            opposing_team_list = None
        for opposing_tank in opposing_team_list:
            if opposing_tank.destroyed_flag:
                continue
            # Tank shell collisions
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

    def capture_area_check(arena: Arena, tank: Tank, team_num: int, game: Game):
        """Checks if Tank is "touching" capture area and performs relevant
        actions.

        Parameters:
            arena: Arena class object.
            tank: Tank class object.
            team_num: Team number (1 or 2).
            game: Game class object.
        """

        if tank.destroyed_flag:
            return
        # Check touch
        col = pygame.sprite.collide_mask(arena.CaptureArea, tank)
        if col is not None:
            if tank.capturing_flag:
                # Add time in capture area to the team (and Tank)
                tank.captured_time += game.dt
                if team_num == 1:
                    game.team_1_captured_time += game.dt
                elif team_num == 2:
                    game.team_2_captured_time += game.dt
            else:
                tank.capturing_flag = True
        else:
            tank.capturing_flag = False

    # Main handleCollisions() sequence:
    combinations_list = itertools.combinations(game.master_list, 2)
    for tank in game.master_list:
        tank_arena_collision(game.arena, tank, game.dt)
    for comb in combinations_list:
        tank_tank_collision(comb[0], comb[1], game.arena)
    for tank in game.team_1_list:
        tshell_collisions(arena, game, tank, 2)
    for tank in game.team_2_list:
        tshell_collisions(arena, game, tank, 1)
    for tank1 in game.team_1_list:
        capture_area_check(arena, tank1, 1, game)
    for tank2 in game.team_2_list:
        capture_area_check(arena, tank2, 2, game)
    for tank in game.master_list:
        # Update antennas
        tank.measure_distances(arena)


# MAIN LOOP
def main():
    game = Game(MAP_FILENAME, (WIDTH, HEIGHT), NUM_OF_ROUNDS)

    # init()
    t1 = [(100, 100, 0)]
    t2 = [(WIDTH - 100, HEIGHT - 100, 180)]

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
            # Place to draw on top of tanks
            game.update_frame()
            game.check_state()
            # -------------------------------------------------------
            #  OUTPUT
            # -------------------------------------------------------
            # print("FPS: ", (1/(game.last_millis/1000)))


if __name__ == "__main__":
    main()
