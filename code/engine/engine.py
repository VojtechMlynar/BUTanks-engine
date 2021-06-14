""" BUTanks engine v0.0.7 - dev_DB branch

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
import pygame
from pathlib import Path

# MAP
MAP_FILENAME = "map1.png"
WIDTH, HEIGHT = 1000, 1000
WHITE = (100,100,100)  # HACK Dark mode 

# Game settings
TEAM_1 = 1                # Number of Team 1 tanks
TEAM_2 = 1                # Number of Team 2 tanks
TARGET_CAPTURE_TIME = 5   # Time that must be spent in capture area [s]

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

# WINDOW init
pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# PATHS
p = Path(__file__).parents[1]
ASSETS_DIR = os.path.join(p,"assets")
MAPS_DIR = os.path.join(ASSETS_DIR,"maps")
IMGS_DIR = os.path.join(ASSETS_DIR,"images")

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

    def __init__(self):
        """Initializes Game class object with following attributes."""

        # Settings
        self.target_capture_time = 5
        self.team_1_alive = TEAM_1
        self.team_2_alive = TEAM_2
                
        # Stats
        self.dt = 0
        self.team_1_capture_time = 0
        self.team_2_capture_time = 0

        # State related
        self.team_1_captured_flag = False
        self.team_2_captured_flag = False
        
        # Rendering
        self.targetFPS = TARGET_FPS
        self.render_all_frames = RENDER_ALL_FRAMES
        self.target_frame = TARGET_FRAME
        self.i_frame = 0
        self.dt_fixed = FIXED_DT

        # Team lists
        self.team_1_list = []
        self.team_2_list = []
        self.master_list = None
    
    def checkState(self):
        """Check game conditions and eventually switch game state."""
        
        # Check if team had already captured area
        if self.team_1_capture_time >= self.target_capture_time:
            self.team_1_captured_flag = True
            print("TEAM 1 JUST CAPTURED AREA!")
        if self.team_2_capture_time >= self.target_capture_time:
            self.team_2_captured_flag = True
            print("TEAM 2 JUST CAPTURED AREA!")
        # Check if whole team is destroyded
        if self.team_1_alive == 0:
            print("TEAM 1 JUST DIED!")
        if self.team_2_alive == 0:
            print("TEAM 2 JUST DIED!")


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
        self.image = pygame.transform.scale(self.image,(WIDTH,HEIGHT))
        self.rect = self.image.get_rect()
        self.LOS_mask = pygame.surfarray.array2d(self.image)
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

#  TODO contorl_mode comment and add relative attribute comments
class Tank(pygame.sprite.Sprite):
    """Tank object (most important of all)."""

    def __init__(self, pos0_x: float, pos0_y: float, phi0: float, 
                phi_rel0: float, img_body: str, img_turret: str, 
                img_tshell: str, cntrl_mode: int):
        """
        Paramters:
            pos0_x: Initial center coordinate on x axis in pixels.
            pos0_y:  Initial center coordinate on y axis in pixels.
            phi0: Initial body angle in degrees.
            phi_rel0: Initial relative angle of canon in degrees.
            img_body: Tank body asset filename.
            img_turret: Tank turret asset filename.
            img_tshell: Tank shell asset filename.
            cntrl_mode: [description]
        """

        super(Tank, self).__init__()
        self.control_mode = cntrl_mode
        # Load resources
        self.im_body = pygame.image.load(os.path.join(IMGS_DIR,img_body))
        self.im_body.convert()
        self.im_turret = pygame.image.load(os.path.join(IMGS_DIR,img_turret))
        self.im_turret.convert()
        self.im_tshell = pygame.image.load(os.path.join(IMGS_DIR,img_tshell))
        self.im_tshell.convert()
        self.canon_len = self.im_turret.get_height()/2
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
        
    # TODO: NEED CHANGE!
    def input(self, key: pygame.key, key_event: int):
        """Handles inputs

        Parameters:
            key (pygame.event.key): pygame key associated with event 
            key_event (int): 0 = KEYUP, 1 = KEYDOWN
        """
        
        if self.destroyedFlag:
            return

        if self.control_mode == 0:
            a = 0
        elif self.control_mode == 1:
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


    def update(self, dt: float, mode: int = 0):
        """Update tank related objects based on input.

        Args:
            dt (float): Delta time from last "frame"
            mode (int, optional): Defaults to 0.
                0 = basic update for next "frame",
                1 = for revertWall collision handling (iterative method),
                2 = for tankArenaCollision (iterative method inverse)
        """

        # Save last position in case of collision
        self.last_x = self.x
        self.last_y = self.y
        self.last_phi = self.phi
        if mode == 0:
            # Handle shooting
            if self.shoot == 1:
                if self.shoot_delay >= self.GUN_COOLDOWN:
                    s_phi = self.phi + self.phi_rel
                    s_x0 = self.x + self.canon_len * math.sin(math.radians(s_phi))
                    s_y0 = self.y + self.canon_len * math.cos(math.radians(s_phi))
                    self.tshell_list.append(TankShell(self.im_tshell, s_x0, s_y0, s_phi, self.TSHELL_SPEED))
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

        if mode == 2:
            speed = speed*(-1)
            turn_speed = turn_speed*(-1)

        self.x += speed*(self.v_in)*math.sin(math.radians(self.phi)) *dt
        self.y += speed*(self.v_in)*math.cos(math.radians(self.phi)) *dt
        self.phi += turn_speed*self.phi_in *dt
        if mode == 0:
            self.phi_rel += self.TURRET_TURN_SPEED*self.phi_rel_in *dt
        self.moveAssets()


    def revertToLast(self):
        self.x = self.last_x
        self.y = self.last_y
        self.phi = self.last_phi
        

    def revertWall(self, map: Arena, dt):

        MAX_ITER = 40
        self.revertToLast()
        dt_step = dt/MAX_ITER
        for i in range(MAX_ITER):
            col = pygame.sprite.collide_mask(self, map)
            if col is not None:
                self.revertToLast()
                self.moveAssets()
                break
            self.update(dt_step, 1)

    def wallCollision(self, map: Arena, dt):
        """Advaced handling of collision with wall

        Args:
            map (Map): Map class object
            dt (float): delta time for current "frame"
        """
        MAX_ITER = 20
        dt_step = dt/MAX_ITER

        # Check required movement:
        # Check simultal inputs -> More complex solution required
        if (self.v_in != 0) and (self.phi_in != 0):
            caution = 1
            translate_step = 0.5
        else:
            caution = 0
        # Save collision positions
        x = self.x
        y = self.y
        phi = self.phi

        last_x = self.last_x
        last_y = self.last_y
        last_phi = self.last_phi

        fix = 0
        # Iterate to non colliding position
        for i in range(MAX_ITER):
            self.update(dt_step, 2)
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
            fix_dist = math.sqrt((abs(x-self.x))**2 + (abs(y-self.y))**2)
            # Reset colliding positions
            self.x = x
            self.y = y
            self.phi = phi
            # Iterate to non coliding position using alternative approach
            for i in range(MAX_ITER):
                self.x += (translate_step)*math.sin(math.radians((self.phi_in*(-90))+(self.phi)))
                self.y += (translate_step)*math.cos(math.radians((self.phi_in*(-90))+(self.phi)))
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
                # if abs(phi-phi_fix) > 

                self.moveAssets()
        # Check if collision was solved
        col = pygame.sprite.collide_mask(self, map)
        if col is not None:
            self.last_x = last_x
            self.last_y = last_y
            self.last_phi = last_phi
            # self.revertWall(map, dt)
            self.revertToLast()
            self.moveAssets()

    def destroy(self):
        self.x = -100
        self.y = -100
        self.moveAssets()

    def draw(self):
        """Draw tank and fired shells."""

        # Draw body
        WIN.blit(self.im_body_rot, self.rect)
        # Draw fired tank shells
        for tshell in self.tshell_list:
            WIN.blit(tshell.image, tshell.rect)
        WIN.blit(self.im_turret_rot, self.turret_rect)           


def handleCollisions(arena: Arena, game: Game):

    def tankArenaCollision(arena: Arena, tank: Tank, dt):
        if tank.destroyedFlag:
            return
        col = pygame.sprite.collide_mask(tank, arena)
        if col is not None:
            tank.wallCollision(arena, dt)

    def tankToTankCollison(tank1: Tank, tank2: Tank):
        if tank1.destroyedFlag or tank2.destroyedFlag:
            return
        col = pygame.sprite.collide_mask(tank1, tank2)
        if col is not None:
            tank1.revertToLast()
            tank2.revertToLast()
            tank1.moveAssets()
            tank2.moveAssets()                
            print("Tanks boink together:",col)

    def shellCollisions(arena: Arena, game: Game, tank: Tank, opposing_team):

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
                        opposing_tank.destroyedFlag = True
                        opposing_tank.destroy()
                        if opposing_team == 1:
                            game.team_1_alive -= 1
                            print("Team 1 tank destroyed")
                        elif opposing_team == 2:
                            game.team_2_alive -= 2
                            print("Team 2 tank destroyed")
                else:
                    shell_list2.append(shell)
            tank.tshell_list = shell_list2       

    def captureAreaCheck(arena: Arena, tank: Tank, team, game: Game):
        if tank.destroyedFlag:
            return
        col = pygame.sprite.collide_mask(arena.CaptureArea, tank)
        if col is not None:
            if tank.capturing:
                tank.timeCaptured += game.dt
                if team == 1:
                    game.team_1_capture_time += game.dt
                elif team == 2:
                    game.team_2_capture_time += game.dt                    
            else:
                tank.capturing = True
        else:
            tank.capturing = False

    combinations_list = itertools.combinations(game.master_list,2)
    
    for tank in game.master_list:
        tankArenaCollision(arena, tank, game.dt)
    for comb in combinations_list:
        tankToTankCollison(comb[0], comb[1])
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
    # Init
    game = Game()
    pygame.display.set_caption("BUTanks engine")
    fpsClock = pygame.time.Clock()
    last_millis = 0
    # Objects
    for i in range(game.team_1_alive):
        game.team_1_list.append(Tank(200+(200*i), 100, 90, 0, "tank1.png", "turret1.png", "tank_shell.png",  1))
    for i in range(game.team_2_alive):
        game.team_2_list.append(Tank(WIDTH-200-(200*i), HEIGHT-100, 180, 0, "tank2.png", "turret2.png", "tank_shell.png", 0))

    game.master_list = game.team_1_list + game.team_2_list
    
    arena = Arena(MAP_FILENAME)

    # GAME LOOP
    run = True
    while run:
        # Get delta time
        if game.dt_fixed is None:
            game.dt = last_millis/1000
        else:
            game.dt = game.dt_fixed
        # Check for events (including input)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                for tank in game.master_list:
                    tank.input(event.key, 1)
            if event.type == pygame.KEYUP:
                for tank in game.master_list:
                    tank.input(event.key, 0)
        # Update objects based on input
        for tank in game.master_list:
            tank.update(game.dt)
        # Check and resolve eventual collisions
        handleCollisions(arena, game)
        # Draw frame based on preferences
        if (game.i_frame == game.target_frame) or (game.render_all_frames):
            WIN.fill(WHITE)
            pygame.sprite.RenderPlain(arena).draw(WIN)
            for tank in game.master_list:
                tank.draw()
            pygame.display.update()
            game.i_frame = 0
        else:
            game.i_frame += 1
        game.checkState()
        # Get time from previous tick and limit FPS
        last_millis = fpsClock.tick(game.targetFPS)
        # print("FPS: ", 1/(last_millis/1000))
    pygame.quit()

if __name__ == "__main__":
    main()