""" BUTanks engine v0.0.4 - dev_DB branch

TODO:
[ ] Tank class (NOT FINISHED)
    [ ] Health
    [ ] Destroy
    [ ] Capture time
    [x] Collisions refine (better than before)
[ ] Capture area
[ ] Time invariance
[=] Comments (PARTIALLY)
[=] Clean (PARTIALLY)
""" 

import os
import math
import pygame
from pathlib import Path
from pygame.locals import *

# CONSTANTS
WIDTH, HEIGHT = 1000, 1000
WHITE = (100,100,100)  # HACK Dark mode 
TARGET_FPS = 60

# Tank settings
FORWARD_SPEED = 150  # 150
BACKWARD_SPEED = 80
TURN_SPEED = 250  # 250
TURRET_TURN_SPEED = 250
TSHELL_SPEED = 150
GUN_COOLDOWN = 1

# WINDOW init
pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# PATHS
p = Path(__file__).parents[1]
ASSETS_DIR = os.path.join(p,"assets")
MAPS_DIR = os.path.join(ASSETS_DIR,"maps")
IMGS_DIR = os.path.join(ASSETS_DIR,"images")


class Game():

    def __init__(self):
        self.who = 0
        
        # Rendering
        self.render_all_frames = True
        self.target_frame = 10
        self.i_frame = 0
        self.dt_fixed = 0.03
        

class Map(pygame.sprite.Sprite):
    """Map class, also known as map."""
    
    def __init__(self, img_filename):
        """
        Args:
            img_filename (string): Image filename.
                Image must have black walls and transparent background.
        """

        super(Map, self).__init__()
        self.x = 0
        self.y = 0
        self.image = pygame.image.load(os.path.join(MAPS_DIR,img_filename))
        self.image.convert_alpha()
        self.image = pygame.transform.scale(self.image,(WIDTH,HEIGHT))
        self.rect = self.image.get_rect()
        self.LOS_mask = pygame.surfarray.array2d(self.image)
        self.mask = pygame.mask.from_surface(self.image)
        self.switch = False


class Tankshell(pygame.sprite.Sprite):
    """Tank shell class."""

    def __init__(self, im_tank_shell, x0, y0, phi, v):
        
        super(Tankshell, self).__init__()
        self.phi = phi
        self.x = x0
        self.y = y0
        self.v = v

        self.image = pygame.transform.rotate(im_tank_shell, self.phi)
        self.rect = self.image.get_rect(center = (self.x, self.y))

    def update(self, dt):

        self.phi = self.phi
        self.x += self.v*math.sin(math.radians(self.phi)) *dt
        self.y += self.v*math.cos(math.radians(self.phi)) *dt
        self.rect = self.image.get_rect(center = (self.x, self.y))
        self.mask = pygame.mask.from_surface(self.image)


class Tank(pygame.sprite.Sprite):
    """Tank class, main class of tank."""

    def __init__(self, pos0_x, pos0_y, phi0, phi_rel0, img_body, img_turret, cntrl_mode):
        """
        Args:
            pos0_x (int or float): Initial coordinate of tank cennter on x axis
            pos0_y (int or float): Initial coordinate of tank cennter on y axis
            phi0 (int or float): Initial angle of tank
            phi_rel0 (int or float): Initial relative angle of tank turret
            img_body (string): Tank body asset filename
            img_turret (string): Tank turret asset filename
            cntrl_mode (int): Control mode: 0 = computer
                                            1 = keybind 1 (for dev)
                                            2 = keybind 2 (for dev)
        """

        super(Tank, self).__init__()
        self.control_mode = cntrl_mode
        # Load resources
        self.im_body = pygame.image.load(os.path.join(IMGS_DIR,img_body))
        self.im_body.convert()
        self.im_turret = pygame.image.load(os.path.join(IMGS_DIR,img_turret))
        self.im_turret.convert()
        self.im_tshell = pygame.image.load(os.path.join(IMGS_DIR,"tank_shell.png"))
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


    def input(self, key, key_event):
        """Handles inputs

        Args:
            key (pygame.event.key): pygame key associated with event 
            key_event (int): 0 = KEYUP, 1 = KEYDOWN
        """
        
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
        else:
            if key_event == 1:
                if key == pygame.K_d:
                    self.phi_in = 1
                elif key == pygame.K_g:
                    self.phi_in = -1
                elif key == pygame.K_r:
                    self.v_in = 1
                elif key == pygame.K_f:
                    self.v_in = -1
                elif key == pygame.K_q:
                    self.phi_rel_in = 1
                elif key == pygame.K_e:
                    self.phi_rel_in = -1
                elif key == pygame.K_w:
                    self.shoot = 1
            else:
                if key == pygame.K_d:
                    self.phi_in = 0
                elif key == pygame.K_g:
                    self.phi_in = 0
                elif key == pygame.K_r:
                    self.v_in = 0
                elif key == pygame.K_f:
                    self.v_in = 0
                elif key == pygame.K_q:
                    self.phi_rel_in = 0
                elif key == pygame.K_e:
                    self.phi_rel_in = 0
                elif key == pygame.K_w:
                    self.shoot = 0


    def rotAssets(self):
        """Rotates assets based on positions and updates class rectangle and mask."""
        # Base
        self.im_body_rot = pygame.transform.rotate(self.im_body, self.phi)
        self.rect = self.im_body_rot.get_rect(center = (self.x, self.y))
        self.mask = pygame.mask.from_surface(self.im_body_rot)
        # Turret
        self.im_turret_rot = pygame.transform.rotate(self.im_turret, self.phi + self.phi_rel)
        self.turret_rect = self.im_turret_rot.get_rect(center = (self.x, self.y))


    def update(self, dt, mode = 0):
        """Update game objects based on input.

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
                    self.tshell_list.append(Tankshell(self.im_tshell, s_x0, s_y0, s_phi, self.TSHELL_SPEED))
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
        self.rotAssets()


    def revertLast(self):
        self.x = self.last_x
        self.y = self.last_y
        self.phi = self.last_phi
        

    def revertWall(self, map: Map, dt):

        MAX_ITER = 40
        self.revertLast()
        dt_step = dt/MAX_ITER
        for i in range(MAX_ITER):
            col = pygame.sprite.collide_mask(self, map)
            if col is not None:
                self.revertLast()
                self.rotAssets()
                break
            self.update(dt_step, 1)

    def wallCollision(self, map: Map, dt):
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
                self.rotAssets()
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

                self.rotAssets()
        # Check if collision was solved
        col = pygame.sprite.collide_mask(self, map)
        if col is not None:
            self.last_x = last_x
            self.last_y = last_y
            self.last_phi = last_phi
            # self.revertWall(map, dt)
            self.revertLast()
            self.rotAssets()

    def draw(self):
        """Draw tank and fired shells."""

        # Draw body
        WIN.blit(self.im_body_rot, self.rect)
        # Draw fired tank shells
        for tshell in self.tshell_list:
            WIN.blit(tshell.image, tshell.rect)
        WIN.blit(self.im_turret_rot, self.turret_rect)           


def handleCollisions(arena:Map, tank1:Tank, tank2:Tank, dt):

    def tankArenaCollision(arena:Map, tank:Tank, dt):
        col = pygame.sprite.collide_mask(tank, arena)
        if col is not None:
            col2 = pygame.sprite.collide_mask(arena, tank)
            print("Col1: ", col, " Col2: ", col2)
            tank.wallCollision(arena, dt)

    def tankToTankCollison(tank1: Tank, tank2: Tank):
        col = pygame.sprite.collide_mask(tank1, tank2)
        if col is not None:
            tank1.revertLast()
            tank2.revertLast()
            tank1.rotAssets()
            tank2.rotAssets()                
            print("Tanks boink together:",col)

    def shellCollisions(arena, att_tank, enemy_tank):
        # Tank shell colllisions
        shell_list = []
        for shell in att_tank.tshell_list:
            col = pygame.sprite.collide_mask(shell, arena)
            if col is None:
                shell_list.append(shell)
        shell_list2 = []
        for shell in shell_list:
            col = pygame.sprite.collide_mask(shell, enemy_tank)
            if col is not None:
                print("bam!")
            else:
                shell_list2.append(shell)
        att_tank.tshell_list = shell_list2

    # Main collision handle
    tankArenaCollision(arena, tank1, dt)
    tankArenaCollision(arena, tank2, dt)
    tankToTankCollison(tank1, tank2)
    shellCollisions(arena, tank1, tank2)
    shellCollisions(arena, tank2, tank1)
        

# MAIN LOOP
def main():
    # Init
    game = Game()
    pygame.display.set_caption("BUTanks engine")
    fpsClock = pygame.time.Clock()
    last_millis = 0
    # Objects
    Tank_1 = Tank(200, 100, 90, 20, "tank1.png", "turret1.png", 1)
    Tank_2 = Tank(400, 100, -90, 10, "tank2.png", "turret2.png", 2)
    arena = Map("map0.png")

    # GAME LOOP
    run = True
    while run:
        # Get delta time
        if (game.i_frame == game.target_frame) or (game.render_all_frames):
            dt = last_millis/1000
        else:
            dt = game.dt_fixed
        # Check for events (including input)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                Tank_1.input(event.key, 1)
                Tank_2.input(event.key, 1)
            if event.type == pygame.KEYUP:
                Tank_1.input(event.key, 0)
                Tank_2.input(event.key, 0)
        # Update objects based on input
        Tank_1.update(dt)
        Tank_2.update(dt)
        # Check and resolve eventual collisions
        handleCollisions(arena, Tank_1, Tank_2, dt)
        # Draw frame based on preferences
        if (game.i_frame == game.target_frame) or (game.render_all_frames):
            WIN.fill(WHITE)
            pygame.sprite.RenderPlain(arena).draw(WIN)
            Tank_1.draw()
            Tank_2.draw()
            pygame.display.update()
            game.i_frame = 0
        else:
            game.i_frame += 1
        # Get time from previous tick and limit FPS
        last_millis = fpsClock.tick(TARGET_FPS)
    pygame.quit()

if __name__ == "__main__":
    main()