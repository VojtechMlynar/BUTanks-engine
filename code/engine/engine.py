""" BUTanks engine v0.0.2 - dev_DB branch

TODO: 
[ ] Tank class (NOT FINISHED)
    [x] Tank shells
        [x] Gun cooldown
    [x] Collision detection between tanks
    [x] Collision between shell and tank
    [x] Tank controls
    [ ] Health
    [ ] Destroy
    [ ] Capture time
    [ ] Push from wall!
[ ] Capture area
[x] Keyboard controls for devs
[x] Light methods overhaul
[x] Non graphic option
[ ] Time invariance
[ ] Comments
[ ] Clean
    [ ] Input to another file
""" 

import os
import time
import math
import pygame
import numpy as np
from pathlib import Path
from pygame.locals import *

# HACK non  graphic
target_i = 1
i = 0
dt_fix = 0.03

# CONSTANTS
WIDTH, HEIGHT = 1000, 1000
# WHITE = (255,255,255)
WHITE = (100,100,100)  # Dark mode
TARGET_FPS = 60

# WINDOW init
pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# PATHS
p = Path(__file__).parents[1]
ASSETS_DIR = os.path.join(p,"assets")
MAPS_DIR = os.path.join(ASSETS_DIR,"maps")
IMGS_DIR = os.path.join(ASSETS_DIR,"images")


class Tank(pygame.sprite.Sprite):
    """Tank class"""

    def __init__(self, pos0_x, pos0_y, phi0, phi_rel0, img_body, img_turret, cntrl_mode):
        """Class method docstring."""

        super(Tank, self).__init__()

        # Control mode: 0 = computer, 1 = keyboard_1, 2 = keyboard_2
        self.control_mode = cntrl_mode

        # Load resources
        self.im_body = pygame.image.load(os.path.join(IMGS_DIR,img_body))
        self.im_body.convert()
        self.im_turret = pygame.image.load(os.path.join(IMGS_DIR,img_turret))
        self.im_turret.convert()
        self.im_tshell = pygame.image.load(os.path.join(IMGS_DIR,"tank_shell.png"))
        self.im_tshell.convert()
        self.canon_len = self.im_turret.get_height()/2
       
        # Initial conditions body
        self.x = pos0_x
        self.y = pos0_y
        self.phi = phi0
        # Initial conditions turret
        self.phi_rel = phi_rel0

        # Properties
        self.FORWARD_SPEED = 60
        self.BACKWARD_SPEED = 40
        self.TURN_SPEED = 10
        self.TURRET_TURN_SPEED = 10
        self.TSHELL_SPEED = 150
        self.GUN_COOLDOWN = 0

        # Controls
        self.phi_in = 0
        self.v_in = 0
        self.phi_rel_in = 0
        self.shoot = 0

        # Pseudo position memory
        self.last_x = self.x
        self.last_y = self.y
        self.last_phi = self.phi

        # Memory for draw
        self.mask = None
        self.rect = None
        self.im_body_rot = None
        self.turret_rect = None
        self.im_turret_rot = None

        # Shell
        self.tshell_list = []
        self.shoot_delay = self.GUN_COOLDOWN

    def input(self, key, key_event):
        
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
        # Base
        self.im_body_rot = pygame.transform.rotate(self.im_body, self.phi)
        self.rect = self.im_body_rot.get_rect(center = (self.x, self.y))
        self.mask = pygame.mask.from_surface(self.im_body_rot)
        # Turret
        self.im_turret_rot = pygame.transform.rotate(self.im_turret, self.phi + self.phi_rel)
        self.turret_rect = self.im_turret_rot.get_rect(center = (self.x, self.y))

    def update(self, dt):
        """Class method docstring."""

        # Save in case of collision
        self.last_x = self.x
        self.last_y = self.y
        self.last_phi = self.phi

        # Handle shooting
        if self.shoot == 1:
            if self.shoot_delay >= self.GUN_COOLDOWN:
                s_phi = self.phi + self.phi_rel
                s_x0 = self.x + self.canon_len * math.sin(math.radians(s_phi))
                s_y0 = self.y + self.canon_len * math.cos(math.radians(s_phi))
                self.tshell_list.append(Tshell(self.im_tshell, s_x0, s_y0, s_phi, self.TSHELL_SPEED))
                self.shoot_delay = 0
        
        if self.shoot_delay > self.GUN_COOLDOWN:
            self.shoot_delay = self.GUN_COOLDOWN
        elif self.shoot_delay < self.GUN_COOLDOWN:
            self.shoot_delay += dt

        if not self.v_in < 0:
            speed = self.FORWARD_SPEED
        else:
            speed = self.BACKWARD_SPEED
        self.x += speed*(self.v_in)*math.sin(math.radians(self.phi)) *dt
        self.y += speed*(self.v_in)*math.cos(math.radians(self.phi)) *dt
        self.phi += self.TURN_SPEED*self.phi_in
        self.phi_rel += self.TURRET_TURN_SPEED*self.phi_rel_in
        self.rotAssets()
        for tshell in self.tshell_list:
            tshell.update(dt)

    def revert(self):
        self.x = self.last_x
        self.y = self.last_y
        self.phi = self.last_phi
        self.rotAssets()

    def draw(self):
        # HACK non  graphic
        if i == target_i:
            WIN.blit(self.im_body_rot, self.rect)
            for tshell in self.tshell_list:
                WIN.blit(tshell.image, tshell.rect)
            WIN.blit(self.im_turret_rot, self.turret_rect)
            

class Tshell(pygame.sprite.Sprite):

    def __init__(self, im_tank_shell, x0, y0, phi, v):
        
        super(Tshell, self).__init__()
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

        

class Background(pygame.sprite.Sprite):
    """Background class docstring."""
    
    def __init__(self, img_path):
        """Class method docstring."""

        super(Background, self).__init__()
        self.x = 0
        self.y = 0
        self.image = pygame.image.load(os.path.join(MAPS_DIR,img_path))
        self.image.convert_alpha()
        self.image = pygame.transform.scale(self.image,(WIDTH,HEIGHT))
        self.rect = self.image.get_rect()
        self.LOS_mask = pygame.surfarray.array2d(self.image)
        self.mask = pygame.mask.from_surface(self.image)


def handleCollisions(arena, tank1, tank2):

    def tankArenaCollision(arena, tank):
        col = pygame.sprite.collide_mask(tank, arena)
        if col is not None:
            print("Boink: ", col)
            # tank.revert()

    def tankToTankCollison(tank1, tank2):
        col = pygame.sprite.collide_mask(tank1, tank2)
        if col is not None:
            tank1.revert()
            tank2.revert()
            print("Tank boink together:",col)

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
    tankArenaCollision(arena, tank1)
    tankArenaCollision(arena, tank2)
    tankToTankCollison(tank1, tank2)
    shellCollisions(arena, tank1, tank2)
    shellCollisions(arena, tank2, tank1)
        


# MAIN LOOP
def main():
    global i
    pygame.display.set_caption("BUTanks engine")

    fpsClock = pygame.time.Clock()
    last_millis = 0

    # Objects
    Tank_1 = Tank(200, 100, 90, 20, "tank1.png", "turret1.png", 1)
    Tank_2 = Tank(400, 100, -90, 10, "tank2.png", "turret2.png", 2)
    arena = Background("map0.png")

    # GAME LOOP
    run = True
    while run:
        start = time.time()  # Loop time start

        # HACK non  graphic
        if target_i == 1:
            dt = last_millis/1000
        else:
            dt = dt_fix
        
        # Check for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                Tank_1.input(event.key, 1)
                Tank_2.input(event.key, 1)
            if event.type == pygame.KEYUP:
                Tank_1.input(event.key, 0)
                Tank_2.input(event.key, 0)

        Tank_1.update(dt)
        Tank_2.update(dt)

        handleCollisions(arena, Tank_1, Tank_2)
        
        # HACK non  graphic
        if i == target_i:
            # Rendering
            WIN.fill(WHITE)
            pygame.sprite.RenderPlain(arena).draw(WIN)
            Tank_1.draw()
            Tank_2.draw()
            pygame.display.update()
            i = 0

        last_millis = fpsClock.tick(TARGET_FPS)

        # print("This loop: ", time.time()-start)  # Print loop time
        i += 1

    pygame.quit()

if __name__ == "__main__":
    main()