""" BUTanks engine v0.0.2 - dev_DB branch

TODO:
- Tank class (NOT FINISHED)
    - Tank shells
    - Collision detection between tanks
    - Collision between shell and tank
+ Light methods overhaul (Done?)
""" 

import os
import math
import pygame
import numpy as np
from pathlib import Path
from pygame.locals import *

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

    def __init__(self, pos0_x, pos0_y, phi0, v0, phi_rel0, img_body, img_turret):
        """Class method docstring."""

        super(Tank, self).__init__()
        # Load resources
        self.im_body = pygame.image.load(os.path.join(IMGS_DIR,img_body))
        self.im_body.convert()
        self.im_turret = pygame.image.load(os.path.join(IMGS_DIR,img_turret))
        self.im_turret.convert()
       
        # Initial conditions body
        self.x = pos0_x
        self.y = pos0_y
        self.phi = phi0
        self.v = v0
        # Initial conditions turret
        self.phi_rel = phi_rel0

        # Properties
        self.FORWARD_SPEED = 1
        self.BACKWARD_SPEED = 0.6
        self.TURN_SPEED = 1
        self.TURRET_TURN_SPEED = 1
        self.GUN_COOLDOWN = TARGET_FPS * 1

    def draw(self):

        # Base
        rot_im = pygame.transform.rotate(self.im_body, self.phi)
        im_body_rect = rot_im.get_rect()
        im_body_rect.center = self.im_body.get_rect(top=self.y, left=self.x).center
        WIN.blit(rot_im, im_body_rect)
        self.rect = im_body_rect
        self.mask = pygame.mask.from_surface(rot_im)

        # Turret
        rot_im = pygame.transform.rotate(self.im_turret, self.phi + self.phi_rel)
        im_rect = rot_im.get_rect()
        im_rect.center = im_body_rect.center
        WIN.blit(rot_im, im_rect)

    def update(self, environment, dt):
        """Class method docstring."""

        self.x += self.v*math.sin(math.radians(self.phi)) *dt
        self.y += self.v*math.cos(math.radians(self.phi)) *dt
        self.phi += self.TURN_SPEED
        self.phi_rel += 10
        self.draw()

        col= pygame.sprite.collide_mask(self,environment)
        if col is not None:
            self.v = -1*self.v
            print("boink :",col)


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


# MAIN LOOP
def main():
    pygame.display.set_caption("BUTanks engine")

    fpsClock = pygame.time.Clock()
    last_millis = 0

    # Objects
    Tank_1 = Tank(200, 200, 10, 100, 20, "tank1.png", "turret1.png")
    Tank_2 = Tank(WIDTH-200, HEIGHT-200, 0, 100, 10, "tank2.png", "turret2.png")
    arena = Background("map0.png")

    # GAME LOOP
    run = True
    while run:
        dt = last_millis/1000
        
        # Check for window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Rendering
        WIN.fill(WHITE)
        
        pygame.sprite.RenderPlain(arena).draw(WIN)    
        Tank_1.update(arena, dt)
        Tank_2.update(arena, dt)
        pygame.display.update()

        last_millis = fpsClock.tick(TARGET_FPS)

    pygame.quit()

if __name__ == "__main__":
    main()