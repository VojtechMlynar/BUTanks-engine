""" BUTanks engine v0.0.1 - dev_DB branch

Main goal: FPS independence and style corrections (not final)
""" 

import os
import math
import time
import pygame
import numpy as np
from pathlib import Path

# CONSTANTS
WIDTH, HEIGHT = 1000, 1000
WHITE = (255,255,255)
TARGET_FPS = 60

# PATHS
p = Path(__file__).parents[1]
ASSETS_DIR = os.path.join(p,"assets")
MAPS_DIR = os.path.join(ASSETS_DIR,"maps")
IMGS_DIR = os.path.join(ASSETS_DIR,"images")


class Ball(pygame.sprite.Sprite):
    """Ball class docstring."""

    def __init__(self, pos_x, pos_y, v0, phi0, f, img_path):
        """Class method docstring."""
        super(Ball, self).__init__()
        self.x = pos_x
        self.y = pos_y
        self.phi = phi0
        self.v = v0
        self.f = f
        self.image = pygame.image.load(os.path.join(IMGS_DIR,img_path))
        self.image.convert()
        self.w, self.h = self.image.get_size()
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)

    def update(self, environment, dt):
        """Class method docstring."""
        self.v += -1*np.sign(self.v)*(self.f) *dt
        self.x += self.v*math.sin(self.phi) *dt
        self.y += self.v*math.cos(self.phi) *dt
        self.rect = pygame.Rect(self.x, self.y, self.w, self.h)

        col= pygame.sprite.collide_mask(self,environment)
        if col is not None:
            self.v = -1*self.v
            print("boink :",col)

class Background(pygame.sprite.Sprite):
    """Ball class docstring."""
    
    def __init__(self, img_path):
        """Class method docstring."""
        super(Background, self).__init__()
        self.x = 0
        self.y = 0
        self.image = pygame.image.load(os.path.join(MAPS_DIR,img_path))
        self.image.convert_alpha()
        # self.w, self.h = self.image.get_size()
        self.image = pygame.transform.scale(self.image,(WIDTH,HEIGHT))
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        """Class method docstring."""
        pass        


def draw_window():
    """Function docstring."""

    #WIN.blit(SPRITE,(300,100))
    pygame.display.update()

# MAIN LOOP
def main():
    prev_time = time.time()

    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BUTanks engine")
    clock = pygame.time.Clock()
    ball = Ball(200, 200, 40, 0.1, 0, "ball.png")
    ballsprite = pygame.sprite.RenderPlain(ball)
    arena = Background("map0.png") #map_plain.png
    arenasprite = pygame.sprite.RenderPlain(arena)

    run = True

    # GAME LOOP
    while run:
        clock.tick(TARGET_FPS)  # Limit framerate
        
        # Compute delta time
        now = time.time()
        dt = now - prev_time
        prev_time = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        WIN.fill(WHITE)
        #ball.update()
              
        ball.update(arena, dt)
        arenasprite.draw(WIN)    
        ballsprite.draw(WIN)
        
        pygame.display.update()     
        #draw_window()

    pygame.quit()

if __name__ == "__main__":
    main()