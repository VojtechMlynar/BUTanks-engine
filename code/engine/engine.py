# BUTanks engine v0

import os
from numpy.core.fromnumeric import size
import pygame
import numpy as np
import math
from pathlib import Path

WIDTH, HEIGHT = 1000, 1000
WHITE = (255,255,255)
FPS = 60

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("BUTanks engine")

p = Path(__file__).parents[1]
ASSETS_DIR = os.path.join(p,"assets")
MAPS_DIR = os.path.join(ASSETS_DIR,"maps")
IMGS_DIR = os.path.join(ASSETS_DIR,"images")

class Ball(pygame.sprite.Sprite):
    def __init__(self, pos_x, pos_y, v0, phi0, f, img_path):
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

    def update(self, environment):
        self.v += -1*np.sign(self.v)*(self.f)/FPS
        self.x += self.v*math.sin(self.phi)/FPS
        self.y += self.v*math.cos(self.phi)/FPS
        self.rect = pygame.Rect(self.x, self.y, self.w, self.h)

        col= pygame.sprite.collide_mask(self,environment)
        if col is not None:
            self.v = -1*self.v
            print("boink :",col)



class Background(pygame.sprite.Sprite):
    def __init__(self, img_path):
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
        pass        

def draw_window():
    
    #WIN.blit(SPRITE,(300,100))
    pygame.display.update()

def main():
    clock = pygame.time.Clock()
    ball = Ball(200, 200, 40, 0.1, 0, "ball.png")
    ballsprite = pygame.sprite.RenderPlain(ball)
    arena = Background("map0.png") #map_plain.png
    arenasprite = pygame.sprite.RenderPlain(arena)

    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        WIN.fill(WHITE)
        #ball.update()
              
        ball.update(arena)
        arenasprite.draw(WIN)    
        ballsprite.draw(WIN)
        
        pygame.display.update()     
        #draw_window()

    pygame.quit()

if __name__ == "__main__":
    main()