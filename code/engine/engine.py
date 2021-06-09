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

CAST_LIM = WIDTH-1
TOLERANCE = 0.001

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
        self._ant_num = 8
        self.antennas = np.linspace(
            0, 2*np.pi - (2*np.pi/self._ant_num), self._ant_num)
        self.ant_distances = np.zeros(self._ant_num)
        
    def update(self, environment):
        self.v += -1*np.sign(self.v)*(self.f)/FPS
        self.x += self.v*math.sin(self.phi)/FPS
        self.y += self.v*math.cos(self.phi)/FPS
        self.rect = pygame.Rect(self.x, self.y, self.w, self.h)
        
        for i in range(0,self._ant_num):
            self.ant_distances[i],xt ,yt = cast_line(
                self.x + self.w/2, self.y + self.h/2, 
                self.antennas[i], environment.LOS_mask)
            pygame.draw.line(WIN, (255, 0, 0), 
                (self.x+self.w/2, self.y+self.h/2), (xt, yt), 1)
            
        #print("Distcs: ", self.ant_distances)
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
        self.LOS_mask = pygame.surfarray.array2d(self.image)
        self.mask = pygame.mask.from_surface(self.image)
  
        #print(self.mask.outline())

    def update(self):
        pass        

def is_close(a,b,tolerance):
    if abs(a-b) < tolerance:
        return True
    else:
        return False        

def cast_line(x0, y0, phi, env):
    x = round(x0)
    y = round(y0) 

    i = 0
    k = np.math.tan(phi)
    
    if abs(k) > 1:
        while (env[round(x),round(y)] == 0):
            if is_close(1/k, 0, 1e-3) is True:
                if phi == np.pi/2:
                    y += 1
                else:
                    y -= 1
            else:
                if (phi > np.pi/2) and (phi < np.pi*3/2):
                    y -= 1
                else:
                    y += 1
            
            x = round(x0 + 1/k*(y-y0))
            i += 1
    else:
        while (env[round(x),round(y)] == 0):
            if is_close(k, 0, 1e-3) is True:
                if phi == 0:
                    x += 1
                else:
                    x -= 1
            else:
                if (phi > 0) and (phi < np.pi):
                    x += 1
                else:
                    x -= 1
            
            y = round(y0 + k*(x-x0))
            i += 1

    dist = np.sqrt((x-x0)**2 + (y-y0)**2)
    return dist, x, y
    
    
def draw_window():
    pygame.display.update()

def main():
    clock = pygame.time.Clock()
    ball = Ball(500, 500, 40, 0, 0, "ball.png")
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