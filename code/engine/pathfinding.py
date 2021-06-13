import os
from numpy.core.fromnumeric import size
import pygame
import numpy as np
import math
from pathlib import Path
import pyastar2d
import tanks_utility as tu


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
NAV_MARGIN = 25 # Pixels distance used for navigation

class Ball(pygame.sprite.Sprite):
    """Simple ball Class - used for testing only!"""
    def __init__(self, pos_x, pos_y, v0, phi0, f, img_path):
        """Init ball class 
        
        pos_x -- initial X position
        pos_y -- initial Y position
        v0 -- initial velocity
        phi0 -- initial heading
        f -- damping "force"
        img_path -- path to used image (ball-like preferably)
        """
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
        self.path = np.array([pos_x, pos_y], ndmin=2)
        # Distance measuring antennas - how many, which angles
        self._ant_num = 8
        self.antennas = np.linspace(
            0, 2*np.pi - (2*np.pi/self._ant_num), self._ant_num)
        self.ant_distances = np.zeros(self._ant_num)
        self.xm = self.x + round(self.w/2)
        self.ym = self.y + round(self.h/2)

    def update(self, environment):
        """Ball class update method 
        
        environment -- Sprite class with line of sight blocking
                       and collidable terrain.
                       
        """

        #self.v += -1*np.sign(self.v)*(self.f)/FPS
        waypoint_dist = abs(self.path[0,0]-self.xm) + abs(self.path[0,1]-self.ym)
        waypoints_left = np.size(self.path,0) 

        if (waypoints_left > 1):
            pygame.draw.line(WIN, (0, 255, 0), 
                (self.xm, self.ym), (self.path[0,]), 2)
            for i in range(np.size(self.path,0)-1):
                pygame.draw.line(WIN, (0, 255, 0), 
                    (self.path[i,]), (self.path[i+1,]), 2)

        if (waypoint_dist < self.w*0.4):
            if (waypoints_left > 1):
                self.path = np.delete(self.path, 0, 0) # delete reached waypoint
            else:
                self.v = 0
        else:
            self.v = 40*60/FPS

        self.phi = np.math.atan2((self.path[0,0]-self.xm),
                                 (self.path[0,1]-self.ym))
        self.x += self.v*math.sin(self.phi)/FPS
        self.y += self.v*math.cos(self.phi)/FPS
        self.rect = pygame.Rect(self.x, self.y, self.w, self.h)
        self.xm = self.x + round(self.w/2)
        self.ym = self.y + round(self.h/2)
        
        # Measure distances to obstacles in specified directions
        for i in range(0,self._ant_num):
            self.ant_distances[i],xt ,yt = tu.cast_line(
                self.x + self.w/2, self.y + self.h/2, 
                self.antennas[i], environment.LOS_mask)
            pygame.draw.line(WIN, (255, 0, 0), 
                (self.x+self.w/2, self.y+self.h/2), (xt, yt), 1)
            
        col= pygame.sprite.collide_mask(self,environment)
        if col is not None:
            self.v = -1*self.v
""" End of Ball class """



class Background(pygame.sprite.Sprite):
    """Engine environment class
    
    !This should not change in time and only be read!
    
    This creates navigation map from low-resolution .png (i.e. 100x100)
    and scales it up to WIDTH and HEIGHT dimensions. Navigation is done
    on "weights" attribute, string pulling on "weights_scaled",
    graphics with "image" attribute and line of sight checking with
    "LOS_mask" attribute. "Weights" array is dilated with constant 
    "NAV_MARGIN" and then scaled up to "weights_scaled". 
    """
    
    def __init__(self, img_path):
        """Init method for Background class
        
        img_path -- path to source low-resolution image (.png)
        """
        super(Background, self).__init__()
        self.x = 0
        self.y = 0
        self.image = pygame.image.load(os.path.join(MAPS_DIR,img_path))
        self.image.convert_alpha()
        w, h = self.image.get_size()

        # load image and create weights array
        self.weights_scale = WIDTH/w
        self.weights = pygame.surfarray.array2d(self.image)*-1
        self.weights = self.weights.astype(np.float32)

        # dilate image to get safety navigation margin
        dil = math.ceil(NAV_MARGIN/self.weights_scale)
        dilated = tu.dilate_image(self.weights, dil, "max")
        self.weights_scaled = tu.resize_image(dilated, WIDTH, HEIGHT) 
        obstacles = np.where(dilated != 0)
        self.weights[obstacles[0], obstacles[1]] = np.inf
        self.weights += 1 # For astar search - lowest value is 1

        self.image = pygame.transform.scale(self.image, (WIDTH, HEIGHT))
        self.rect = self.image.get_rect()
        self.LOS_mask = pygame.surfarray.array2d(self.image)
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        pass  # placeholder
"""End of Background class """      
    
    
def draw_window():
    pygame.display.update()

def main():
    X0, Y0 = 11, 11 # low-res grid!!!
    XT, YT = 80, 58 # low-res grid!!!

    clock = pygame.time.Clock()
    arena = Background("map_0_100x100.png") #map_plain.png
    arenasprite = pygame.sprite.RenderPlain(arena)

    ball = Ball(X0*arena.weights_scale, Y0*arena.weights_scale,
                0, 0, 0, "ball.png")
    ballsprite = pygame.sprite.RenderPlain(ball)
    
    # PATHFINDING
    path = pyastar2d.astar_path(arena.weights,
        (X0, Y0), (XT, YT), allow_diagonal = False)

    if path is not None:
        path_scaled = path * arena.weights_scale + math.floor(arena.weights_scale/2)
        path_optimal = tu.string_pulling(path_scaled, arena.weights_scaled)
        print(path_optimal)
        ball.path = path_optimal.copy()

    WIN.fill(WHITE)
    #WIN.blit(arena.weights_mask, arena.rect)
    arenasprite.draw(WIN)

    print(path_optimal)
    for i in range(np.size(path,0)-1):
        pygame.draw.line(WIN, (255, 0, 0), 
            (path_scaled[i,]), (path_scaled[i+1,]), 2)
    
    for i in range(np.size(path_optimal,0)-1):
        pygame.draw.line(WIN, (0, 255, 0), 
            (path_optimal[i,]), (path_optimal[i+1,]), 2)

    pygame.display.update()
    # PATHFINDING END

    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        WIN.fill(WHITE)
        ball.update(arena)
        arenasprite.draw(WIN)    
        ballsprite.draw(WIN)
        
        pygame.display.update()     
        #draw_window()

    pygame.quit()

if __name__ == "__main__":
    main()