"""
BUTanks engine utility module

contains various function used in the engine, mainly image manipulation
"""

import numpy as np


def string_pulling(path, environment):
    """Straighten path from grid to account for non-grid movement
    
    path -- numpy array n x 2 of [x,y] path waypoints
    environment -- numpy array of movement blocking terrain

    returns m x 2 numpy array of optimized path
    """
    path = path.astype(int)

    start = 0 # starting node
    target = path[-1,]

    n = np.size(path,0)
    k = 1
    opt_path = np.array(path[0,], ndmin=2)
    while k < n:
        if has_LOS(path[start,0], path[start,1], 
                path[k,0], path[k,1], environment) is False:
            opt_path = np.append(opt_path, np.atleast_2d(path[k-1]), 0)
            start = k
        k += 1
    
    opt_path = np.append(opt_path, np.atleast_2d(path[-1,]), 0)

    return opt_path    



def has_LOS(x0, y0, xt, yt, environment, stride = 1):
    """Check if [x0, y0] has line of sight to [xt, yt]
    
    Check pixel by pixel whether environment block LOS between two coordinates
    x0, y0 -- first pixel (integer)
    xt, yt -- target pixel (integer)
    environment -- numpy array of LOS blocking terrain
                -- 0 means non blocking, other values blocking 

    return True/False whether there is/is not LOS between points            
    """
    if(environment[x0,y0] != 0):
        return False
        
    if (xt-x0) == 0:
        kinv = 0
        if (yt-y0) == 0:
            return True
    else:    
        k = (yt-y0)/(xt-x0)
        if k != 0:
            kinv = 1/k

    passed = True
    
    if abs(x0-xt) >= abs(y0-yt):
        for x in range(x0, xt+1, stride*np.sign(xt-x0)):
            y = round(y0 + k*(x-x0))
            if environment[x,y] != 0:
                passed = False
                break
    else:
        for y in range(y0, yt+1, stride*np.sign(yt-y0)):
            x = round(x0 + kinv*(y-y0))
            if environment[x,y] != 0:
                passed = False
                break  
    
    return passed

def cast_line(x0, y0, phi, env, stride = 15):
    """Project line from point at an angle until it reaches obstacle 
    
    x0, y0 -- starting point
    phi -- line angle
    env -- numpy array of environment
    stride -- function checks only each nth pixel and then backsteps to find
              intersection point. Higher value ~ better performance, but can
              skip corners, or entire walls when set too great.

    returns euclidean distance to the obstacle and that point's coordinates 
    """

    if phi > 2*np.pi:
        phi = phi - 2*np.pi*(phi // (2*np.pi))
    elif phi < 0:
        phi = phi - 2*np.pi*(phi // (2*np.pi))

    x = round(x0)
    y = round(y0) 

    i = 0
    k = np.math.tan(phi)

    if abs(k) > 1: # Y axis is primary
        while (env[x,y] == 0):
            if (phi > 0) & (phi < np.pi):
                y += stride
            else:
                y -= stride
            
            x = round(x0 + 1/k*(y-y0))
        # Backstepping
        while (env[x, y] != 0):
            if (phi > 0) & (phi < np.pi):
                y -= 1
            else:
                y += 1
            x = round(x0 + 1/k*(y-y0))

    else: # X axis is primary
        while (env[round(x),round(y)] == 0):
            if (phi > np.pi*3/2) | (phi < np.pi/2):
                x += stride
            else:
                x -= stride 
            y = round(y0 + k*(x-x0))
        # Backstepping
        while (env[round(x),round(y)] != 0):
            if (phi > np.pi*3/2) | (phi < np.pi/2):
                x -= 1
            else:
                x += 1 
            y = round(y0 + k*(x-x0))

    dist = np.sqrt((x-x0)**2 + (y-y0)**2)
    return dist, x, y

def is_close(a,b,tolerance):
    """Check if abs(a-b) is within tolerance """
    if abs(a-b) < tolerance:
        return True
    else:
        return False

def dilate_image(im_src, dilation_level = 1, method = "min"):
    """Use min or max filter to dilate/erode image
    
    Uses min/max filter on kernel of specified size. Kernel size is 
    dilation_level*2 + 1. Replaces middle value with min/max value from
    the kernel area. Works best on binary (highly contrast) images

    Arguments:
    im_src -- numpy array of pixel values
    dilation_level -- radius of kernel (default = 1)
    method -- "min" or "max" filter (default "min")
    """

    ker_size = 2*dilation_level + 1
    skip = dilation_level

    # pad array with constants
    im_pad = np.pad(array=im_src, pad_width=skip, mode='constant')
    pad_shape = im_pad.shape
    im_copy = im_src.copy()

    # iterate through array and apply filter
    for r in range(skip, pad_shape[0]-skip-1):
        for c in range(skip, pad_shape[1]-skip-1):
            if method == "min":
                im_copy[r-skip, c-skip] = np.min(
                    im_pad[(r-skip):(r+skip+1), (c-skip):(c+skip+1)])
            elif method == "max":
                im_copy[r-skip, c-skip] = np.max(
                    im_pad[(r-skip):(r+skip+1), (c-skip):(c+skip+1)])
    
    return im_copy

def resize_image(im_src, new_w, new_h):
    """Resize input image to specified dimensions
    
    im_src -- numpy array of pixel values
    new_w -- desired image width
    new_h -- desired image height
    """

    original_image = im_src.copy()
    width, height = new_w, new_h
    resize_im = np.zeros(shape=(width,height))
    
    for W in range(width):
        for H in range(height):
            new_width = int( W * original_image.shape[0] / width )
            new_height = int( H * original_image.shape[1] / height )
            resize_im[W][H] = original_image[new_width][new_height]
    
    return resize_im
