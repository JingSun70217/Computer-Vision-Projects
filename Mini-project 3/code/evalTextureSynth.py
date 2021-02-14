import numpy as np
from skimage import io
from datetime import datetime
from time import time
from random import randint
from math import ceil, floor

def synthRandomPatch(im, tileSize, numTiles, outSize):
    start_t = datetime.now()

    [width, length, depth] = im.shape
    im_out = np.zeros([outSize, outSize, depth])
    half_tile_size = int(tileSize/2)
    for i in range(numTiles):
        for j in range(numTiles):
            rand_x = np.random.randint(low=0, high=length-tileSize)
            rand_y = np.random.randint(low=0, high=width-tileSize)
            tile = im[rand_y:rand_y+tileSize,
                      rand_x:rand_x+tileSize,
                      :]
            im_out[i*tileSize:i*tileSize + tileSize, j*tileSize:j*tileSize+tileSize, :] = tile

    end_t = datetime.now()
    print "Process Time of RandomPatch: ", int((end_t - start_t).total_seconds() * 1000), " milliseconds"
    return im_out

def process_pixel(x, y, img_data, new_img_data, mask, win_size):
    ErrThreshold = 0.1
    win_size = int(win_size/2)
    x0 = max(0, x - win_size)
    y0 = max(0, y - win_size)
    x1 = min(new_img_data.shape[0], x + win_size+1)
    y1 = min(new_img_data.shape[1], y + win_size+1)

    new_window = new_img_data[x0 : x1, y0 : y1, 0]
    mask_window = mask[x0 : x1, y0 : y1]
    #plt.figure()
    #plt.imshow(neigh_window.astype(int))
    #plt.show()

    len_mask = float(sum(sum((mask_window))))

    [xs, ys] = new_window.shape
    img_xsize = img_data.shape[0]
    img_ysize = img_data.shape[1]

    cx = int(np.floor(xs/2))
    cy = int(np.floor(ys/2))

    BestMatches = []
    dists = []

    for i in range(xs, img_xsize - xs):
        for j in range(ys, img_ysize - ys):
            if(np.random.randint(0,2) != 0): continue # To improve runtime

            origin_window = img_data[i : i+xs, j : j+ys, 0]

            # distance
            s = (origin_window - new_window)


            error_matrix = (s**2)*mask_window

            d = sum(sum(error_matrix)) / len_mask

            BestMatches.append(origin_window[cx, cy])
            dists.append(d)

    best_dists_index = (dists) <= (1 + ErrThreshold) *  min(dists)

    BestMatches = np.extract(best_dists_index, BestMatches)

    # pick random among candidates
    if len(BestMatches) < 1:
        return 0.0
    else:
        if len(BestMatches) != 1:
            r = np.random.randint(0, len(BestMatches) - 1)
        else:
            r = 0

    return BestMatches[r]

def getPixelList(valid_pixel_map):
    pixel_list = []

    for i in range(0, valid_pixel_map.shape[0]):
        for j in range(0, valid_pixel_map.shape[1]):
            x0 = max(0, i - 1)
            y0 = max(0, j - 1)
            x1 = min(valid_pixel_map.shape[0], i + 2)
            y1 = min(valid_pixel_map.shape[1], j + 2)

            if (valid_pixel_map[i,j] == 0):
                if(sum(sum(valid_pixel_map[x0:x1, y0:y1])) > 0):
                    pixel_list.append([i, j])

    return np.array(pixel_list)

def synthEfrosLeung(img, winSize, outSize):
    start_t = datetime.now()
    [width, length, depth] = img.shape

    seed_size = 3
    half_output_size = int(outSize/2)
    half_seed_size = 1
    half_win_size = int(winSize/2)

    img_out = np.zeros([outSize, outSize, depth])
    valid_pixel_map = np.zeros((outSize, outSize))  # 0 means pixel not filled yet, 1 means pixel already filled

    seed_x = np.random.randint(low=0, high=img.shape[0] - seed_size)
    seed_y = np.random.randint(low=0, high=img.shape[1] - seed_size)

    # take 3x3 start image (seed) in the original image
    seed_data = img[seed_x: seed_x + seed_size, seed_y: seed_y + seed_size, :]

    img_out[half_output_size - half_seed_size: half_output_size + half_seed_size+1,
            half_output_size - half_seed_size: half_output_size + half_seed_size+1, :] = seed_data

    valid_pixel_map[half_output_size - half_seed_size: half_output_size + half_seed_size+1,
                    half_output_size - half_seed_size: half_output_size + half_seed_size+1] = 1

    # TO DO: non-square images
    while (int(np.sum(valid_pixel_map)) != valid_pixel_map.size):#(getPixelList(valid_pixel_map.astype(int)).all()): #
        print "Processed {} of {} pixels".format(np.sum(valid_pixel_map), valid_pixel_map.size)
        pixel_list = getPixelList(valid_pixel_map.astype(int))

        for pixel_position in pixel_list:
            x = pixel_position[0]
            y = pixel_position[1]

            pixel_out = process_pixel(x, y, img, img_out, valid_pixel_map, winSize)
            img_out[x, y] = pixel_out
            valid_pixel_map[x, y] = 1

    end_t = datetime.now()
    print "Process Time of Efros Matching: ", int((end_t - start_t).total_seconds() * 1000), " milliseconds"
    return img_out

# Load images
img = io.imread('../data/texture/D20.png')
#img = io.imread('../data/texture/Texture2.bmp')
#img = io.imread('../data/texture/english.jpg')


# Random patches
tileSize = 20 # specify block sizes
numTiles = 5
outSize = numTiles * tileSize # calculate output image size
# implement the following, save the random-patch output and record run-times
im_patch = synthRandomPatch(img, tileSize, numTiles, outSize).astype(int)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(im_patch)
plt.show()

# Non-parametric Texture Synthesis using Efros & Leung algorithm  
winsize = 11 # specify window size (5, 7, 11, 15)
outSize = 50 # specify size of the output image to be synthesized (square for simplicity)
# implement the following, save the synthesized image and record the run-times
im_synth = synthEfrosLeung(img, winsize, outSize)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(im_synth[:, :, 0:3].astype(int))
plt.show()