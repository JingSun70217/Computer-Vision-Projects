# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#

import os
import time
import numpy as np
import matplotlib.pyplot as plt 
import sys
from utils import imread
from scipy.ndimage.filters import *
from scipy.signal import *

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def denoising(input, method="Gaussian"):
    output = input.copy()
    if method is "Gaussian":
        sigma = 1.0
        gaussian_filter(input, sigma, output=output, mode="constant", cval=0)
    if method is "Median":
        output = medfilt2d(input, kernel_size= 5)
    return output

def denoisingNL(input, patch_size = 5, search_window_size = 5, bandwidth = 1):
    [width, length] = input.shape
    output = np.zeros(input.shape)
    half_patch_size = int(patch_size/2)
    half_search_window_size = int(search_window_size/2)
    input_padded = np.pad(input, pad_width=half_patch_size, mode=pad_with)

    for i in range(width):
        for j in range(length):
            i1 = i + half_patch_size
            j1 = j + half_patch_size

            window1 = input_padded[i1 - half_patch_size:i1 + half_patch_size, j1 - half_patch_size:j1 + half_patch_size]

            wmax = 0
            average = 0
            sweight = 0

            rmin = max(i1 - half_search_window_size, half_patch_size)
            rmax = min(i1 + half_search_window_size, width + half_patch_size)
            smin = max(j1 - half_search_window_size, half_patch_size)
            smax = min(j1 + half_search_window_size, length + half_patch_size)

            for r in range(rmin, rmax, 1):
                for s in range(smin, smax, 1):
                    if(r== i1 and s == j1):
                        continue
                    else:
                        window2 = input_padded[r - half_patch_size:r + half_patch_size, s - half_patch_size:s + half_patch_size]

                        d = sum(sum((window1 - window2) * (window1 - window2)))

                        w = np.exp(-bandwidth * d)

                        if w > wmax:
                            wmax = w

                        sweight = sweight + w
                        average = average + w * input_padded[r, s]

            average = average + wmax * input_padded[i1, j1]
            sweight = sweight + wmax
            if sweight > 0:
                output[i, j] = average / sweight
            else:
                output[i, j] = input[i, j]

    return output

im = imread('../data/denoising/saturn.png')
noise1 = imread('../data/denoising/saturn-noise1g.png')
noise2 = imread('../data/denoising/saturn-noise1sp.png')

noise1 = denoisingNL(noise1)
noise2 = denoising(noise2, "Median")
error1 = ((im - noise1)**2).sum()
error2 = ((im - noise2)**2).sum()

print 'Input, Errors: {:.2f} {:.2f}'.format(error1, error2)

plt.figure(1)

plt.subplot(131)
plt.imshow(im)
plt.title('Input')

plt.subplot(132)
plt.imshow(noise1)
plt.title('SE {:.2f}'.format(error1))

plt.subplot(133)
plt.imshow(noise2)
plt.title('SE {:.2f}'.format(error2))

plt.show()

# Denoising algorithm (Gaussian filtering)

# Denoising algorithm (Median filtering)

# Denoising algorithm (Non-local means)
