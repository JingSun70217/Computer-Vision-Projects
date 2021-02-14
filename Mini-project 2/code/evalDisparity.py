# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji

import numpy as np
import matplotlib.pyplot as plt
from utils import imread
from depthFromStereo import depthFromStereo

#Read test images
img1 = imread("../data/disparity/poster_im2.jpg")
img2 = imread("../data/disparity/poster_im6.jpg")
#img1 = imread("../data/disparity/tsukuba_im1.jpg")
#img2 = imread("../data/disparity/tsukuba_im5.jpg")

#Compute depth
n = 11
depth = depthFromStereo(img1, img2, ws = n, compare = "normCorrelation")

#Show result
cax = plt.imshow(depth)
plt.title("ws = {}".format(n))
#cbar = plt.colorbar(cax, ticks=[45, 25, 0])
cbar = plt.colorbar(cax, ticks=[3.6, 1.8, 0.0])
cbar.ax.set_yticklabels(['smaller \n detph', 'Medium', 'bigger'])

plt.show()


