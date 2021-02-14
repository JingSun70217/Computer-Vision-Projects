import os
import numpy as np
import matplotlib.pyplot as plt
from utils import imread
from detectBlobs import detectBlobs
from drawBlobs import drawBlobs

# Evaluation code for image stitching
# Your goal is to implement image stitching with RANSAC 
#
# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji

imageName = 'butterfly.jpg'
numBlobsToDraw = 500
imName = imageName.split('.')[0]

datadir = os.path.join('..', 'data', 'blobs')
im = imread(os.path.join(datadir, imageName))

blobs = detectBlobs(im)

drawBlobs(im, blobs, numBlobsToDraw)

