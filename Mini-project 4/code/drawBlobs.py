import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# DRAWBLOBS overlays the image with blobs as circles
#   DRAWBLOBS(IM, BLOBS, THRESHOLD) overalys IM converted to a grayscale
#   image with BLOBS that are above a THRESHOLD. If THRESHOLD is not
#   specified it is set to 0. 
#
# Input:
#   IM - the image (if rgb image is provided it is converted to grayscale)
#   BLOBS - n x 4 matrix with each row is a blob (x, y, radius, score) 
#   THRESHOLD - only blobs above this are shown (default = 0)
#
# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
def drawBlobs(im, blobs, nmax=None):
    
    if nmax is None:
        nmax = blobs.shape[0]
    nmax = min(nmax, blobs.shape[0])

    if im.shape[2] > 1:
        im = rgb2gray(im)

    plt.figure()
    plt.imshow(im, cmap="gray")

    if nmax < 1:
        return 

    order = np.argsort(-blobs[:, 3])
    theta = np.linspace(0, 2*np.pi, 24)
    for i in xrange(nmax):
        r = blobs[order[i], 2]
        plt.plot(blobs[order[i], 0] + r*np.cos(theta),
                 blobs[order[i], 1] + r*np.sin(theta),
                 'r-', linewidth=2)

    plt.show()
