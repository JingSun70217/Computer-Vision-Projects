# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#

import numpy as np

def detectBlobs(im, param=None):
    # Input:
    #   IM - input image
    #
    # Ouput:
    #   BLOBS - n x 4 array with blob in each row in (x, y, radius, score)
    #
    # Dummy - returns a blob at the center of the image
    blobs = np.round(np.array([[im.shape[1]*0.5, 
                                im.shape[0]*0.5,
                                0.25*min(im.shape[0], im.shape[1]),
                                1]]))
    return blobs
