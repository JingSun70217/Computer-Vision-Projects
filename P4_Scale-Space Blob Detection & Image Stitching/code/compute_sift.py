import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import imread
from skimage.color import rgb2gray
from skimage.feature import corner_orientations
from skimage.feature import hog

def compute_sift(I, circles, enlarge_factor=1.5):
# I - image
# circles - Nx3 array where N is the number of circles, where the
#    first column is the x-coordinate, the second column is the y-coordinate,
#    and the third column is the radius
# enlarge_factor is by how much to enarge the radius of the circle before
#    computing the descriptor (a factor of 1.5 or larger is usually necessary
#    for best performance)
# The output is an Nx128 array of SIFT descriptors

    sift = cv2.xfeatures2d.SIFT_create()

    angle_bins = 36
    I = rgb2gray(I)
    hist = hog(I, orientations=angle_bins, pixels_per_cell=(10, 10),
            cells_per_block=(1,1), feature_vector=False, block_norm='L2-Hys')

    xcoord = np.floor(circles[:,0]/10.0).astype(int)
    ycoord = np.floor(circles[:,1]/10.0).astype(int)

    circ_hist = hist[ycoord, xcoord, 0, 0, :]
    angles = np.rad2deg(np.argmax(circ_hist, axis=1) * 2*np.pi/angle_bins)

    img_gray = (I*255.0).astype('uint8')

    kpts = []
    for i in xrange(angles.shape[0]):
        kpts.append(cv2.KeyPoint(circles[i, 0], circles[i, 1], 
            _size=enlarge_factor*circles[i, 2],
            _angle=angles[i]))

    _, des = sift.compute(img_gray, kpts)
    return des

