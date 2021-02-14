import numpy as np
import matplotlib.pyplot as plt
import os
from utils import imread
from utils import showMatches
from detectBlobs import detectBlobs
from compute_sift import compute_sift
from computeMatches import computeMatches
from ransac import ransac
from mergeImages import mergeImages

#Image directory
dataDir = os.path.join('..', 'data', 'stitching')

#Read input images
testExamples = ['hill', 'field', 'ledge', 'pier', 'river' 'roofs', 'building', 'uttower']
exampleIndex = 0
imageName1 = '{}1.jpg'.format(testExamples[exampleIndex])
imageName2 = '{}2.jpg'.format(testExamples[exampleIndex])

im1 = imread(os.path.join(dataDir, imageName1))
im2 = imread(os.path.join(dataDir, imageName2))

#Detect keypoints
blobs1 = detectBlobs(im1)
blobs2 = detectBlobs(im2)

#Compute SIFT features
sift1 = compute_sift(im1, blobs1[:, 0:3])
sift2 = compute_sift(im2, blobs2[:, 0:3])

#Find the matching between features
matches = computeMatches(sift1, sift2)
showMatches(im1, im2, blobs1, blobs2, matches)

#Ransac to find correct matches and compute transformation
inliers, transf = ransac(matches, blobs1, blobs2)

goodMatches = np.zeros_like(matches)
goodMatches[inliers] = matches[inliers]

showMatches(im1, im2, blobs1, blobs2, goodMatches)

#Merge two images and display the output
stitchIm = mergeImages(im1, im2, transf)
plt.figure()
plt.imshow(stitchIm)
plt.title('stitched image: {}'.format(testExamples[exampleIndex]))
