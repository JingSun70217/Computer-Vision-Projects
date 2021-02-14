# This code is part of:
# 
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
# 
# Evaluation code for photometric stereo
# 
# Your goal is to implement the three functions prepareData(), 
# photometricStereo() and getSurface() to estimate the albedo and shape of
# the objects in the scene from multiple images. 
# 
# Start with setting subjectName='debug' which sets up a toy scene with
# known albedo and height which you can compare against. After you have a
# good implementation of this part, set the subjectName='yaleB01', etc. to
# run your code against real images of people. 
# 
# Credits: The homework is adapted from a similar one developed by
# Shvetlana Lazebnik (UNC/UIUC)


import os
import time
import numpy as np
import matplotlib.pyplot as plt 
import skimage.io as io

from alignChannels import *
from utils import *
from getSurface import *
from photometricStereo import *
from loadFaceImages import *
from toyExample import *
from prepareData import *
from displayOutput import *
from plotSurfaceNormals import *

subjectName = 'yaleB05' #debug, yaleB01, yaleB02, yaleB05, yaleB07
numImages = 128
writeOutput = True
data_dir = os.path.join('..', 'data')
out_dir = os.path.join('..', 'output', 'photometricStereo')
image_dir = os.path.join(data_dir, 'photometricStereo', subjectName)
integrationMethod = 'random'
mkdir(out_dir)

if subjectName == 'debug':
    imageSize = (64, 64)
    (ambientImage, imArray, lightDirs, trueAlbedo, trueSurfaceNormals, trueHeightMap) = toyExample(imageSize, numImages)
else:
    (ambientImage, imArray, lightDirs) = loadFaceImages(image_dir, subjectName, numImages)

# print 'gt of toy: ', trueAlbedo.shape, trueHeightMap.shape, trueSurfaceNormals.shape 

imArray = prepareData(imArray, ambientImage)

(albedoImage, surfaceNormals) = photometricStereo(imArray, lightDirs)
heightMap = getSurface(surfaceNormals, integrationMethod)

# plt.figure()
# plt.imshow(heightMap)
# plt.show()
# print 'pre of yale:',albedoImage.shape, heightMap.shape, surfaceNormals.shape
# print (heightMap-trueHeightMap, albedoImage-trueAlbedo, trueSurfaceNormals-surfaceNormals)
displayOutput(albedoImage, heightMap)
plotSurfaceNormals(surfaceNormals)

if subjectName == 'debug':
    displayOutput(trueAlbedo, trueHeightMap)
    plotSurfaceNormals(trueSurfaceNormals)


if writeOutput:
    imageName = os.path.join(out_dir, '{}_albedo.jpg'.format(subjectName))
    io.imsave(imageName, albedoImage)

    imageName = os.path.join(out_dir, '{}_normals_color.jpg'.format(subjectName))
    io.imsave(imageName, surfaceNormals)

    imageName = os.path.join(out_dir, '{}_normals_x.jpg'.format(subjectName))
    io.imsave(imageName, surfaceNormals[:, :, 0])

    imageName = os.path.join(out_dir, '{}_normals_y.jpg'.format(subjectName))
    io.imsave(imageName, surfaceNormals[:, :, 1])

    imageName = os.path.join(out_dir, '{}_normals_z.jpg'.format(subjectName))
    io.imsave(imageName, surfaceNormals[:, :, 2])

