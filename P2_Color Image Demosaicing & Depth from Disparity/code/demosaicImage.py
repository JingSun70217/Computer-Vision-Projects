# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2018
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Homework 1 

import numpy as np
import scipy.ndimage.filters as scif


def demosaicImage(image, method):
    ''' Demosaics image.

    Args:
        img: np.array of size NxM.
        method: demosaicing method (baseline or nn).

    Returns:
        Color image of size NxMx3 computed using method.
    '''
    if method.lower() == "baseline":
        return demosaicBaseline(image.copy())
    elif method.lower() == 'nn':
        return demosaicNN(image.copy()) # Implement this
    elif method.lower() == 'linear':
        return demosaicLinear(image.copy()) # Implement this
    elif method.lower() == 'adagrad':
        return demosaicAdagrad(image.copy()) # Implement this
    elif method.lower() == "transformratio":
        return demosaicTransformRatio(image.copy())  # Extension
    elif method.lower() == 'transformlogratio':
        return demosaicTransformLogRatio(image.copy())    # Extension
    else:
        raise ValueError("method {} unkown.".format(method))


def demosaicBaseline(img):
    '''Baseline demosaicing.
    
    Replaces missing values with the mean of each color channel.
    
    Args:
        img: np.array of size NxM.

    Returns:
        Color image of sieze NxMx3 demosaiced using the baseline 
        algorithm.
    '''
    
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape

    red_values = img[0:image_height:2, 0:image_width:2]
    mean_value = red_values.mean()
    mos_img[:, :, 0] = mean_value
    mos_img[0:image_height:2, 0:image_width:2, 0] = img[0:image_height:2, 0:image_width:2]

    blue_values = img[1:image_height:2, 1:image_width:2]
    mean_value = blue_values.mean()
    mos_img[:, :, 2] = mean_value
    mos_img[1:image_height:2, 1:image_width:2, 2] = img[1:image_height:2, 1:image_width:2]

    mask = np.ones((image_height, image_width))
    mask[0:image_height:2, 0:image_width:2] = -1
    mask[1:image_height:2, 1:image_width:2] = -1
    green_values = mos_img[mask > 0]
    mean_value = green_values.mean()

    green_channel = img
    green_channel[mask < 0] = mean_value
    mos_img[:, :, 1] = green_channel

    return mos_img


def demosaicNN(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    height, width = img.shape
    mask = np.ones((height, width))
    mask[0:height:2, 0:width:2] = -1
    mask[1:height:2, 1:width:2] = -1

    ## red channel
    mos_img[0:height:2, 1:width:2, 0] = mos_img[0:height:2, 0:width-1:2, 0]  ## left
    mos_img[1:height:2, 0:width, 0] = mos_img[0:height-1:2, 0:width, 0]  ## up

    # ## green channel
    mos_img[0:height:2, 0:width-1:2, 1] = mos_img[0:height:2, 1:width:2, 1] ## right
    if width %2 == 1:
        mos_img[0:height:2, width-1, 1] = mos_img[0:height:2, width-2, 1] ## left 
    mos_img[1:height:2, 1:width:2, 1] = mos_img[1:height:2, 0:width-1:2, 1] ## left

    ## blue channel
    mos_img[1:height:2, 0:width-1:2, 2] = mos_img[1:height:2, 1:width:2, 2]  ## right
    mos_img[0:height-1, 0:width, 2] = mos_img[1:height, 0:width, 2]  ## down
    if height % 2 == 1 :
        mos_img[height-1, :, 2] = mos_img[height-2, :, 2]
    if width %2 == 1:
        mos_img[:, width-1, 2] = mos_img[:, width-2, 2]
    
    return mos_img


def demosaicLinear(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    height, width = img.shape
    mask = np.ones((height, width))
    mask[0:height:2, 0:width:2] = -1
    mask[1:height:2, 1:width:2] = -1

    ## red channel
    mos_img[1:height:2, 1:width:2, 0] = 0  # blue=0
    mos_img[mask>0, 0] = 0         # green = 0
    kernel1 = np.array([1,2,1], dtype=np.float)
    kernel1 /= 2
    mos_img_row = scif.convolve1d(mos_img[:,:,0], kernel1, axis = 1)

    mos_img_row[0:height:2, 0:width:2] = img[0:height:2, 0:width:2]
    mos_img[:,:,0] = scif.convolve1d(mos_img_row, kernel1, axis = 0)
    mos_img[0:height:2, 0:width:2, 0] = img[0:height:2, 0:width:2]

    ## green channel
    mos_img[0:height:2, 0:width:2, 1] = 0   ## red = 0
    mos_img[1:height:2, 1:width:2, 1] = 0  # blue=0
    kernel2 = np.array([[0,1,0], [1,4,1], [0,1,0]], dtype=np.float)
    kernel2 /= 4
    mos_img[:,:,1] = scif.convolve(mos_img[:,:,1], kernel2)
    mos_img[mask>0, 1] = img[mask>0] 

    ## blue channel
    mos_img[mask>0, 2] = 0   ## green = 0
    mos_img[0:height:2, 0:width:2, 2] = 0   ## red = 0
    kernel3 = kernel1
    mos_img_row = scif.convolve1d(mos_img[:,:,2], kernel3, axis = 1)

    mos_img_row[1:height:2, 1:width:2] = img[1:height:2, 1:width:2]
    mos_img[:,:,2] = scif.convolve1d(mos_img_row, kernel3, axis = 0)
    mos_img[1:height:2, 1:width:2, 2] = img[1:height:2, 1:width:2]

    return mos_img


def demosaicAdagrad(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    height, width = img.shape
    mask = np.ones((height, width))
    mask[0:height:2, 0:width:2] = -1
    mask[1:height:2, 1:width:2] = -1

    ## red channel - linear
    mos_img[1:height:2, 1:width:2, 0] = 0  # blue=0
    mos_img[mask>0, 0] = 0         # green = 0
    kernel1 = np.array([1,2,1], dtype=np.float)
    kernel1 /= 2
    mos_img_row = scif.convolve1d(mos_img[:,:,0], kernel1, axis = 1)

    mos_img_row[0:height:2, 0:width:2] = img[0:height:2, 0:width:2]
    mos_img[:,:,0] = scif.convolve1d(mos_img_row, kernel1, axis = 0)
    mos_img[0:height:2, 0:width:2, 0] = img[0:height:2, 0:width:2]

    ## green channel - adaptive gradient channel
    mos_img[0:height:2, 0:width:2, 1] = 0   ## red = 0
    mos_img[1:height:2, 1:width:2, 1] = 0  # blue=0

    kernel2_diff = np.array([1, 0, -1], dtype=np.float)
    diff_vertical = np.abs(scif.convolve1d(mos_img[:, :, 1], kernel2_diff, axis = 0))
    diff_horizontal = np.abs(scif.convolve1d(mos_img[:, :, 1], kernel2_diff, axis = 1))
    
    compare = diff_vertical - diff_horizontal
    
    kernel2 = np.array([1, 0, 1], dtype = np.float)
    kernel2 /= 2
    mos_img_temp1 = scif.convolve1d(mos_img[:, :, 1], kernel2, axis = 1)
    mos_img[compare > 0, 1] = mos_img_temp1[compare > 0]
    mos_img_temp2 = scif.convolve1d(mos_img[:, :, 1], kernel2, axis = 0)
    mos_img[compare <= 0, 1] = mos_img_temp2[compare <= 0]
    # kernel2 = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype = np.float)
    # kernel2 /= 4
    # mos_img_temp = scif.convolve(mos_img[:, :, 1], kernel2)
    # mos_img[compare == 0, 1] = mos_img_temp[compare == 0]
    mos_img[mask>0, 1] = img[mask>0]

    ## blue channel =  linear
    mos_img[mask>0, 2] = 0   ## green = 0
    mos_img[0:height:2, 0:width:2, 2] = 0   ## red = 0
    kernel3 = kernel1
    mos_img_row = scif.convolve1d(mos_img[:,:,2], kernel3, axis = 1)

    mos_img_row[1:height:2, 1:width:2] = img[1:height:2, 1:width:2]
    mos_img[:,:,2] = scif.convolve1d(mos_img_row, kernel3, axis = 0)
    mos_img[1:height:2, 1:width:2, 2] = img[1:height:2, 1:width:2]

    return mos_img

def demosaicTransformRatio(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    height, width = img.shape
    mask = np.ones((height, width))
    mask[0:height:2, 0:width:2] = -1
    mask[1:height:2, 1:width:2] = -1

    ## green channel - adaptive gradient channel
    mos_img[0:height:2, 0:width:2, 1] = 0   ## red = 0
    mos_img[1:height:2, 1:width:2, 1] = 0  # blue=0

    kernel2_diff = np.array([1, 0, -1], dtype=np.float)
    diff_vertical = np.abs(scif.convolve1d(mos_img[:, :, 1], kernel2_diff, axis = 0))
    diff_horizontal = np.abs(scif.convolve1d(mos_img[:, :, 1], kernel2_diff, axis = 1))
    compare = diff_vertical - diff_horizontal
    kernel2 = np.array([1, 2, 1], dtype = np.float)
    kernel2 /= 2
    mos_img_temp1 = scif.convolve1d(mos_img[:, :, 1], kernel2, axis = 1)
    mos_img[compare > 0, 1] = mos_img_temp1[compare > 0]
    mos_img_temp2 = scif.convolve1d(mos_img[:, :, 1], kernel2, axis = 0)
    mos_img[compare <= 0, 1] = mos_img_temp2[compare <= 0]
 
    mos_img[mask>0, 1] = img[mask>0]

    ## red channel - transform
    mos_img[:, :, 0] /= mos_img[:, :, 1] + 1e-6  ## first transform to R/G
    mos_img[1:height:2, 1:width:2, 0] = 0  # blue=0
    mos_img[mask>0, 0] = 0         # green = 0
    kernel1_row = np.array([[0,1,0],[0,2,0],[0,1,0]], dtype=np.float)
    kernel1_row /= 2
    mos_img_row = scif.convolve(mos_img[:,:,0], kernel1_row)

    kernel1_col = np.array([[0,0,0],[1,2,1],[0,0,0]], dtype=np.float)
    kernel1_col /= 2
    mos_img[:,:,0] = scif.convolve(mos_img_row, kernel1_col) * (mos_img[:, :, 1] + 1e-6)
    mos_img[0:height:2, 0:width:2, 0] = img[0:height:2, 0:width:2]    ## transform back

    ## blue channel =  linear
    mos_img[:, :, 2] /= mos_img[:, :, 1] + 1e-6  ## first transform to B/G
    mos_img[mask>0, 2] = 0   ## green = 0
    mos_img[0:height:2, 0:width:2, 2] = 0   ## red = 0
    kernel3_row = kernel1_row
    mos_img_row = scif.convolve(mos_img[:,:,2], kernel3_row)

    kernel3_col = kernel1_col
    mos_img[:,:,2] = scif.convolve(mos_img_row, kernel3_col) * (mos_img[:, :, 1] + 1e-6)
    mos_img[1:height:2, 1:width:2, 2] = img[1:height:2, 1:width:2]

    return mos_img

def demosaicTransformLogRatio(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    height, width = img.shape
    mask = np.ones((height, width))
    mask[0:height:2, 0:width:2] = -1
    mask[1:height:2, 1:width:2] = -1

    ## green channel - adaptive gradient channel
    # mos_img[0:height:2, 0:width:2, 1] = 0   ## red = 0
    mos_img[mask<0, 1] = 0  # blue=0

    kernel2_diff = np.array([1, 0, -1], dtype=np.float)
    diff_vertical = np.abs(scif.convolve1d(mos_img[:, :, 1], kernel2_diff, axis = 0))
    diff_horizontal = np.abs(scif.convolve1d(mos_img[:, :, 1], kernel2_diff, axis = 1))
    compare = diff_vertical - diff_horizontal
    kernel2 = np.array([1, 0, 1], dtype = np.float)
    kernel2 /= 2
    mos_img_temp1 = scif.convolve1d(mos_img[:, :, 1], kernel2, axis = 1)
    mos_img[compare > 0, 1] = mos_img_temp1[compare > 0]
    mos_img_temp2 = scif.convolve1d(mos_img[:, :, 1], kernel2, axis = 0)
    mos_img[compare <= 0, 1] = mos_img_temp2[compare <= 0]
    mos_img[mask>0, 1] = img[mask>0]

    ## red channel - transform
    mos_img[:, :, 0] = np.log(mos_img[:, :, 0]/(mos_img[:, :, 1] + 1e-6) + 1e-6)  ## first transform to R/G
    mos_img[1:height:2, 1:width:2, 0] = 0  # blue=0
    mos_img[mask>0, 0] = 0         # green = 0
    kernel1_row = np.array([[0,1,0],[0,2,0],[0,1,0]], dtype=np.float)
    kernel1_row /= 2
    mos_img_row = scif.convolve(mos_img[:,:,0], kernel1_row)

    kernel1_col = np.array([[0,0,0],[1,2,1],[0,0,0]], dtype=np.float)
    kernel1_col /= 2
    mos_img[:,:,0] = np.exp(scif.convolve(mos_img_row, kernel1_col)) * (mos_img[:, :, 1] + 1e-6)
    # mos_img[:, :, 0] = np.power(* mos_img[:, :, 1]
    mos_img[0:height:2, 0:width:2, 0] = img[0:height:2, 0:width:2]    ## transform back

    ## blue channel =  linear
    mos_img[:, :, 2] = np.log(mos_img[:, :, 2]/(mos_img[:, :, 1] + 1e-6) + 1e-6)  ## first transform to B/G
    mos_img[mask>0, 2] = 0   ## green = 0
    mos_img[0:height:2, 0:width:2, 2] = 0   ## red = 0
    kernel3_row = kernel1_row
    mos_img_row = scif.convolve(mos_img[:,:,2], kernel3_row)

    kernel3_col = kernel1_col
    mos_img[:,:,2] = np.exp(scif.convolve(mos_img_row, kernel3_col)) * (mos_img[:, :, 1] + 1e-6)
    mos_img[1:height:2, 1:width:2, 2] = img[1:height:2, 1:width:2]

    return mos_img
