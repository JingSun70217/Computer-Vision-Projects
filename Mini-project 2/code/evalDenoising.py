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
from utils import imread, gaussian
import scipy.ndimage.filters as scif
import scipy.signal as scinal
import time

def DenoiseGaussian(im, noise_image):
	'''
	Denoising algorithm (Gaussian filtering)
	'''
	sigma = (0.8, 0.9, 1, 2, 3)
	width = (3, 5, 7, 11)
	plt.figure(2)
	for n in range(4):
		error_best = np.Inf
		noise = noise_image[n]
		for w in width:
			for s in sigma:
				kernel_Gaussian = gaussian(w, s)
				output_n = scif.convolve(noise, kernel_Gaussian, mode='constant', cval=0.0)
				# output_n2 = scif.convolve(noise2, kernel_Gaussian, mode='constant', cval=0.0)
				error = ((im - output_n)**2).sum()
				# error_2 = ((im - output_n2)**2).sum()
				print 'filter size = {}, sigma = {}, error = {}'.format(w, s, error)
				if error < error_best:
					error_best = error
					wb = w
					sb = s
		print wb, sb
		denoise_best = scif.convolve(noise, gaussian(wb,sb), mode='constant', cval=0.0)
		plt.subplot(141+n)
		plt.imshow(denoise_best)
		plt.title('SE {:.2f}'.format(error_best))

	plt.show()

def DenoiseMedian(im, noise_image):
	'''
	Denoising algorithm (Median filtering)
	'''
	neighbor_size = (1, 3, 5, 7)
	plt.figure(3)
	for n in range(4):
		error_best = np.Inf
		noise = noise_image[n]
		for sz in neighbor_size:
			denoise = scinal.medfilt2d(noise, sz)
			error = ((im - denoise)**2).sum()
			print 'neighbor_size = {}, error = {}'.format(sz, error)
			if error < error_best:
					error_best = error
					szb = sz
		print szb
		denoise_best = scinal.medfilt2d(noise, szb)

		plt.subplot(141+n)
		plt.imshow(denoise_best)
		plt.title('SE {:.2f}'.format(error_best))

	plt.show()


def DebugCrop(im, noise):
	'''
	parameters debug on crop image
	'''
	height,width = noise.shape
	crop_im = noise2[int(0.2*height):int(0.5*height), int(0.2*height):int(0.5*width)]
	y,x = crop_im.shape
	# pp = (3, 5, 7)
	# ww = (5, 7, 9, 11)
	# rr = (1, 2, 3, 4)
	# pp = (9,11)
	# ww = (13, 15, 17, 21)
	# rr = (5, 6, 7, 9)
	pp = (3, 5, 7)
	ww = (5, 9, 11)
	rr = (1, 2, 3, 4)

	for p in pp:
		for w in ww:
			for r in rr:
				start = time.time()
				denoise_crop = np.zeros(crop_im.shape, dtype=np.float)
				pad_size = (w-1)/2 + (p-1)/2
				pad_im = np.pad(crop_im, pad_size, 'constant')
				
				pshift = (p-1)/2
				wshift = (w-1)/2
				for i in range(pad_size, y+pad_size):
					for j in range(pad_size, x+pad_size):
						P_x = pad_im[i-pshift: i+pshift+1, j-pshift: j+pshift+1]
						diff_up = 0.0
						diff_down = 0.0
						# print 'i = {}, j ={}'.format(i,j)
						for k in range(i-wshift, i+wshift+1):
							for t in range(j-wshift, j+wshift+1):
								P_y = pad_im[k-pshift: k+pshift+1, t-pshift: t+pshift+1]
								# print P_y.shape, P_x.shape
								Euclidean_dis = np.linalg.norm((P_x - P_y), 2)**2
								diff_up += np.exp(-r* Euclidean_dis) * pad_im[k, t]
								diff_down += np.exp(-r * Euclidean_dis) 
						denoise_crop[i-pad_size,j-pad_size] = diff_up/diff_down
						
				t = time.time() - start
				error1 = ((denoise_crop - im[int(0.2*height):int(0.5*height), int(0.2*height):int(0.5*width)])**2).sum()
				print 'p = {}, w = {}, r = {}, error = {}'.format(p, w, r, error1)
				print 'time is {}'.format(t)

def DenoiseNLM(im, noise_image):
	'''
	Denoising algorithm (Non-local means)
	'''
	pp = (3, 5, 7)
	ww = (9, 11)
	rr = (1, 2, 3)
	y,x = im.shape

	fig2 = plt.figure(4)
	for n in range(4):
		error_best = np.Inf
		for p in pp:
			for w in ww:
				for r in rr:
					noise = noise_image[n]
					start = time.time()
					denoise = np.zeros(noise.shape, dtype=np.float)
					pad_size = (w-1)/2 + (p-1)/2
					pad_im = np.pad(noise, pad_size, 'constant')


					pshift = (p-1)/2
					wshift = (w-1)/2
					for i in range(pad_size, y+pad_size):
						for j in range(pad_size, x+pad_size):
							P_x = pad_im[i-pshift: i+pshift+1, j-pshift: j+pshift+1]
							diff_up = 0.0
							diff_down = 0.0
							for k in range(i-wshift, i+wshift+1):
								for t in range(j-wshift, j+wshift+1):
									# print (i,j),(k, t)
									P_y = pad_im[k-pshift: k+pshift+1, t-pshift: t+pshift+1]
									# print P_y.shape, P_x.shape
									Euclidean_dis = np.linalg.norm((P_x - P_y), 2)**2
									diff_up += np.exp(-r* Euclidean_dis) * pad_im[k, t]
									diff_down += np.exp(-r * Euclidean_dis)
							denoise[i-pad_size,j-pad_size] = diff_up/diff_down

					t = time.time() - start
					error = ((denoise - im)**2).sum()
					print 'p = {}, w = {}, r = {}, error = {}'.format(p, w, r, error)
					print 'time is {}'.format(t)
					if error < error_best:
						error_best = error
						wb = w
						rb = r
						pb = p
						denoise_best = denoise
		print pb, wb, rb 

		plt.subplot(141+n)
		plt.imshow(denoise_best)
		plt.title('SE {:.2f}'.format(error_best))

	plt.show()


im = imread('../data/denoising/saturn.png')
noise11 = imread('../data/denoising/saturn-noise1g.png')
noise12 = imread('../data/denoising/saturn-noise2g.png')
noise21 = imread('../data/denoising/saturn-noise1sp.png')
noise22 = imread('../data/denoising/saturn-noise2sp.png')

# print im.shape, noise1.shape
error11 = ((im - noise11)**2).sum()
error21 = ((im - noise21)**2).sum()

error12 = ((im - noise12)**2).sum()
error22 = ((im - noise22)**2).sum()

print 'Input, Errors: {:.2f} {:.2f} {:.2f} {:.2f}'.format(error11, error12, error21, error22)

plt.figure(1)

plt.subplot(211)
plt.imshow(im)
plt.title('Input')

plt.subplot(245)
plt.imshow(noise11)
plt.title('SE {:.2f}'.format(error11))

plt.subplot(246)
plt.imshow(noise12)
plt.title('SE {:.2f}'.format(error12))

plt.subplot(247)
plt.imshow(noise21)
plt.title('SE {:.2f}'.format(error21))

plt.subplot(248)
plt.imshow(noise22)
plt.title('SE {:.2f}'.format(error22))

plt.show()

noise_image = (noise11, noise12, noise21, noise22)

DenoiseGaussian(im, noise_image)
DenoiseMedian(im, noise_image)
DenoiseNLM(im, noise_image)