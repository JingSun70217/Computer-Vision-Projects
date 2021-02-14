import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def depthFromStereo(img1, img2, ws=3, compare="SSD"):
	#ws is the patch size
	max_diff = 50
	img1 = rgb2gray(img1)
	img2 = rgb2gray(img2)

	plt.show()
	img1 = img1/np.max(img1) * 255
	img2 = img2/np.max(img2) * 255
	# print img1, img2
	# print "*************"
	depth = np.zeros(img2.shape, dtype=np.float32)
	m, n = depth.shape
	# print m, n
	r = int(ws/2)
	# print r
	pad_img1 = np.lib.pad(img1, ((r,r),(r,r)), 'constant')
	pad_img2 = np.lib.pad(img2, ((r,r),(r,r)), 'constant')
	# print pad_img1.shape, pad_img2.shape
	for i in range(r, m+r):
		for j in range(r, n+r):
			measure = np.empty([max_diff, 1])

			left = pad_img1[(i-r):(i+r+1), (j-r):(j+r+1)]
			# print "***********"
			# print "i = {}, j = {}".format(i,j)
			for k in range(j-1, j-max_diff-1, -1):
				if (k-r) >= 0:
					right = pad_img2[(i-r):(i+r+1), (k-r):(k+r+1)]
					# print i, j, k
					# print "index = {}".format(j-1-k)
					if compare == "SSD":
						measure[j-1-k] = np.sum((left - right)**2)
					if compare == "normCorrelation":
						measure[j-1-k] = np.sum((left - np.mean(left))/np.linalg.norm(left-np.mean(left), 2) * (right - np.mean(right))/np.linalg.norm(right-np.mean(right), 2))
			# print measure
			# print np.argmin(measure)
			if compare == "SSD":
				disparity = j - (j-np.argmin(measure))
				print np.argmin(measure)
			if compare == "normCorrelation":
				disparity = j - (j-np.argmax(measure))
				print np.argmax(measure)
			if disparity == 0:
				disparity = 10
			# print np.argmin(ssd)
			# print 1.0/disparity
			depth[i-r,j-r] = -np.log(1.0/disparity)
			# depth[i-r,j-r] = disparity	
	# print depth
	return depth

def getPatchInRow(img2, row, ws):
# 	# compute ssd to get the smallest one
	patchInRow = np.zeros((img2.shape[1], ws*ws))
	r = int(ws/2)
	pad_img2 = np.lib.pad(img2, r, 'constant')
	for k in range(r, img2.shape[1]+r):
		patch = pad_img2[row-r:row+r+1, k-r: k+r+1]
		pt2v = np.ndarray.flatten(patch)
		patchInRow[k-r] = pt2v
		# print k, patch1.shape, patch2.shape

	return patchInRow
