import numpy as np
import scipy.linalg as sl

def photometricStereo(imarray, lightdirs):
	
	h = imarray.shape[0]
	w = imarray.shape[1]
	n = imarray.shape[2]
	print h,w,n
	imarray = np.reshape(imarray, (h*w, n))
	imarray = np.transpose(imarray)
	# print 'imarray shape and lightdirs shape', imarray.shape, lightdirs.shape
	# print imarray.shape, lightdirs.shape
	# imarray = np.transpose(imarray)
	# lightdirs = np.transpose(lightdirs)
	g, res, rnk, s = sl.lstsq(lightdirs, imarray)
	# print '** g shape', g.shape, g
	rho = np.sqrt(np.sum(g**2, axis=0))
	# print 'rho shape', rho.shape
	
	# rho_temp = np.transpose(rho)
	# rho = np.reshape(rho, (rho.shape[0], 1))
	rho_temp = np.tile(rho, (3, 1))
	# rho_temp = np.repeat(rho_temp, 3, axis=1)
	
	# g = np.transpose(g)
	normal = g/rho_temp

	normal = np.transpose(normal)
	normal = np.reshape(normal, (h, w, 3))

	rho = np.reshape(rho, (h, w))

	# print 'normal shape', normal.shape
	# print 'rho shape', rho.shape, rho
	
	return rho, normal