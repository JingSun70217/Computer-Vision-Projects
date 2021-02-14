import numpy as np



def predError(ref, img, x, y, method):
	# preprocess image
	# np.pad(ref, (7,8), 'wrap')
	ref = np.roll(ref, x, axis=1)
	ref = np.roll(ref, y, axis=0)

	if method == 'ncc':
		ref = ref/np.linalg.norm(ref, 2)
		img = img/np.linalg.norm(img, 2)
		# error = np.dot(ref, img)
		error = np.sum(np.dot(ref, np.transpose(img)))

	if method == 'ssd':
		error = np.sum((ref - img)**2)
	
	return error

def alignChannels(img, max_shift, method):
	# raise NotImplementedError("You should implement this.")
	
	pred_shift = np.zeros((2,2), dtype = np.int8)
	low_err = 1e20
	for x1 in range(-max_shift[0], max_shift[0]+1):
		for y1 in range(-max_shift[1], max_shift[1]+1):
			# np.pad(new_img[:,:,0], (7,8), 'wrap')
			error = predError(img[:,:,0], img[:,:,1], x1, y1, method)
			if error < low_err:
				low_err = error
				pred_shift[0,:] = [x1,y1]

	low_err = 1e20
	for x2 in range(-max_shift[0], max_shift[0]+1):
		for y2 in range(-max_shift[1], max_shift[1]+1):
			# np.pad(new_img[:,:,2], (8,7), 'symmetric')
			# pred_shift[1] = predShift(img[:,:,2], img[:,:,1], x2, y2, method)
			error = predError(img[:,:,2], img[:,:,1], x2, y2, method)
			if error < low_err:
				low_err = error
				pred_shift[1,:] = [x2,y2]
	
	img[:,:,0] = np.roll(img[:,:,0], pred_shift[0][0], axis=1)
	img[:,:,0] = np.roll(img[:,:,0], pred_shift[0][1], axis=0)

	img[:,:,2] = np.roll(img[:,:,2], pred_shift[1][0], axis=1)
	img[:,:,2] = np.roll(img[:,:,2], pred_shift[1][1], axis=0)

	img = img[30:-15, 25:-15,:]
	return img, pred_shift



