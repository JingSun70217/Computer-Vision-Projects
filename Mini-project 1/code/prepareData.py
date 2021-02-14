import numpy as np

def prepareData(imArray, ambientImage):
	new_im = imArray.copy()
	
	for i in range(imArray.shape[2]):
		new_im[:,:,i] = imArray[:,:,i] - ambientImage;
	new_im[new_im<0] = 0
	# max_in = np.amax(new_im)
	# print 'max value of imArray: ',max_in
	new_im = new_im/255
	return new_im
	
