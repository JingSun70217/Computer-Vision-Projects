import numpy as np

def getSurface(surfaceNormals, method):
	print '**********  method is ', method

	fx = surfaceNormals[:,:,0]/surfaceNormals[:,:,2]
	fy = surfaceNormals[:,:,1]/surfaceNormals[:,:,2]
	h = fx.shape[0]
	w = fx.shape[1]
	if method == 'column':
		
		fx = np.tile(np.cumsum(fx[0,:])[np.newaxis, :], (h,1))
		fy = np.cumsum(fy, axis=0)

		return fx + fy
	if method == 'row':
		
		# fy = np.tile(np.cumsum(fy[:, 1], axis=1),(1,w))
		fx = np.cumsum(fx, axis=1)
		fy = np.tile(np.cumsum(fy[:, 0])[:, np.newaxis], (1, w))
		
		return fx+fy
	if method == 'average':

		fx1 = np.tile(np.cumsum(fx[0,:])[np.newaxis, :], (h,1))
		fy1 = np.cumsum(fy, axis=0)

		fx2 = np.cumsum(fx, axis=1)
		fy2 = np.tile(np.cumsum(fy[:, 0])[:, np.newaxis], (1, w))

		return (fy1+fx1+fx2+fy2)/2
	if method == 'random':
		# raise NotImplementedError("You should implement this.")
		n = 50
		height = np.zeros((h,w), dtype = np.float)
		for i in range(n):
			for x in range(h):
				for y in range(w):
					# if x % 10 == 0 and y %
					height[x, y] += randomPath(fx, fy, x, y)

		return height/float(n)


def randomPath(fx, fy, x, y):

	xi = 0
	yi = 0
	pathsum = fx[0,0] + fy[0,0]
	for i in range(x+y):
		
		if xi < x and yi < y:
			direction = np.random.randint(2)
			
			if direction == 1:
				xi += 1
				pathsum += fy[xi,yi]
			else:
				yi += 1
				pathsum += fx[xi,yi]
		else:
			if xi >= x and yi < y: 
				xi = x
				yi += 1
				pathsum +=fx[xi, yi]
			elif yi >= y and xi < x:
				xi += 1
				yi = y
				pathsum += fy[xi, yi]
	return pathsum




