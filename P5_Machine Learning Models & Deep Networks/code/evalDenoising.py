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
import torch
import torch.optim as optim
import torch.nn as nn
from dip import EncDec
from utils import imread
import torch.nn.functional as F

# Load clean and noisy image
# im = imread('../data/denoising/saturn.png')
# noise1 = imread('../data/denoising/saturn-noisy.png')
im = imread('../data/denoising/lena.png')
noise1 = imread('../data/denoising/lena-noisy.png')

error1 = ((im - noise1)**2).sum()

print 'Noisy image SE: {:.2f}'.format(error1)

plt.figure(1)

plt.subplot(121)
plt.imshow(im, cmap='gray')
plt.title('Input')

plt.subplot(122)
plt.imshow(noise1, cmap='gray')
plt.title('Noisy image SE {:.2f}'.format(error1))

plt.show(block=False)


################################################################################
# Denoising algorithm (Deep Image Prior)
################################################################################

#Create network
net = EncDec()

# Loads noisy image and sets it to the appropriate shape
noisy_img = torch.FloatTensor(noise1).unsqueeze(0).unsqueeze(0).transpose(2, 3)
clean_img = torch.FloatTensor(im).unsqueeze(0).unsqueeze(0).transpose(2,3)
# Creates \eta (noisy input)
eta = torch.randn(*noisy_img.size())
eta.detach()


###
# Your training code goes here.
iterations = 800 
model = net
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# scheduler = optim.lr_scheduler.MultiStepLR(optimzxsxizer, milestones=[400, 700, 850], gamma=0.5)
error_test = torch.zeros(iterations)
error_train = torch.zeros(iterations)
output_opt = torch.zeros(*noisy_img.size())

for i in xrange(iterations):
    # scheduler.step()
        # data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(eta)
    loss = F.mse_loss(output, noisy_img)
    loss.backward()
    optimizer.step()
    # error_t = (torch.norm(output - noisy_img))**2
    error_train[i] = ((output - noisy_img)**2).sum()
    # print(error_t, error_t .rain[i])
    # assert(error_train[i]==error_t)
    # error_test[i] = (torch.norm(output - clean_img))**2
    error_test[i] = ((output - clean_img)**2).sum()
    
    if i >= iterations-150:
        output_opt += output
    if i % 50 == 0:
        print('training error: ({:.2f}), testing error: ({:.2f})\tLoss: {:.6f}'.format(error_train[i], error_test[i], loss.item()))
output_opt /= 150.0
error_train = error_train.detach().numpy()
error_test = error_test.detach().numpy()
plt.figure(2)
tr = plt.plot(xrange(iterations), error_train, 'r-', label='training error')
te = plt.plot(xrange(iterations), error_test, 'b-', label='testing error')
plt.legend()

# Shows final result
# out = net(eta)
out = output_opt
print(out.size())
out_img = out[0, 0, :, :].transpose(0,1).detach().numpy()

error1 = ((im - noise1)**2).sum()
error2 = ((im - out_img)**2).sum()

plt.figure(3)
plt.axis('off')

plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.title('Input')

plt.subplot(132)
plt.imshow(noise1, cmap='gray')
plt.title('SE {:.2f}'.format(error1))

plt.subplot(133)
plt.imshow(out_img, cmap='gray')
plt.title('SE {:.2f}'.format(error2))

plt.show()

