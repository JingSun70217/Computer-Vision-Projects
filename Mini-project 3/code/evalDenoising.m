% Entry code for evaluating demosaicing algorithms
% The code loops over all images and methods, computes the error and
% displays them in a table.
% 
%
% This code is part of:
%
%   CMPSCI 670: Computer Vision
%   University of Massachusetts, Amherst
%   Instructor: Subhransu Maji
%
% Load images
im = im2double(imread('../data/denoising/saturn.png'));
noise1 = im2double(imread('../data/denoising/saturn-noise1g.png'));
noise2 = im2double(imread('../data/denoising/saturn-noise1sp.png'));

% Compute errors
error1 = sum(sum((im - noise1).^2));
error2 = sum(sum((im - noise2).^2));
fprintf('Input, Errors: %.2f %.2f\n', error1, error2)

% Display the images
figure(1);
subplot(1,3,1); imshow(im); title('Input');
subplot(1,3,2); imshow(noise1); title(sprintf('SE %.2f', error1));
subplot(1,3,3); imshow(noise2); title(sprintf('SE %.2f', error2));

%% Denoising algorithm (Gaussian filtering)

%% Denoising algorithm (Median filtering)

%% Denoising alogirthm (Non-local means)