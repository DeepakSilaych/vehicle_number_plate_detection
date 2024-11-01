# src/filtering.py

import numpy as np

def apply_smoothing_filter(image, size=3):
    kernel = np.ones((size, size)) / (size * size)
    return convolution(image, kernel)

def apply_high_pass_filter(image):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return convolution(image, kernel)

def convolution(image, kernel):
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='constant')
    output = np.zeros_like(image)
    
    for i in range(pad, padded_image.shape[0] - pad):
        for j in range(pad, padded_image.shape[1] - pad):
            region = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            output[i - pad, j - pad] = np.sum(region * kernel)
    return output
