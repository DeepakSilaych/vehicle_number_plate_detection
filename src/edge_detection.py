# src/edge_detection.py

import numpy as np

def apply_sobel(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    grad_x = convolution(image, sobel_x)
    grad_y = convolution(image, sobel_y)
    
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return magnitude.astype(np.uint8)

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
