# src/contrast.py

import numpy as np

def linear_contrast_stretch(image):
    min_val, max_val = image.min(), image.max()
    stretched = (image - min_val) * (255 / (max_val - min_val))
    return stretched.astype(np.uint8)

def histogram_equalization(image):
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = histogram.cumsum()  # cumulative distribution function
    cdf_normalized = cdf * 255 / cdf[-1]
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    return equalized_image.reshape(image.shape).astype(np.uint8)
