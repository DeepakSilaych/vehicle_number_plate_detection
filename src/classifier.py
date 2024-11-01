# src/classifier.py


import numpy as np
from .edge_detection import apply_sobel
from .contrast import linear_contrast_stretch

class NumberPlateClassifier:
    def __init__(self):
        self.templates = []

    def train(self, images):
        self.templates = [self.extract_features(img) for img in images]

    def classify(self, image):
        features = self.extract_features(image)
        distances = [np.linalg.norm(features - template) for template in self.templates]
        return np.argmin(distances)

    def extract_features(self, image):
        edges = apply_sobel(image)
        contrast = linear_contrast_stretch(image)
        combined_features = np.hstack((edges.flatten(), contrast.flatten()))
        return combined_features
