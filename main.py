# src/main.py

import cv2
import os
import numpy as np
from src.contrast import linear_contrast_stretch
from src.edge_detection import apply_sobel
from src.filtering import apply_smoothing_filter, apply_high_pass_filter
from src.classifier import NumberPlateClassifier

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = linear_contrast_stretch(image)
    image = apply_smoothing_filter(image)
    edges = apply_sobel(image)
    return edges

def main():
    classifier = NumberPlateClassifier()
    
    # Load and preprocess training images
    training_image_paths = [f"data/train/{img}" for img in os.listdir("data/train")]
    preprocessed_training = [preprocess_image(img_path) for img_path in training_image_paths]
    classifier.train(preprocessed_training)
    
    # Test on a new image
    test_image_path = "data/test/Cars285.png"
    test_image = preprocess_image(test_image_path)
    result = classifier.classify(test_image)
    
    print("Detected number plate class:", result)

if __name__ == "__main__":
    main()
