import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

def rotate_image(image, angle):
    """Rotate the image by a given angle and return the rotated image."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated, matrix

def inverse_transform_box(x, y, w, h, matrix):
    """Transform the box coordinates back to the original orientation."""
    points = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    original_points = cv2.transform(np.array([points]), matrix)
    x_min, y_min = np.min(original_points[0], axis=0)
    x_max, y_max = np.max(original_points[0], axis=0)
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

def color_filter(roi, color="white"):
    """Filter ROIs based on color proportion."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if color == "white":
        lower_color = np.array([0, 0, 150], dtype=np.uint8)
        upper_color = np.array([180, 55, 255], dtype=np.uint8)
    elif color == "yellow":
        lower_color = np.array([15, 50, 150], dtype=np.uint8)
        upper_color = np.array([35, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    color_ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])
    return color_ratio > 0.4  # Filter if >40% of ROI matches target color

def edge_density_filter(roi):
    """Filter ROIs based on edge density."""
    edges = cv2.Canny(roi, 100, 200)
    edge_density = np.sum(edges) / (roi.shape[0] * roi.shape[1])
    return edge_density > 0.05  # Edge density threshold

# Load the image from a URL
image_url = 'https://c.ndtvimg.com/2018-10/a7sp8ji_high-security-number-plates_625x300_12_October_18.jpg'
response = requests.get(image_url)

if response.status_code == 200:
    image_data = np.frombuffer(response.content, np.uint8)
    original_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if original_image is None:
        print("Error: Could not decode the image.")
    else:
        angles = [-15, -10, -5, 0, 5, 10, 15]  # Rotation angles to check
        candidates = []  # Store candidates across all angles

        for angle in angles:
            # Rotate the image and store the transformation matrix
            rotated_image, matrix = rotate_image(original_image, angle)
            
            # Step 1: Convert rotated image to grayscale and resize
            gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
            scale_percent = 150  # Resize for better accuracy if needed
            width = int(gray.shape[1] * scale_percent / 100)
            height = int(gray.shape[0] * scale_percent / 100)
            gray = cv2.resize(gray, (width, height))
            image_resized = cv2.resize(rotated_image, (width, height))

            # Step 2: Apply Gaussian Blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Step 3: Enhance the image using adaptive thresholding
            enhanced = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)

            # Step 4: Morphological Transformations to connect regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morphed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

            # Step 5: Set up Selective Search for Region Proposals
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(image_resized)
            ss.switchToSelectiveSearchQuality()

            rects = ss.process()[:200]

            # Step 6: Filter regions based on size and aspect ratio to detect plate-like regions
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            for x, y, w, h in rects:
                aspect_ratio = w / h
                area = w * h
                if 2.0 < aspect_ratio < 6.0 and 4000 < area < 40000:
                    padding = 10
                    x, y = max(0, x - padding), max(0, y - padding)
                    w, h = min(width - x, w + 2 * padding), min(height - y, h + 2 * padding)
                    
                    # Convert the bounding box back to the original orientation
                    x_orig, y_orig, w_orig, h_orig = inverse_transform_box(x, y, w, h, cv2.invertAffineTransform(matrix))
                    
                    # Get the ROI in the original image
                    if 0 <= x_orig < original_image.shape[1] and 0 <= y_orig < original_image.shape[0] and \
                       x_orig + w_orig <= original_image.shape[1] and y_orig + h_orig <= original_image.shape[0]:
                        roi = original_image[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]

                        # Apply color and edge density filters
                        if color_filter(roi, "white") or color_filter(roi, "yellow"):
                            if edge_density_filter(roi):
                                candidates.append((x_orig, y_orig, w_orig, h_orig))

        # Draw all detected candidates on the original image
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        for (x, y, w, h) in candidates:
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

        plt.imshow(image_rgb)
        plt.title("Detected Plate Candidates with Color and Edge Filters")
        plt.show()

        # Display refined candidates as possible license plates
        for i, (x, y, w, h) in enumerate(candidates):
            roi = original_image[y:y + h, x:x + w]
            detected_plate_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            plt.imshow(detected_plate_rgb)
            plt.title(f"Detected License Plate Candidate {i + 1}")
            plt.show()
else:
    print("Error: Could not retrieve the image. Please check the URL and your internet connection.")
