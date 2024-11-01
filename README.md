# Vehicle Number Plate Detection

A project to detect vehicle number plates using computer vision techniques. This pipeline employs traditional image processing, selective search, and custom filtering to identify regions of interest in an image.

## Features
- Detects license plates in various orientations (horizontal and rotated)
- Filters candidates based on aspect ratio, color, and edge density
- Handles images taken from different angles for robustness

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Challenges and Limitations](#challenges-and-limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/deepaksilaych/number-plate-detection.git
    cd number-plate-detection
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the main script to detect number plates in a sample image:
    ```bash
    python src/main.py --image_path path_to_your_image.jpg
    ```

2. For interactive exploration, open the Jupyter notebook:
    ```bash
    jupyter notebook notebooks/Number_Plate_Detection.ipynb
    ```

## Methodology

This pipeline consists of several steps to process images and detect license plates effectively:

1. **Preprocessing**:
    - Converts images to grayscale.
    - Resizes and applies Gaussian blur for noise reduction.

2. **Selective Search**:
    - Proposes candidate regions that might contain license plates using selective search.

3. **Candidate Filtering**:
    - **Aspect Ratio and Area Filtering**: Filters regions based on aspect ratios and sizes typical for license plates.
    - **Color Filtering**: In HSV color space, filters for regions predominantly white or yellow (common license plate backgrounds).
    - **Edge Density Filtering**: Retains regions with moderate edge density, often indicative of text-heavy regions like license plates.

4. **Rotation Handling**:
    - The image is processed at multiple angles to handle rotated plates.
    - Detected regions are transformed back to the original orientation.

## Results

The detection pipeline has shown effective results in detecting both horizontally aligned and rotated license plates. 
<!-- Below are some sample outputs:

![Example 1](data/example_output1.jpg)
![Example 2](data/example_output2.jpg) -->

## Challenges and Limitations

- **False Positives**: Similar-looking regions may be mistakenly identified as license plates.
- **Lighting Conditions**: Varying lighting or shadow conditions can impact detection accuracy.
- **Rotation and Filtering**: The rotation-based approach can be computationally expensive.

## Future Improvements

- **Integrate OCR**: Implement optical character recognition (OCR) to read text on detected plates.
- **Real-Time Processing**: Optimize the pipeline to allow for real-time processing.
- **Reduce False Positives**: Consider machine learning models or deep learning-based detection to improve accuracy and reduce false positives.

<!-- ## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->
