# Dog vs Cat Classification using Transfer Learning (MobileNetV2)

This project implements a deep learning model to classify images of dogs and cats using transfer learning with the MobileNetV2 architecture. The dataset is sourced from the Kaggle competition "Dogs vs Cats".

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction System](#prediction-system)
- [How to Run](#how-to-run)
- [Requirements](#requirements)

## Project Overview
The goal of this project is to classify images of dogs and cats using a pre-trained MobileNetV2 model. Transfer learning is used to leverage the pre-trained weights of MobileNetV2, which is fine-tuned for this binary classification task.

## Dataset
The dataset is downloaded from the Kaggle competition [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats). It contains images of dogs and cats in `.jpg` format.

### Steps to Download the Dataset:
1. Place your `kaggle.json` file in the appropriate directory.
2. Use the Kaggle API to download the dataset:
   ```bash
   !kaggle competitions download -c dogs-vs-cats
Preprocessing
Extract the dataset from the downloaded .zip files.
Resize all images to 224x224 pixels to match the input size required by MobileNetV2.
Normalize the pixel values to the range [0, 1].
Assign labels:
0 for cats
1 for dogs
Model Architecture
The model uses the MobileNetV2 architecture from TensorFlow Hub as a feature extractor. A dense layer with 2 output units is added for binary classification.

Key Components:
Pre-trained MobileNetV2: Used as a feature extractor with frozen weights.
Dense Layer: Added for classification.
Training
The model is trained using the following configuration:

Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metrics: Accuracy
Epochs: 5
Evaluation
The model is evaluated on a test set split from the dataset. The accuracy and loss are reported.

Prediction System
A predictive system is implemented to classify a single input image as either a dog or a cat. The user provides the path to the image, and the model predicts the label.

How to Run
Clone this repository and navigate to the project directory.
Ensure the dataset is downloaded and extracted.
Install the required dependencies (see Requirements).
Run the Jupyter Notebook Dog_vs_Cat_classification_using_Transfer_learning.ipynb to execute the code step by step.
Requirements
Python 3.x
TensorFlow 2.15.0
TensorFlow Hub
Keras 2.15.0
NumPy
Matplotlib
OpenCV
Kaggle API
Install the dependencies using:

Results
The model achieves satisfactory accuracy on the test set and can classify images of dogs and cats effectively.

Acknowledgments
Dataset: Kaggle Dogs vs Cats Competition
Pre-trained Model: MobileNetV2 on TensorFlow Hub
License
This project is licensed under the MIT License. See the LICENSE file for details.

