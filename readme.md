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