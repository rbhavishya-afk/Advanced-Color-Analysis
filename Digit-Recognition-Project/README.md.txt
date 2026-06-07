# Task 2: Handwritten Digit Recognition

## Project Summary
This project implements a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The trained system automatically extracts image features and maps them to numerical digit predictions.

## Technical Stack
- **Language:** Python 3
- **Deep Learning Framework:** TensorFlow / Keras
- **Data Manipulation & Visualization:** NumPy, Matplotlib
- **Metrics Analysis:** Scikit-Learn

## Architecture Metrics Summary
- **Network Design:** 2x Convolutional Layers + Max Pooling, 1x Dropout Regularization Layer, 1x Softmax Classification Dense Head.
- **Model Training Performance:** ~99% Target Accuracy achieved across 5 optimization epochs.

## File Descriptions
- `digit_recognition.py`: The single-file source code to train and evaluate the network.
- `saved_models/digit_recog_cnn.h5`: The exported weights artifact of the trained AI model.
- `training_performance_curves.png`: Visual metrics showcasing loss reduction and accuracy scaling.
- `sample_test_predictions.png`: Visual verification grid mapping model predictions against true labels.