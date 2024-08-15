# LeNet-5 C++ Implementation

## Overview

This project is a handcrafted implementation of the LeNet-5 convolutional neural network architecture using C++. LeNet-5 is a classical CNN architecture originally designed for handwritten digit recognition (MNIST dataset). This implementation is created from scratch without relying on deep learning frameworks like TensorFlow or PyTorch, providing an in-depth understanding of how a CNN works at the fundamental level.

## Features

- **Custom Implementation**: All components, including convolutional layers, pooling layers, fully connected layers, and activation functions, are manually implemented in C++.
- **Modular Design**: The code is organized into modular classes for easy understanding and modification.
- **MNIST Dataset Support**: The implementation is designed to work with the MNIST dataset, which contains 60,000 training images and 10,000 testing images of handwritten digits.
- **Forward and Backward Propagation**: Both forward and backward passes are implemented, allowing the network to learn from the data via backpropagation and gradient descent.
- **Training and Evaluation**: The project includes functionality for training the network and evaluating its performance on test data.

## Architecture

The LeNet-5 architecture implemented in this project consists of the following layers:

1. **Input Layer**: 32x32 grayscale image.
2. **C1 - Convolutional Layer**: 6 feature maps, 5x5 kernel, followed by a sigmoid activation function.
3. **S2 - Subsampling Layer**: 6 feature maps, 2x2 average pooling.
4. **C3 - Convolutional Layer**: 16 feature maps, 5x5 kernel, followed by a sigmoid activation function.
5. **S4 - Subsampling Layer**: 16 feature maps, 2x2 average pooling.
6. **C5 - Fully Connected Convolutional Layer**: 120 nodes, followed by a sigmoid activation function.
7. **F6 - Fully Connected Layer**: 84 nodes, followed by a sigmoid activation function.
8. **Output Layer**: 10 nodes with softmax activation (for digit classification).


## How to run
### Transfer MNIST ubytes file to jpg
g++ ubytes_to_jpg.cpp
./a.out
### Adjust the file path in the main.cpp
### Run program
g++ main.cpp
./a.out
