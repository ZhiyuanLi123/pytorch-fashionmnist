PyTorch FashionMNIST Classifier

This repository contains a simple PyTorch project for training and evaluating a neural network on the FashionMNIST dataset.
The goal of this project is to practice the basic PyTorch workflow, including data loading, batching with DataLoader, model training, and evaluation.

Project Overview

This project demonstrates:

Loading datasets using torchvision.datasets

Using torch.utils.data.Dataset and DataLoader

Training a neural network in PyTorch

Evaluating model performance on a test set

It is intended for learning and educational purposes.

Project Structure
pytorch-fashionmnist/
├── train.py            # Main training and testing script
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation

Dataset

The project uses the FashionMNIST dataset provided by torchvision.

60,000 training images

10,000 test images

Image size: 28 × 28 (grayscale)

10 clothing categories

The dataset will be automatically downloaded when the script is run for the first time.

Environment Setup

It is recommended to use a virtual environment.

Install dependencies with:

pip install -r requirements.txt


Dependencies:

torch

torchvision

How to Run

Run the training script:

python train.py


The script will:

Load the FashionMNIST dataset

Create DataLoaders with batching

Train the model

Evaluate the model on the test dataset

DataLoader Output Format

Each batch returned by the DataLoader has the following shapes:

Input images X: [batch_size, 1, 28, 28]

Labels y: [batch_size]

Example output:

Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64])

Notes

Trained model files (.pth) are not included in this repository.

The dataset files are downloaded automatically and are not tracked by Git.

This project focuses on clarity and learning rather than model performance.

References

PyTorch Documentation
https://pytorch.org/docs/stable/index.html

FashionMNIST Dataset
https://github.com/zalandoresearch/fashion-mnist