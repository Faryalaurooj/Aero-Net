# utils.py

import torch
import torch.nn as nn
import os
import cv2
import numpy as np

# Helper functions for data loading and transformations

def load_image(image_path, height=512, width=512):
    """
    Loads and preprocesses an image.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (width, height))
    image = image / 255.0  # Normalize to [0, 1]
    return image

def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Saves the model checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """
    Loads the model checkpoint.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath}")
    return model, optimizer, epoch, loss

def calculate_metrics(predictions, ground_truth):
    """
    Calculates accuracy, precision, recall, F1 score.
    """
    # Example - implement your own metrics calculation here.
    accuracy = np.mean(predictions == ground_truth)
    return accuracy

