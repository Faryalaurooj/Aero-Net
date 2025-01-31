# config.py

import os

# Model hyperparameters
class Config:
    # Image size and channels
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    CHANNELS = 3

    # Model parameters
    N_LAYERS = 5  # Number of convolutional layers
    N_SCALES = 3  # Number of scales for multi-scale attention

    # Batch size and learning rate
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 50

    # Dataset paths
    TRAIN_DATA_PATH = './data/train/'
    VALIDATION_DATA_PATH = './data/validation/'

    # Pretrained model path (if applicable)
    PRETRAINED_PATH = None  # Set None if no pre-trained model is being used

    # Output directories
    OUTPUT_DIR = './output/'
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints/')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs/')

    # Loss weights
    LAMBDA_LOC = 1.0
    LAMBDA_CLS = 1.0
    LAMBDA_FC = 1.0

    # CUDA settings (if using GPUs)
    USE_CUDA = True


