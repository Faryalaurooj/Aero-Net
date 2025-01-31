import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import FighterNetModel  # Assuming this is the model definition file
import os

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train Fighter-Net Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--model-size', type=str, choices=['small', 'medium', 'large'], default='medium', help='Model size: small, medium, or large')
    parser.add_argument('--save-dir', type=str, default='./trained_models', help='Directory to save trained models')
    return parser.parse_args()

# Load configuration (you should implement this function to read from config.yaml)
def load_config(config_path):
    # Placeholder for loading YAML config
    # Example:
    config = {
        'backbone': {'model_size': 'medium'},
        'optimizer': {'type': 'Adam', 'learning_rate': 0.001, 'weight_decay': 1e-5},
        'training': {'epochs': 50, 'batch_size': 32}
    }
    return config

# Training function
def train(model, dataloader, criterion, optimizer, epochs, device, save_dir):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # Print every 100 batches
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        # Save the model checkpoint after each epoch
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_save_path = os.path.join(save_dir, f'fighter_net_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    print("Training Finished.")

# Main function to train the model
def main():
    # Parse arguments
    args = parse_args()

    # Load the config
    config = load_config(args.config)

    # Set device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model with the specified size
    model = FighterNetModel(model_size=args.model_size).to(device)

    # Define the loss function (you can modify this to use your custom loss)
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=config['optimizer']['weight_decay'])

    # Data transformation (example, replace with your own dataset logic)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Placeholder dataset (replace with actual dataset)
    train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Train the model
    train(model, train_loader, criterion, optimizer, args.epochs, device, args.save_dir)

if __name__ == '__main__':
    main()

