import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os
from models import FighterNetModel  # Assuming this is the model definition file

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Fighter-Net Model Prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml file')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image for prediction')
    parser.add_argument('--model-size', type=str, choices=['small', 'medium', 'large'], default='medium', help='Model size: small, medium, or large')
    parser.add_argument('--output-dir', type=str, default='predict', help='Directory to save prediction images')
    return parser.parse_args()

# Load configuration (you should implement this function to read from config.yaml)
def load_config(config_path):
    # Placeholder for loading YAML config
    # Example:
    config = {
        'backbone': {'model_size': 'medium'},
        'training': {'batch_size': 32}
    }
    return config

# Function to perform inference
def predict(model, image_path, transform, device):
    model.eval()  # Set model to evaluation mode
    image = Image.open(image_path).convert('RGB')  # Open and convert image to RGB
    image = transform(image).unsqueeze(0).to(device)  # Apply transformations and add batch dimension

    with torch.no_grad():  # Disable gradient computation for inference
        output = model(image)  # Perform the forward pass

    return output, image

# Function to save the image with prediction labels
def save_prediction(image, output, output_dir, image_name, predicted_class):
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert image to a drawable format
    image_with_label = image.copy()
    draw = ImageDraw.Draw(image_with_label)
    
    # Define label for the predicted class
    label = f"Predicted Class: {predicted_class.item()}"
    
    # Optionally: Add a simple font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Draw the label on the image
    draw.text((10, 10), label, fill="red", font=font)
    
    # Save the resulting image
    save_path = os.path.join(output_dir, f"predicted_{image_name}")
    image_with_label.save(save_path)
    print(f"Prediction saved to {save_path}")

# Main function to load the model and perform predictions
def main():
    # Parse arguments
    args = parse_args()

    # Load the config
    config = load_config(args.config)

    # Set device for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model with the specified size
    model = FighterNetModel(model_size=args.model_size).to(device)

    # Load the trained model
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
        print(f"Model loaded from {args.model_path}")
    else:
        print(f"Model file not found at {args.model_path}")
        return

    # Define the image transformation pipeline (modify as needed for your data)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get the input image name for saving
    image_name = os.path.basename(args.image_path)

    # Perform the prediction
    output, image = predict(model, args.image_path, transform, device)

    # Post-process the output (modify as needed based on your model's output)
    # Example: If it's classification, we might want to get the class with the highest probability
    _, predicted_class = torch.max(output, 1)
    print(f"Predicted class: {predicted_class.item()}")

    # Save the image with the predicted label
    save_prediction(image, output, args.output_dir, image_name, predicted_class)

if __name__ == '__main__':
    main()

