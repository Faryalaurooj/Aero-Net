import torch
import torch.nn as nn
import torch.nn.functional as F

class FighterNetModel(nn.Module):
    def __init__(self, model_size='medium'):
        super(FighterNetModel, self).__init__()
        
        self.model_size = model_size

        # Backbone configuration based on model size
        if self.model_size == 'small':
            self.conv_layers = 3  # Less convolutional layers
            self.num_filters = 32  # Fewer filters
            self.transformer_layers = 4  # Fewer transformer layers
        elif self.model_size == 'medium':
            self.conv_layers = 5
            self.num_filters = 64
            self.transformer_layers = 6
        elif self.model_size == 'large':
            self.conv_layers = 7  # More convolutional layers
            self.num_filters = 128  # More filters
            self.transformer_layers = 8  # More transformer layers
        else:
            raise ValueError("Unknown model size. Choose from 'small', 'medium', or 'large'.")

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, self.num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters * 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_filters * 2, self.num_filters * 4, kernel_size=3, stride=1, padding=1)
        
        # Add transformer encoder layers based on model size
        self.transformer_layers = nn.ModuleList([TransformerEncoderLayer(self.num_filters * 4) for _ in range(self.transformer_layers)])

    def forward(self, x):
        # Pass through the convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output for the transformer layers
        x = x.view(x.size(0), -1)

        # Pass through the transformer layers
        for transformer in self.transformer_layers:
            x = transformer(x)

        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(input_dim, num_heads=4)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim)
        )

    def forward(self, x):
        # Attention layer
        x, _ = self.attn(x, x, x)
        x = x + x  # Add residual connection

        # Feed-forward layer
        x = self.ffn(x)
        return x

