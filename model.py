# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FighterNet(nn.Module):
    def __init__(self):
        super(FighterNet, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Transformer encoder components (simplified version)
        self.msa = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.ffn = nn.Linear(256, 256)

    def forward(self, x):
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten and pass through transformer encoder
        x = x.flatten(2).permute(2, 0, 1)  # Reshape for multihead attention
        attn_output, _ = self.msa(x, x, x)
        x = self.ffn(attn_output)

        return x

