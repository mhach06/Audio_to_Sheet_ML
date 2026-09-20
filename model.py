import torch
import torch.nn as nn
import torch.nn.functional as F

class PianoTranscriptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.activation = nn.ReLU() 

        # LazyLinear automatically calculates the correct 'in_features' math 
        # based on the tensor shape during the very first training batch.
        self.fc_pitch = nn.LazyLinear(out_features=88)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Insert the missing channel dimension (batch, 1_channel, freq, time)
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.activation(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor for the linear layer
        pitch_probabilities = self.sigmoid(self.fc_pitch(x))

        return pitch_probabilities