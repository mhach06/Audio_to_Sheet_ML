import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PianoTranscriptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN with 32 kernels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.activation = nn.ReLU() # 

        self.fc_pitch = nn.Linear(in_features=..., out_features=88)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        pitch_probabilities = self.sigmoid(self.fc_pitch(x))

        return pitch_probabilities