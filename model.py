import torch
import torch.nn as nn
import torch.nn.functional as F

class PianoTranscriptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Added padding=(1, 1) so the convolution doesn't shrink the 100 time frames to 98
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.activation = nn.ReLU() 

        self.fc_pitch = nn.LazyLinear(out_features=88)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Add channel dimension -> (batch, 1, freq, time)
        x = x.unsqueeze(1)
        
        # 2. Convolution -> (batch, 32, freq, time)
        x = self.conv1(x)
        x = self.activation(x)

        # 3. Reshape to preserve the time dimension
        b, c, f, t = x.size()
        x = x.view(b, c * f, t)  # Flatten only channels and frequency -> (batch, features, time)
        
        # 4. Transpose so time is the sequence -> (batch, time, features)
        x = x.transpose(1, 2)
        
        # 5. Predict 88 pitches for EACH time frame -> (batch, time, 88)
        x = self.fc_pitch(x)
        x = self.sigmoid(x)

        # 6. Transpose back to match Y's shape exactly -> (batch, 88, time)
        pitch_probabilities = x.transpose(1, 2)

        return pitch_probabilities