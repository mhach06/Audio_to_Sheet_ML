import torch
import os
from torch.utils.data import Dataset
import pretty_midi
from dsp import load_file, transform, get_loudness, SR, HOP_LENGTH

class AudioDataset(Dataset):

    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        self.fps = SR / HOP_LENGTH  # frames per second

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_path = os.path.join(self.audio_dir, self.audio_files[index])
        midi_path = audio_path.replace('.wav', '.mid')

        # handle audio file
        audio_data, sr = load_file(audio_path)
        ind_freq = transform(audio_data, sr)
        X = get_loudness(ind_freq)

        # handle midi file 
        pm = pretty_midi.PrettyMIDI(midi_path) # generates matrix of active notes perfectly synchronized with CQT frame rate

        Y_full = pm.get_piano_roll(fs=self.fps) # creates matrix of shape (128 rows, N time columns). Since each song dif N, make it fixed
        Y = Y_full[21:109, :]  # only keep the 88 piano keys

        chunk_size = 100 # 100 frames ~= 3.2 seconds
        total_frames = X.shape[1]

        # randomly select a starting column for the chunk
        start_col = random.randint(0, total_frames - chunk_size)

        # slice 2D matrices
        X_chunk = X[:, start_col:start_col + chunk_size]
        Y_chunk = Y[:, start_col:start_col + chunk_size]

        X_tensor = torch.tensor(X_chunk, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_chunk, dtype=torch.float32)

        # Y_tensor holds velocities. For probability, only need 0 or 1. So, convert to binary
        Y_tensor = (Y_tensor > 0).float()

        # returns X, which is fed to model, and Y, which is target output for the model (truth value)
        return X_tensor, Y_tensor