import torch
import os
import random
import numpy as np
import h5py
import note_seq
from torch.utils.data import Dataset
from dsp import transform, get_loudness, SR, HOP_LENGTH

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)

class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.fps = SR / HOP_LENGTH
        self.samples = []

        h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        for h5_name in h5_files:
            h5_path = os.path.join(data_dir, h5_name)
            with h5py.File(h5_path, 'r') as f:
                for song_key in f.keys():
                    if 'audio' in f[song_key] and 'midi' in f[song_key]:
                        self.samples.append((h5_path, song_key))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        h5_path, song_key = self.samples[index]

        with h5py.File(h5_path, 'r') as f:
            # 1. Load raw audio and MIDI bytes
            audio_int16 = f[f'{song_key}/audio'][()]
            audio_data = int16_to_float32(audio_int16)
            
            midi_string = f[f'{song_key}/midi'][()].tobytes()
            
        # 2. Decompress MIDI and generate Piano Roll
        ns = note_seq.NoteSequence.FromString(midi_string)
        pm = note_seq.note_sequence_to_pretty_midi(ns)

        Y_full = pm.get_piano_roll(fs=self.fps) 
        Y = Y_full[21:109, :]  # 88 piano keys

        # 3. Dynamic Chunking (BEFORE heavy DSP)
        chunk_size = 100
        total_frames = Y.shape[1] 
        start_col = 0 if total_frames <= chunk_size else random.randint(0, total_frames - chunk_size)
        
        # Slice MIDI labels
        Y_chunk = np.zeros((88, chunk_size))
        y_end = min(start_col + chunk_size, total_frames)
        copy_len = max(0, y_end - start_col)
        if copy_len > 0:
            Y_chunk[:, :copy_len] = Y[:, start_col:y_end]
            
        # Convert frame indices to raw audio sample indices
        start_sample = start_col * HOP_LENGTH
        # Add 1 extra hop length as a buffer to guarantee the DSP generates enough frames
        end_sample = start_sample + ((chunk_size + 1) * HOP_LENGTH) 
        
        audio_chunk = audio_data[start_sample:end_sample]
        
        # 4. Transform ONLY the tiny 3.2-second chunk
        ind_freq = transform(audio_chunk, SR)
        X = get_loudness(ind_freq)
        
        # Force X to exactly match the chunk_size dimension
        X_chunk = np.zeros((X.shape[0], chunk_size))
        x_copy_len = min(chunk_size, X.shape[1])
        if x_copy_len > 0:
            X_chunk[:, :x_copy_len] = X[:, :x_copy_len]

        # 5. Convert to tensors
        X_tensor = torch.tensor(X_chunk, dtype=torch.float32)
        Y_tensor = (torch.tensor(Y_chunk, dtype=torch.float32) > 0).float()

        return X_tensor, Y_tensor