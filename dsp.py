import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt

SR = 16000
HOP_LENGTH = 512

def load_file(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    return y, sr

def transform(y, sr):
    # applies cqt transform to the audio signal and returns 1D array of complex numbers representing the frequency content of the signal
    result = librosa.cqt(y=y, sr=sr, hop_length=HOP_LENGTH, fmin=librosa.note_to_hz('A0'), n_bins=88, bins_per_octave=12)
    return result

def get_loudness(ind_freq):
    # gets the loudness of the audio signal by converting the complex numbers to decibels
    abs_ind_freq = np.abs(ind_freq)
    loudness = librosa.amplitude_to_db(abs_ind_freq, ref=np.max)
    return loudness

def main():
    
    parser = argparse.ArgumentParser(description="Audio Processing Script")
    parser.add_argument("input_file", type=str, help="Path to the input audio file (.mp3 or .wav)")

    args = parser.parse_args()
    in_file = args.input_file

    print(f"Loading {in_file}...")
    y, sr = load_file(in_file)

    print("Running Constant-Q Transform...")
    ind_freq = transform(y, sr)

    print("Converting to Decibels...")
    loudness = get_loudness(ind_freq)

    print("Generating Plot...")

    fig, ax = plt.subplots(figsize=(10, 6))
    img_loud = librosa.display.specshow(loudness, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', ax=ax)
    
    fig.colorbar(img_loud, ax=ax, format="%+2.0f dB")
    plt.title("Constant-Q Transform (Spectrogram)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()