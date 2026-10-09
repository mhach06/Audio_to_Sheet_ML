# Audio to Sheet ML 🎵 ➡️ 🎼

An end-to-end Machine Learning pipeline that transcribes raw audio recordings into readable sheet music and MIDI files.

---

## Overview

Automatic Music Transcription (AMT) converts acoustic signals into symbolic representations (MIDI, MusicXML, or PDF sheet music). This project processes raw audio files, detects pitch and note onset events using machine learning, and renders the transcribed output into playable notation.

---

## Project Files & Architecture

The codebase is highly modular, separating mathematical signal processing from machine learning logic:

- **`dsp.py`**: Handles all Digital Signal Processing. Converts raw `.wav` audio into 2D tensors (spectrograms) using frequency transforms and extracts features like loudness matrices.
- **`dataset.py`**: The data pipeline. Loads raw audio and corresponding ground-truth MIDI files (e.g., from the MAESTRO dataset) in batches, aligns them, and serves tensors to the GPU.
- **`model.py`**: Contains the neural network architecture (`PianoTranscriptionModel`), which uses 2D Convolutional layers to process spectrograms into note probabilities.
- **`train.py`**: The execution script for training. Handles forward passes, loss calculation, backpropagation, and saves the optimized weights (e.g., `piano_model.pth`).
- **`test_model.py`**: The inference and post-processing script. It loads the trained weights, runs an unseen `.wav` audio file through the model, translates the raw probability matrix into discrete `Note` objects, tracks duration, and serializes the result into a standard MIDI file (or MusicXML score).

---

## Tech Stack & Dependencies

- **Language:** Python 3.9+
- **Deep Learning:** PyTorch
- **Audio Processing:** Librosa, SoundFile, SciPy, NumPy
- **Music Notation & MIDI:** Music21, Pretty-MIDI

---

## Local Setup & Installation

### Install Dependencies

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/mhach06/Audio_to_Sheet_ML.git](https://github.com/mhach06/Audio_to_Sheet_ML.git)
cd Audio_to_Sheet_ML
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

*(Note: If using `music21`'s visual score rendering, you must install MuseScore or LilyPond on your machine).*

---

## Steps to Test the Model (Inference)

The following inference workflow is adapted from the Personal Projects - 2026 file.

### Step 1: Provide an Evaluation Audio Sample

Before running the inference script, ensure you have a short solo piano recording in `.wav` format ready (e.g., `twinkle-twinkle-little-star-easy.wav`). A 15-to-30-second clip without vocals or heavy drum tracks is ideal for testing.

### Step 2: Execute Inference (Testing)

Run the test script to process your evaluation audio. The script will load your saved `.pth` weights, run the audio through the network, and convert the frame-level predictions into discrete MIDI events.

```bash
python test_model.py --audio twinkle-twinkle-little-star-easy.wav --model piano_model.pth
```

*(Note: If you are using weights saved from a specific epoch, you can point to that specific file, like `piano_model_epoch24.pth`).*

### Step 3: View the MIDI Output

The script will generate a new file named `output_transcription.mid`. You can import this `.mid` file into MuseScore, Finale, or any standard Digital Audio Workstation (DAW) to visually verify the transcribed pitch accuracy and onset timing.

---

## Google Colab Setup (GPU Inference)

If you are running this in Google Colab to leverage free GPUs for faster inference, use these commands in your notebook cells:

**1. Clone and enter directory:**

```bash
!git clone [https://github.com/mhach06/Audio_to_Sheet_ML.git](https://github.com/mhach06/Audio_to_Sheet_ML.git)
%cd Audio_to_Sheet_ML
```

**2. Install dependencies:**

```bash
!pip install -q -r requirements.txt
!apt-get update -qq && apt-get install -y -qq musescore3
```

**3. Run Testing / Inference:**

```bash
!python test_model.py --audio test_song.wav --model piano_model.pth
```
