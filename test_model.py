import numpy as np
import pretty_midi
import music21
import torch
from model import PianoTranscriptionModel
from dsp import load_file, transform, get_loudness, HOP_LENGTH, SR

def transcribe_song(audio_path, model_path="piano_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PianoTranscriptionModel().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Processing {audio_path}...")
    audio_data, sr = load_file(audio_path)
    ind_freq = transform(audio_data, sr)

    loudness_matrix = get_loudness(ind_freq)

    X_tensor = torch.tensor(loudness_matrix, dtype=torch.float32).unsqueeze(0).to(device)

    print("Running model inference...")
    with torch.no_grad():
        output_tensor = model(X_tensor)

    probability_matrix = output_tensor.squeeze(0).cpu().numpy()

    print("Converting probabilities to MIDI notes...")
    completed_notes = build_midi_notes(probability_matrix)

    print(f"Transcription completed. {len(completed_notes)} notes to output_transcription.mid")

def build_midi_notes(probability_matrix, sr=16000, hop_length=512):
    tracker = np.full(88, -1)
    completed_notes = []

    start_thresholds = np.zeros((88, 1))
    end_thresholds = np.zeros((88, 1))

    # Define thresholds per octave (Octaves 0 through 8)
    # Bass requires high confidence, treble is highly forgiving.
    octave_starts = [0.70, 0.70, 0.65, 0.65, 0.60, 0.55, 0.50, 0.45, 0.45]
    octave_ends   = [0.45, 0.45, 0.40, 0.40, 0.35, 0.30, 0.25, 0.20, 0.20]

    # Octave 0 (Keys 0-2: A0, A#0, B0)
    start_thresholds[0:3, 0] = octave_starts[0]
    end_thresholds[0:3, 0]   = octave_ends[0]

    # Octaves 1 through 7 (Keys 3 to 86, 12 keys per octave)
    for oct_idx in range(1, 8):
        start_key = 3 + (oct_idx - 1) * 12
        end_key = start_key + 12
        start_thresholds[start_key:end_key, 0] = octave_starts[oct_idx]
        end_thresholds[start_key:end_key, 0]   = octave_ends[oct_idx]

    # Octave 8 (Key 87: C8)
    start_thresholds[87, 0] = octave_starts[8]
    end_thresholds[87, 0]   = octave_ends[8]

    MIN_NOTE_FRAMES = 5  

    for frame in range(probability_matrix.shape[1]):
        for note in range(88):
            prob = probability_matrix[note, frame]

            if tracker[note] == -1: 
                if prob > start_thresholds[note, 0]:
                    
                    # HARMONIC SUPPRESSION FILTER (Kept active to block overtones)
                    is_overtone = False
                    for interval in [12, 19]:
                        fundamental = note - interval
                        if fundamental >= 0 and tracker[fundamental] != -1:
                            if probability_matrix[fundamental, frame] > 0.70:
                                if prob < 0.75:
                                    is_overtone = True
                                    break
                                    
                    if not is_overtone:
                        tracker[note] = frame
            else: 
                if frame > 0 and (prob - probability_matrix[note, frame-1]) > 0.30:
                    duration = frame - tracker[note]
                    if duration >= MIN_NOTE_FRAMES:
                        completed_notes.append((note + 21, tracker[note], frame))
                    tracker[note] = frame 
                
                elif prob < end_thresholds[note, 0]:
                    duration = frame - tracker[note]
                    if duration >= MIN_NOTE_FRAMES:
                        completed_notes.append((note + 21, tracker[note], frame))
                    tracker[note] = -1

    for note in range(88):
        if tracker[note] != -1:
            duration = probability_matrix.shape[1] - tracker[note]
            if duration >= MIN_NOTE_FRAMES:
                completed_notes.append((note + 21, tracker[note], probability_matrix.shape[1]))

    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    frame_duration = hop_length / sr  

    for note, start_frame, end_frame in completed_notes:
        start_time = start_frame * frame_duration
        end_time = end_frame * frame_duration
        midi_note = pretty_midi.Note(velocity=100, pitch=note, start=start_time, end=end_time)
        piano.notes.append(midi_note)

    midi.instruments.append(piano)
    midi.write('output_transcription.mid')

    return completed_notes

def generate_sheet_music(midi_path):
    # 1. Parse the MIDI file
    # music21 automatically quantizes the milliseconds into musical durations
    # and attempts to separate the piano track into Treble and Bass clefs.
    score = music21.converter.parse(midi_path)
    
    # 2. Export the structural data as a MusicXML file
    score.write('musicxml', fp='output_sheet.xml')
    
    # 3. If you have notation software (like MuseScore) installed on your machine,
    # this command automatically renders and opens the visual sheet music.
    score.show()


import argparse

# ... (rest of your functions: transcribe_song, build_midi_notes, generate_sheet_music) ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe piano audio to MIDI.")
    parser.add_argument(
        "--audio", 
        type=str, 
        default="twinkle-twinkle-little-star-easy.wav", 
        help="Path to the input piano audio file."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="piano_model.pth", 
        help="Path to the trained model file."
    )
    args = parser.parse_args()

    transcribe_song(args.audio, args.model)