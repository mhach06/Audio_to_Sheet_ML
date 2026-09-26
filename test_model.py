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
    # 1. Create an array of 88 thresholds
    thresholds = np.zeros((88, 1))
    
    # Bass notes (indices 0 to 43): Slightly lower to catch all repeated strikes
    thresholds[:44, 0] = 0.60 
    
    # Treble notes (indices 44 to 87): Much lower to rescue the right-hand melody
    thresholds[44:, 0] = 0.30 

    # Apply the thresholds element-wise across the matrix
    binary_matrix = (probability_matrix > thresholds).astype(int)

    tracker = np.full(88, -1)
    last_off_frame = np.full(88, -1)  
    completed_notes = []

    # 2. Post-processing rules
    MAX_GAP_FRAMES = 0   
    MIN_NOTE_FRAMES = 4  

    # 3. Iterate through frames to track note states
    for frame in range(binary_matrix.shape[1]):
        for note in range(88):
            if binary_matrix[note, frame] == 1:
                if tracker[note] == -1:
                    if last_off_frame[note] != -1 and (frame - last_off_frame[note]) <= MAX_GAP_FRAMES:
                        for i in reversed(range(len(completed_notes))):
                            if completed_notes[i][0] == note + 21 and completed_notes[i][2] == last_off_frame[note]:
                                tracker[note] = completed_notes[i][1] 
                                completed_notes.pop(i) 
                                break
                        if tracker[note] == -1: 
                            tracker[note] = frame
                    else:
                        tracker[note] = frame
            else:
                if tracker[note] != -1:
                    duration = frame - tracker[note]
                    if duration >= MIN_NOTE_FRAMES:
                        completed_notes.append((note + 21, tracker[note], frame))
                        last_off_frame[note] = frame 
                    tracker[note] = -1

    for note in range(88):
        if tracker[note] != -1:
            duration = binary_matrix.shape[1] - tracker[note]
            if duration >= MIN_NOTE_FRAMES:
                completed_notes.append((note + 21, tracker[note], binary_matrix.shape[1]))

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