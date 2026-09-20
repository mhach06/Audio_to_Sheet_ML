import numpy as np
import pretty_midi
import music21
import torch
from model import PianoTranscriptionModel
from dsp import load_file, transform, get_loudness

def transcribe_song(audio_path, model_path="piano_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PianoTranscriptionModel().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Processing {audio_path}...")
    audio_data, sr = load_file(audio_path)
    ind_freq = transform(audio_data, sr)

    loudness_matrix = get_loudness(ind_freq)

    X_tensor = torch.tensor(loudness_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    print("Running model inference...")
    with torch.no_grad():
        output_tensor = model(X_tensor)

    probability_matrix = output_tensor.squeeze(0).cpu().numpy()

    print("Converting probabilities to MIDI notes...")
    completed_notes = build_midi_notes(probability_matrix)

    print(f"Transcription completed. {len(completed_notes)} notes to output_transcription.mid")

def build_midi_notes(probability_matrix):
    binary_matrix = (probability_matrix > 0.5).astype(int)

    tracker = np.full(88, -1)

    completed_notes = []

    # iterate through each frame and note to track the start and end of notes
    for frame in range(binary_matrix.shape[1]):
        for note in range(88):
            if binary_matrix[note, frame] == 1:
                if tracker[note] == -1:
                    tracker[note] = frame
            else:
                if tracker[note] != -1:
                    completed_notes.append((note + 21, tracker[note], frame))
                    tracker[note] = -1

    # hanging notes at the end of the sequence
    for note in range(88):
        if tracker[note] != -1:
            completed_notes.append((note + 21, tracker[note], binary_matrix.shape[1]))

    # convert to MIDI FILE
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    frame_duration = HOP_LENGTH / SR  # duration of each frame in seconds
    # set each note in midi file
    for note, start_frame, end_frame in completed_notes:
        start_time = start_frame * frame_duration
        end_time = end_frame * frame_duration

        # create the note (velocity at 100 temporarily)
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


if __name__ == "__main__":
    transcribe_song("test_piano_audio.wav")