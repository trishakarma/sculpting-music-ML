# look into the following: pyaudio, librosa, sounddevice, crepe/pyin
# pyaudio for I/O; pyin to pitchdetect; pitch shift w librosa?
import pyaudio
import numpy as np 
import librosa

def closest_pitch(f0):
    midi_note = np.around(librosa.hz_to_midi(f0))
    nan_indices = np.isnan(f0)
    midi_note[nan_indices] = np.nan
    return librosa.midi_to_hz(midi_note)