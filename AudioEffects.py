# look into the following: pyaudio, librosa, sounddevice, crepe/pyin
# pyaudio for I/O; pyin to pitchdetect; pitch shift w librosa?
import pyaudio
import numpy as np 
import librosa

class AudioEffects:

    def __init__(self, sample_rate = 44100, chunk_size = 4096, format=pyaudio.paFloat32):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.format = format
        self.channels = 1

        #pyaudio initializations
        self.p = pyaudio.PyAudio()
        self.stream = None

        self.frame_length = 2048
        self.hop_length = self.frame_length // 4
        self.fmin = librosa.note_to_hz('C2')
        self.fmax = librosa.note_to_hz('C7')


    def closest_pitch(self, f0):
        midi_note = np.around(librosa.hz_to_midi(f0))
        midi_note[np.isnan(f0)] = np.nan
        return librosa.midi_to_hz(midi_note)
    
    def apply_autotune(self, audio):
        