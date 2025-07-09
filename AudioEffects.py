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
        self.p = pyaudio.PyAudio()
        self.stream = None
        

      