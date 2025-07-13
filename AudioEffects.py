import pyaudio
import numpy as np 
import librosa
import psola
from collections import deque
import threading
import time

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

        self.input_buffer = deque(maxlen=sample_rate * 2)  
        self.output_buffer = deque(maxlen=sample_rate * 2)

        self.is_processing = False
        self.autotune_enabled = False
        self.processing_thread = None
        self.correction_strength = 1.0

    def closest_pitch(self, f0):
        midi_note = np.around(librosa.hz_to_midi(f0))
        midi_note[np.isnan(f0)] = np.nan
        return librosa.midi_to_hz(midi_note)
    
    def apply_autotune(self, audio):
        f0, voiced_flag, voiced_probabilities = librosa.pyin(
                audio,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                sr=self.sample_rate,
                fmin=self.fmin,
                fmax=self.fmax
            )
        corrected_f0 = self.closest_pitch(f0)

        corrected_audio = psola.vocode(
                audio, 
                sample_rate=int(self.sample_rate), 
                target_pitch=corrected_f0, 
                fmin=self.fmin, 
                fmax=self.fmax
            )
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            print(f"Audio status: {status}")
        audio_input = np.frombuffer(in_data, dtype=np.float32)
        self.input_buffer.extend(audio_input)
        output_data = np.zeros(frame_count, dtype=np.float32)
        
        if len(self.output_buffer) >= frame_count:
            for i in range(frame_count):
                if self.output_buffer:
                    output_data[i] = self.output_buffer.popleft()
        else:
            output_data = audio_input
        
        return (output_data.tobytes(), pyaudio.paContinue)
    
    def processing_worker(self):
        while self.is_processing:
                if len(self.input_buffer) >= self.chunk_size:
                    chunk = np.array([self.input_buffer.popleft() for _ in range(min(self.chunk_size, len(self.input_buffer)))])
                    processed_chunk = self.apply_autotune(chunk)
                    self.output_buffer.extend(processed_chunk)
                    while len(self.output_buffer) > self.sample_rate:  
                        self.output_buffer.popleft()
    
    def start_processing(self):
        if self.is_processing:
            print("Already processing audio")
            return
        
        try:
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self.processing_worker)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                output=True,
                frames_per_buffer=self.chunk_size // 4,  # Smaller buffer for lower latency
                stream_callback=self.audio_callback,
                start=False
            )
            
            self.stream.start_stream()
            print("Audio processing started")
            
        except Exception as e:
            print(f"Error starting audio processing: {e}")
            self.is_processing = False
    
    def stop_processing(self):
        if not self.is_processing:
            return
        
        self.is_processing = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            
        print("Audio processing stopped")
    
    def set_autotune_enabled(self, enabled):
        self.autotune_enabled = enabled
        status = "enabled" if enabled else "disabled"
        print(f"Autotune {status}")
    
    def set_correction_strength(self, strength):
        self.correction_strength = max(0, min(1, strength))
        print(f"Correction strength set to {self.correction_strength}")
    
    def __del__(self):
        if hasattr(self, 'p'):
            self.p.terminate()

class HandGestureAudioController:
    def __init__(self):
        self.autotune_processor = AudioEffects(
            sample_rate=44100,
            chunk_size=4096  # Larger chunk for better quality processing
        )
        
    def start(self):
        self.autotune_processor.start_processing()
        
    def stop(self):
        self.autotune_processor.stop_processing()
        
    def on_gesture_detected(self, gesture):
        if gesture == 'fist':
            self.autotune_processor.set_autotune_enabled(True)
        elif gesture == 'open palm':
            self.autotune_processor.set_autotune_enabled(False)
