import pyaudio
import numpy as np 
import librosa
import psola
from collections import deque
import threading
import os
import wave

class AudioEffects:

    def __init__(self, sample_rate = 44100, chunk_size = 4096, format=pyaudio.paFloat32):
        self.sample_rate = sample_rate
        self.chunk_size =  max(chunk_size, 8192)
        self.format = format
        self.channels = 1

        #pyaudio initializations
        self.p = pyaudio.PyAudio()
        self.stream = None

        # librosa param
        self.frame_length = 4096
        self.hop_length = self.frame_length // 4
        self.fmin = librosa.note_to_hz('C2')
        self.fmax = librosa.note_to_hz('C7')

        self.input_buffer = deque()
        self.output_buffer = deque()

        self.autotune_enabled = False
        self.is_processing = False
        self.processing_thread = None
        self.correction_strength = 0.9

        self.voice_layering_enabled = False
        self.processed_audio = []
        
    def closest_pitch(self, f0):
        midi_note = np.around(librosa.hz_to_midi(f0))
        midi_note[np.isnan(f0)] = np.nan
        return librosa.midi_to_hz(midi_note)
    
    def apply_autotune(self, audio):
        if not self.autotune_enabled:
            return audio
        try:
            if len(audio) < self.frame_length:
                return audio
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            f0, _, _ = librosa.pyin(
                    audio,
                    frame_length=self.frame_length,
                    hop_length=int(self.hop_length // 1.5),
                    sr=self.sample_rate,
                    fmin=self.fmin,
                    fmax=self.fmax
                )
            corrected_f0 = self.closest_pitch(f0)
            
            if np.all(np.isnan(corrected_f0)):
                return audio

            corrected_audio = psola.vocode(
                    audio, 
                    sample_rate=int(self.sample_rate), 
                    target_pitch=corrected_f0, 
                    fmin=self.fmin, fmax=self.fmax
                )
            return (self.correction_strength * corrected_audio + 
                        (1 - self.correction_strength) * audio)
                    
        except Exception as e:
                    print(f"Exception: {e}")
                    return audio

    def apply_voice_layering(self, audio, num_layers = 3):
        if not self.voice_layering_enabled:
            return audio
        layered = audio.copy()
        for i in range(1, num_layers + 1):
            pitch_shift = i * 0.5 
            shifted = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=pitch_shift)
            delay_samples = i * 200
            delayed = np.pad(shifted, (delay_samples, 0), mode='constant')[:len(audio)]
            layered += 0.2 * delayed / (i + 1)
        return layered
    
    def processing_worker(self):
        while self.is_processing:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.float32)
                processed = self.apply_autotune(audio)
                processed = self.apply_voice_layering(processed) 
                self.processed_audio.extend(processed)

    def start_processing(self):
        if self.is_processing:
            return
        self.is_processing = True
        self.processed_audio = []
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            start=True
        )

        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()
        print("Processing started")
            
    def stop_processing(self):
        if not self.is_processing:
            return
        
        self.is_processing = False
        self.processing_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
        print("Recording stopped.")
    
    def set_autotune_enabled(self, enabled):
        self.autotune_enabled = enabled
        status = "enabled" if enabled else "disabled"
        print(f"Autotune {status}")
    
    def set_voice_layering_enabled(self, enabled):
        self.voice_layering_enabled = enabled
        print(f"Layering {'ON' if enabled else 'OFF'}")

    def save_audio(self, filename=None):
        if not self.processed_audio:
            print("No audio to save")
            return None
                
        if filename is None:
            filename = f"processed_audio.wav"
            
        os.makedirs("output", exist_ok=True)
        filepath = os.path.join("output", filename)
            
        audio_array = np.array(self.processed_audio)
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array)) * 0.8
        audio_int16 = (audio_array * 32767).astype(np.int16)
            
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
            
        print(f"Audio saved to: {filepath}")
        return filepath
        
    def __del__(self):
        if hasattr(self, 'p'):
            self.p.terminate()

class HandGestureAudioController:
    def __init__(self):
        self.autotune_processor = AudioEffects(
            sample_rate=44100,
            chunk_size=4096  
        )
        
    def start(self):
        self.autotune_processor.start_processing()
        
    def stop(self):
        self.autotune_processor.stop_processing()

    def save_audio(self, filename=None):
        return self.autotune_processor.save_audio(filename)
        
    def on_gesture_detected(self, gesture):
        if gesture == 'fist':
            self.autotune_processor.set_autotune_enabled(True)
        elif gesture == 'open palm':
            self.autotune_processor.set_autotune_enabled(False)

