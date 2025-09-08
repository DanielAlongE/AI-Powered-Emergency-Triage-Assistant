import numpy as np
from transformers import pipeline

from server.models.transcription.transcriber import Transcriber

class WhisperTranscriber(Transcriber):
    def __init__(self, model_name: str = "openai/whisper-tiny.en", device: str = "mps"):
        print(f"Initializing transcription pipeline with model: {model_name}...")
        
        try:        
            self.transcriber = pipeline("automatic-speech-recognition", model=model_name, device = device)
        except Exception as e:
            print(f"Error initializing the pipeline: {e}")
            self.transcriber = None


    def transcribe(self, audio_data: np.ndarray) -> str:

        if self.transcriber is None:
            return "Transcription failed due to model initialization error."
        
        try:
            sr, y = audio_data
            y = y.astype(np.float32)
            if y.ndim == 2:  # Check if there are two channels
                y = np.mean(y, axis=1)  # Convert to mono by taking the mean of the two channels
            y /= np.max(np.abs(y))

            result = self.transcriber({"sampling_rate":sr, "raw":y})
            return result.get("text", "No text found in transcription.")
        except Exception as e:
            print(f"Error during transcription: {e}")
            return "Transcription failed."



