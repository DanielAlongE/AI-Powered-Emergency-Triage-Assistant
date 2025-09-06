from abc import ABC, abstractmethod
import numpy as np

class Transcription(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def transcribe(self, audio_data: np.ndarray) -> str:
        pass
