from abc import ABC, abstractmethod
import numpy as np

class Transcriber(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def transcribe(self, audio_data: np.ndarray) -> str:
        pass
