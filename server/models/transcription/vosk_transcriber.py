import numpy as np
import json

from vosk import KaldiRecognizer, Model

from server.models.transcription.transcriber import Transcriber


class VoskTranscriber(Transcriber):
    def __init__(self, model_name: str = "en-us"):
        print(f"Initializing transcription pipeline with model: {model_name}...")
        self.model = Model(lang=model_name)

        

    def transcribe(self, data: np.ndarray, state) -> str:
        sample_rate, audio_data = data
        audio_data = audio_data.astype("int16").tobytes()


        if state is None:
            rec = KaldiRecognizer(self.model, sample_rate)
            result = []
        else:
            rec, result = state

        if rec.AcceptWaveform(audio_data):
            text_result = json.loads(rec.Result())["text"]
            if text_result != "":
                result.append(text_result)
            partial_result = ""
        else:
            partial_result = json.loads(rec.PartialResult())["partial"] + " "

        # , (rec, result)

        return "\n".join(result) + "\n" + partial_result




