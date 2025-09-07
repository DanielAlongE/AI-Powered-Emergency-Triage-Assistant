import gradio as gr
from models.transcription.whisper_transcriber import WhisperTranscriber


def transcribe(audio):
    transcriber = WhisperTranscriber()
    return transcriber.transcribe(audio)


demo = gr.Interface(
    transcribe,
    gr.Audio(),
    "text",
)

demo.launch(inbrowser=True)