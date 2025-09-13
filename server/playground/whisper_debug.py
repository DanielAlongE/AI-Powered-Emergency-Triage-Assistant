import re
import gradio as gr
from models.transcription.whisper_transcriber import WhisperTranscriber
from models.transcription import VoskTranscriber


def get_selected_transcriber(key: str):
    match key:
        case "whisper":
            return WhisperTranscriber()
        case "vosk":
            return VoskTranscriber()

def transcribe(audio, transcription_model, state):
    transcriber = get_selected_transcriber(transcription_model)
    text, s = transcriber.transcribe(audio, state)
    print(f"{transcription_model=}")
    return gr.Markdown(markdown_wrapper(text)), transcription_model, gr.State(s)

def highlighted(text):
    return f'<span style="background-color: yellow;">{text}</span>'

def handle_clear():
    return [gr.Markdown(markdown_wrapper('')), gr.Audio(sources=["microphone", "upload"], type="numpy")]

list_of_words = [
    'ache',
    'bleeding',
    'breathe',
    'breathing',
    'dizzy',
    'dizziness',
    'fever',
    'fracture',
    'headache',
    'pain',
    'trauma'
]

words_of_interest = set(list_of_words)

def markdown_wrapper(content):
    result = []

    pattern = r"(\w+|\W)"

    for word in re.split(pattern, content):
        if len(word) > 3 and word.lower() in words_of_interest:
            result.append(highlighted(word))
        elif word == "\n":
            result.append("<br />")
        else:
            print(f"__{word}__")
            result.append(word)

    return f'<div style="min-height: 300px; width:100%; border: 1px solid #ccc;">{''.join(result)}</div>'

with gr.Blocks() as demo:
    state = gr.State(None)
    with gr.Row():
        with gr.Column():
            select = gr.Dropdown(
                    ["whisper", "vosk"], value="whisper", label="Transcriber"
                )

            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", streaming=True)
        with gr.Column():
            text = gr.Markdown(markdown_wrapper(''))
    # audio.input(transcribe, inputs=audio, outputs=text)
    with gr.Row():
        with gr.Column():
            clear = gr.ClearButton()
        with gr.Column():
            submit = gr.Button("Submit", variant='primary')
        submit.click(transcribe, inputs=[audio, select, state], outputs=[text, select, state])
        clear.click(handle_clear, inputs=[audio, text])

demo.launch(inbrowser=True)