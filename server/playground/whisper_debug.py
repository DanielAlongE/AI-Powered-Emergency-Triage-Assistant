import re
import gradio as gr
from models.transcription.whisper_transcriber import WhisperTranscriber


def transcribe(audio):
    transcriber = WhisperTranscriber()
    return transcriber.transcribe(audio)

def onchange(input):
    print(input)
    return gr.Markdown(markdown_wrapper(input))

def highlighted(text):
    return f'<span style="background-color: yellow;">{text}</span>'

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

    return f'<div style="min-height: 300px; width:100%; border: 1 solid #ccc;">{''.join(result)}</div>'

with gr.Blocks() as demo:
    with gr.Row():
        text = gr.Markdown(markdown_wrapper(''))
        input = gr.TextArea()
    with gr.Row():
        with gr.Column():
            clear = gr.ClearButton()
        with gr.Column():
            submit = gr.Button("Submit")
    input.change(onchange, inputs=input, outputs=text)

# demo = gr.Interface(
#     transcribe,
#     gr.Audio(),
#     "text",
# )

demo.launch(inbrowser=True)