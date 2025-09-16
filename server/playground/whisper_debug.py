import re
import gradio as gr
from models.transcription.whisper_transcriber import WhisperTranscriber
from models.transcription import VoskTranscriber
from models.llama.conversation_analizer import ConversationAnalizer

conversation = ConversationAnalizer()


def get_selected_transcriber(key: str):
    match key:
        case "whisper":
            return WhisperTranscriber()
        case "vosk":
            return VoskTranscriber()

def transcribe_stream(audio, state, transcription_model):
    transcriber = get_selected_transcriber(transcription_model)
    text, s = transcriber.transcribe_stream(audio, state)

    return text, s, gr.Markdown(markdown_wrapper(text))

def transcribe(audio, transcription_model):
    transcriber = get_selected_transcriber(transcription_model)
    text = transcriber.transcribe(audio)
    return gr.Markdown(markdown_wrapper(text))

def handle_markdown(text):
    return gr.Markdown(markdown_wrapper(text))

def handle_conversation(text):
        history = conversation.analyze(text)
        print('*' * 100)
        print(text)
        print(history)
        print('$' * 100)

        role = {'NURSE': 'assistant', 'PATIENT': 'user'}

        try:
            r = [{'content': chat['content'], 'role': role[chat['role']]} for chat in history['conversation'] if chat['content'] != ""]
            print(f"{r=}")
            return r
        except:
            return []

    

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
                    ["whisper", "vosk"], value="vosk", label="Transcriber"
                )

            audio = gr.Audio(sources=["microphone"], type="numpy", streaming=True)
            audio2 = gr.Audio(sources=["microphone", "upload"], type="numpy")

        with gr.Column():
            markdown_text = gr.Markdown(markdown_wrapper(''))
            chatbot = gr.Chatbot(type="messages")
            real_text = gr.Textbox(visible=False)
            real_text.change(handle_conversation, inputs=real_text, outputs=chatbot)
            # text.change(handle_conversation, inputs=text, outputs=chatbot)
        audio.stream(transcribe_stream, inputs=[audio, state, select], outputs=[real_text, state, markdown_text])
    # audio.input(transcribe, inputs=audio, outputs=text)
    with gr.Row():
        with gr.Column():
            clear = gr.ClearButton()
        with gr.Column():
            submit = gr.Button("Submit", variant='primary')
        submit.click(transcribe, inputs=[audio2, select], outputs=[markdown_text])
        clear.click(handle_clear, inputs=[audio, markdown_text])

demo.launch(inbrowser=True)