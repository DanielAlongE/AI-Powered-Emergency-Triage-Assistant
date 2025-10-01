# This script uses the Ollama library to analyze a conversation transcript
# and predict which parts of the dialogue are from the nurse and which are from the patient.
#
# Before running, ensure you have the Ollama server installed and running, and the llama3 model pulled:
# 1. Install Ollama from https://ollama.ai/
# 2. Open a terminal and run: ollama run llama3
# 3. Ensure the 'ollama' Python library is installed: pip install ollama
import json
import ollama
from openai import OpenAI
from config import get_settings


MODEL_LLAMA_3 = 'llama3.2'
MODEL_GEMMA_3 = 'gemma3:4b'
MODEL_GPT_OSS = 'gpt-oss:20b'
MODEL_GPT_4O = 'gpt-4o-mini'

RESPONSE_FORMAT = {
    'conversation': [
        {'role': 'assistant', 'content': 'I am the nurse!'},
        {'role': 'user', 'content': 'I am the patient.'},
        ]
    }

class ConversationAnalizer:
    def __init__(self, model='gemma3:4b'):
        self.model = model
        print(f"initialized ConversationAnalizer using {model}")

    def chat_with_ollama(self, prompt: str):
        args = {}
        if self.model == MODEL_LLAMA_3:
            args['format'] = "json"

        try:
            # Use the ollama.chat() method for a conversational approach
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}
                ],
                **args
            )

            # Extract the model's response
            analysis = response.message.content.strip()
            # cleanup the response from gpt-oss:20b
            analysis = analysis.replace('```json', '').replace('```', '')

            try:
                return json.loads(analysis)
            except json.JSONDecodeError:
                return {'conversation':[]}

        except Exception as e:
            print(f"An error occurred with Ollama: {e}")
            return "Analysis failed due to an error. Ensure Ollama is running and the model is available."

    def chat_with_openai(self, prompt: str):
        settings = get_settings()
        if not settings.openai_api_key:
            return "OpenAI API key not configured"

        try:
            client = OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}
                ],
                response_format={"type": "json_object"}
            )

            # Extract the model's response
            analysis = response.choices[0].message.content.strip()
            # cleanup the response
            analysis = analysis.replace('```json', '').replace('```', '')

            try:
                return json.loads(analysis)
            except json.JSONDecodeError:
                return {'conversation':[]}

        except Exception as e:
            print(f"An error occurred with OpenAI: {e}")
            return "Analysis failed due to an error. Ensure OpenAI API key is configured and valid."

    def chat(self, prompt: str):
        if self.model == MODEL_GPT_4O:
            return self.chat_with_openai(prompt)
        else:
            return self.chat_with_ollama(prompt)

        

    def analyze(self, transcript: str):
        prompt = f"""
        You are a helpful assistant specialized in medical conversation analysis. Your task is to analyze the following transcript of a conversation between a triage nurse and a patient.
        The Nurse asks most of the questions about the symptoms the patient has. Correct obvious grammatical errors in the transcription.
        For each line of dialogue, you must identify the speaker as either "NURSE" or "PATIENT".
        Do not generate any other prediction based on the context supplied.
        Transcript:
        {transcript}
        
        Format your response as JSON:
        {RESPONSE_FORMAT}
        """

        return self.chat(prompt)
