# This script uses the Ollama library to analyze a conversation transcript
# and predict which parts of the dialogue are from the nurse and which are from the patient.
#
# Before running, ensure you have the Ollama server installed and running, and the llama3 model pulled:
# 1. Install Ollama from https://ollama.ai/
# 2. Open a terminal and run: ollama run llama3
# 3. Ensure the 'ollama' Python library is installed: pip install ollama
import json
import ollama

RESPONSE_FORMAT = {
    'conversation': [
        {'role': 'assistant', 'content': 'I am the nurse!'},
        {'role': 'user', 'content': 'I am the patient.'},
        ]
    }

class ConversationAnalizer:
    def __init__(self):
        print("initialized ConversationAnalizer")

    def chat(self, prompt: str):
        try:
            # Use the ollama.chat() method for a conversational approach
            response = ollama.chat(
                model='llama3.2',
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.7,
                },
                format = "json"
            )

            # Extract the model's response
            # analysis = response['message']['content'].strip()
            analysis = response.message.content

            try:
                return json.loads(analysis)
            except json.JSONDecodeError:
                return {'conversation':[]}

        except Exception as e:
            print(f"An error occurred with Ollama: {e}")
            return "Analysis failed due to an error. Ensure Ollama is running and the model is available."


    def analyze(self, transcript: str):
        prompt = f"""
        You are a helpful assistant specialized in medical conversation analysis. Your task is to analyze the following transcript of a conversation between a triage nurse and a patient.
        For each line of dialogue, you must identify the speaker as either "NURSE" or "PATIENT".
        
        Transcript:
        {transcript}
        
        Format your response as JSON:
        {RESPONSE_FORMAT}
        """

        return self.chat(prompt)



