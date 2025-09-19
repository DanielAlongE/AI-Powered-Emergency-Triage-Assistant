# AI-Powered-Emergency-Triage-Assistant
This project aims to develop an AI-powered assistant to support nurses during the emergency department (ED) triage process. The system will operate as a "co-pilot," actively listening to the nurse-patient interview via microphone.


## Project LLM
#### Install Ollama
First, you need to install the Ollama application. Ollama is a single-file application that runs a local server for large language models.

Download: Go to the official Ollama website at https://ollama.com/ and download the installer for your operating system (macOS, Linux, or Windows).

Run the Installer: Follow the on-screen instructions to complete the installation.


#### Run the Llama 3.2 Model
```
ollama serve

ollama pull llama3.2

ollama run llama3.2
```



## Project Structure

```
AI-Powered-Emergency-Triage-Assistant/
├── LICENSE
├── README.md
├── client/
│   └── README.md
└── server/
    ├── api/
    ├── services/
    ├── .gitignore
    ├── main.py
    ├── requirements.txt
    ├── models/
    │   ├── __init__.py
    │   ├── llama/
    │   │   └── conversation_analizer.py
    │   └── transcription/
    │       ├── __init__.py
    │       ├── transcriber.py
    │       ├── vosk_transcriber.py
    │       └── whisper_transcriber.py
    ├── playground/
    │   └── whisper_debug.py
    └── services/
```
