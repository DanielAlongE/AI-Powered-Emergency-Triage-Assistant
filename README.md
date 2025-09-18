# AI-Powered-Emergency-Triage-Assistant
This project aims to develop an AI-powered assistant to support nurses during the emergency department (ED) triage process. The system will operate as a "co-pilot," actively listening to the nurse-patient interview via microphone.

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
