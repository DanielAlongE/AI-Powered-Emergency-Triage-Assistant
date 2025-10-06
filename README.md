# AI-Powered-Emergency-Triage-Assistant
This project aims to develop an AI-powered assistant to support nurses during the emergency department (ED) triage process. The system will operate as a "co-pilot," actively listening to the nurse-patient interview via microphone.

## Demo
https://www.loom.com/share/fc4e158a4dc8495fbfe7b3cc2ade4319

[Demo Video](https://www.loom.com/share/fc4e158a4dc8495fbfe7b3cc2ade4319)

## Client

The client is a Vue.js application built with Vite, providing the user interface for the AI-powered triage assistant. It includes components for chat conversations, audio transcription, triage summaries, and dashboard views.

## Server

The server is a Python backend using Poetry for dependency management and FastAPI for the API. It integrates AI models for speech transcription (Vosk, Whisper), conversation analysis (Llama, OpenAI), and emergency severity assessment. It includes database management with SQLite and Alembic for migrations.

## Project Structure

```
AI-Powered-Emergency-Triage-Assistant/
├── LICENSE
├── README.md
├── TESTING.md
├── .gitignore
├── .railwayignore
├── docker-compose.gpu.yml
├── client/
│   ├── README.md
│   ├── package.json
│   ├── vite.config.js
│   ├── eslint.config.js
│   ├── jsconfig.json
│   ├── nginx.conf
│   ├── Dockerfile
│   ├── .env.example
│   ├── .env
│   ├── .env.production
│   ├── public/
│   │   └── favicon.ico
│   ├── src/
│   │   ├── App.vue
│   │   ├── main.js
│   │   ├── assets/
│   │   ├── components/
│   │   ├── plugins/
│   │   ├── router/
│   │   └── views/
│   └── ...
├── server/
│   ├── README.md
│   ├── pyproject.toml
│   ├── poetry.lock
│   ├── Dockerfile
│   ├── Dockerfile.gpu
│   ├── Procfile
│   ├── alembic.ini
│   ├── .env.example
│   ├── .gitignore
│   ├── alembic/
│   │   ├── env.py
│   │   ├── script.py.mako
│   │   └── versions/
│   ├── config/
│   │   └── red_flags.yaml
│   ├── data/
│   │   ├── Emergency_Severity_Index_Handbook.pdf
│   │   └── esi_protocol_samples.md
│   ├── src/
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── database.py
│   │   │   ├── main.py
│   │   │   ├── models.py
│   │   │   ├── routes.py
│   │   │   └── schemas.py
│   │   ├── agents/
│   │   ├── models/
│   │   ├── services/
│   │   ├── utilities/
│   │   └── playground/
│   ├── testing/
│   └── ...
└── tests/
    └── data/
```
