
## Prerequisites

### Install Ollama
Follow the installation instructions at [https://ollama.ai/](https://ollama.ai/)

### Pull Ollama Models
```
ollama pull llama3.2
ollama pull gpt-oss:20b
ollama pull gemma3
```

### Install FFmpeg
```
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html#build-windows
```

### Install SQLite
```
# macOS
brew install sqlite

# Ubuntu/Debian
sudo apt update && sudo apt install sqlite3

# Windows
# SQLite is usually pre-installed or download from https://www.sqlite.org/download.html
```

## Install Poetry
This will make poetry available globally
```
# macOS / Linux / WSL
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

## Install Project
```
cd  server
poetry install
```

## Database Setup
```
poetry run ingest
sqlite3 sessions.db ".quit"
poetry run alembic upgrade head
```

## Run Server Project

```
# development
poetry run uvicorn app.main:app

# or 

# production
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 4
```