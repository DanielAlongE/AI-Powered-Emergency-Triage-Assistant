from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = Field(default="info", alias="LOG_LEVEL")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_completion_model: str = Field(default="gpt-4o-mini", alias="OPENAI_COMPLETION_MODEL")
    openai_transcription_model: str = Field(
        default="gpt-4o-mini-transcribe", alias="OPENAI_TRANSCRIPTION_MODEL"
    )
    openai_tts_model: str = Field(default="gpt-4o-mini-tts", alias="OPENAI_TTS_MODEL")

    # Ollama settings
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_default_model: str = Field(default="qwen2.5:0.5b", alias="OLLAMA_DEFAULT_MODEL")

    # Modal settings for remote GPU inference
    inference_mode: str = Field(default="auto", alias="INFERENCE_MODE")  # local, modal, auto
    modal_endpoint: str | None = Field(default=None, alias="MODAL_ENDPOINT")
    modal_gpu_type: str = Field(default="t4", alias="MODAL_GPU_TYPE")  # t4, a10g, a100
    modal_timeout: int = Field(default=600, alias="MODAL_TIMEOUT")  # seconds
    modal_token: str | None = Field(default=None, alias="MODAL_TOKEN_ID")  # Modal API token

    esi_source_path: Path = Field(default=Path("data/esi_protocol_samples.md"), alias="ESI_SOURCE_PATH")
    chroma_db_path: Path = Field(default=Path("data/chroma"), alias="CHROMA_DB_PATH")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    red_flag_lexicon_path: Path = Field(default=Path("config/red_flags.yaml"))
    request_timeout_seconds: float = Field(default=30.0)
    transcript_window_seconds: int = Field(default=120)
    max_follow_up_questions: int = Field(default=3)


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()