from __future__ import annotations
import asyncio
import json
import re
from io import BytesIO
from typing import Any, Dict, Optional
from openai import OpenAI
from config import get_settings
from logger import get_logger

logger = get_logger(__name__)


def _extension_from_mime(mime_type: str | None) -> str:
    if not mime_type:
        return "webm"
    base = mime_type.split(";", 1)[0].lower()
    mapping = {
        "audio/webm": "webm",
        "audio/ogg": "ogg",
        "audio/mpeg": "mp3",
        "audio/wav": "wav",
    }
    return mapping.get(base, "webm")


def _extract_json_blob(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    stripped = text.strip()
    code_blocks = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    for block in code_blocks:
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue
    brace_index = stripped.find("{")
    while brace_index != -1:
        candidate = stripped[brace_index:]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            brace_index = stripped.find("{", brace_index + 1)
    return {}


class HostedModelGateway:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            logger.warning("openai_key_missing")
            self._openai_client: OpenAI | None = None
        else:
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        self._settings = settings

    async def transcribe(self, audio_bytes: bytes, *, mime_type: str | None = None) -> Any:
        if not self._openai_client:
            raise RuntimeError("OpenAI client not configured")

        file_obj = BytesIO(audio_bytes)
        file_obj.name = f"audio.{_extension_from_mime(mime_type)}"

        def _call() -> Any:
            return self._openai_client.audio.transcriptions.create(
                model=self._settings.openai_transcription_model,
                file=file_obj,
                response_format="json",
                language="en",
            )

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _call)
        logger.info("openai_transcription_complete")
        return result

    async def stream_completion(
        self,
        *,
        prompt: str,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        if not self._openai_client:
            raise RuntimeError("OpenAI client not configured")

        kwargs: Dict[str, Any] = {
            "model": self._settings.openai_completion_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        def _call() -> Dict[str, Any]:
            response = self._openai_client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content
            if json_schema:
                parsed = _extract_json_blob(text)
                if not parsed:
                    logger.warning("json_parse_error", text_preview=text[:200] if text else "")
                return parsed
            return {"text": text}

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _call)
        logger.info("openai_completion_complete")
        return result

    async def text_to_speech(self, text: str, *, voice: str = "alloy") -> bytes:
        if not self._openai_client:
            raise RuntimeError("OpenAI client not configured")

        def _call() -> bytes:
            response = self._openai_client.audio.speech.create(
                model=self._settings.openai_tts_model,
                voice=voice,
                input=text,
            )
            return response.read()  # type: ignore[return-value]

        loop = asyncio.get_running_loop()
        audio_bytes = await loop.run_in_executor(None, _call)
        logger.info("openai_tts_complete")
        return audio_bytes


_gateway: HostedModelGateway | None = None


def get_hosted_model_gateway() -> HostedModelGateway:
    global _gateway
    if _gateway is None:
        _gateway = HostedModelGateway()
    return _gateway