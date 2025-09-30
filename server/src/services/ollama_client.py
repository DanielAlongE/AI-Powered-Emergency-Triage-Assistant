from __future__ import annotations
import asyncio
import json
import re
from typing import Any, Dict, Optional
import ollama
import httpx
from config import get_settings
from logger import get_logger

logger = get_logger(__name__)


def _is_cloud_model(model: str) -> bool:
    """Check if a model is an Ollama cloud model (ends with -cloud)."""
    return model.endswith("-cloud")


def _extract_json_blob(text: str) -> Dict[str, Any]:
    """Extract JSON from text response - same as openai_client.py"""
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


class OllamaGateway:
    """Ollama client gateway for local, cloud, and remote (Modal) LLM inference."""

    def __init__(self, host: Optional[str] = None, inference_mode: Optional[str] = None) -> None:
        settings = get_settings()
        self._host = host or settings.ollama_host
        self._default_model = settings.ollama_default_model
        self._settings = settings

        # Determine inference mode
        self._inference_mode = inference_mode or settings.inference_mode
        self._modal_endpoint = settings.modal_endpoint

        # Simple cache for model availability (to reduce /api/tags calls)
        self._model_cache = {}
        self._cache_timestamp = 0
        self._cache_duration = 30  # Cache for 30 seconds

        # Initialize appropriate client
        if self._inference_mode == "modal" and self._modal_endpoint:
            self._client = None  # Will use HTTP client for Modal
            self._is_modal = True
            self._is_cloud = False
        elif self._inference_mode == "cloud" or settings.ollama_cloud_enabled:
            # For cloud mode, use ollama client with authentication
            self._client = ollama.Client()  # Uses default ollama.com endpoint when authenticated
            self._is_modal = False
            self._is_cloud = True
        else:
            self._client = ollama.Client(host=self._host)
            self._is_modal = False
            self._is_cloud = False

        logger.info("ollama_gateway_initialized",
                   host=self._host,
                   default_model=self._default_model,
                   inference_mode=self._inference_mode,
                   is_modal=self._is_modal,
                   is_cloud=self._is_cloud)

    def _check_model_availability(self, model: str) -> bool:
        """Check if model is available (locally, cloud, or on Modal)."""
        if self._is_modal:
            # For Modal, we assume models can be pulled on demand
            # Or we could make an HTTP request to the list endpoint
            return True

        if self._is_cloud or _is_cloud_model(model):
            # For cloud models, we assume they're available if authenticated
            # The ollama library will handle authentication
            return True

        # Check cache first
        import time
        current_time = time.time()
        if (current_time - self._cache_timestamp < self._cache_duration and
            self._model_cache and model in self._model_cache):
            return self._model_cache[model]

        try:
            models_response = self._client.list()

            # Extract model names from Ollama ListResponse
            available_models = []
            if hasattr(models_response, 'models'):
                for model_info in models_response.models:
                    if hasattr(model_info, 'model'):
                        available_models.append(model_info.model)

            # Update cache
            self._model_cache.clear()
            self._cache_timestamp = current_time
            for available_model in available_models:
                self._model_cache[available_model] = True

            # Direct match first
            if model in available_models:
                self._model_cache[model] = True
                return True

            # Try fuzzy matching (in case of tag differences like :latest)
            for available_model in available_models:
                # Check if the model name matches without considering tags
                model_base = model.split(':')[0]
                available_base = available_model.split(':')[0]
                if model_base == available_base:
                    self._model_cache[model] = True
                    return True

            # Cache the negative result too
            self._model_cache[model] = False
            logger.info("model_not_found",
                       requested_model=model,
                       available_models=available_models)
            return False
        except Exception as e:
            logger.warning("model_availability_check_failed", error=str(e))
            return False

    async def _modal_completion(self, prompt: str, model: str, temperature: float) -> str:
        """Make completion request to Modal endpoint."""
        if not self._modal_endpoint:
            raise RuntimeError("Modal endpoint not configured")

        async with httpx.AsyncClient(timeout=self._settings.modal_timeout, follow_redirects=True) as client:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": temperature}
            }

            response = await client.post(
                self._modal_endpoint,
                json=payload
            )
            response.raise_for_status()

            result = response.json()
            if "error" in result:
                raise RuntimeError(f"Modal inference error: {result['error']}")

            return result.get("message", {}).get("content", "")

    async def stream_completion(
        self,
        *,
        prompt: str,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate completion using Ollama.

        Args:
            prompt: The prompt to complete
            json_schema: Optional JSON schema for structured output
            temperature: Sampling temperature
            model_override: Override default model

        Returns:
            Dict with parsed response
        """
        model = model_override or self._default_model

        # Add JSON formatting instruction if schema is provided
        if json_schema:
            prompt += "\n\nPlease respond with valid JSON only, matching the required schema structure."

        # Handle Modal vs Cloud vs Local inference
        if _is_cloud_model(model):
            # Always use cloud for cloud models regardless of mode
            text = await self._cloud_completion(prompt, model, temperature, json_schema)

        elif self._inference_mode == "cloud" or self._is_cloud:
            # Force cloud mode
            text = await self._cloud_completion(prompt, model, temperature, json_schema)

        elif self._is_modal:
            # Use Modal endpoint
            try:
                text = await self._modal_completion(prompt, model, temperature)
            except Exception as e:
                # Try fallback to local if auto mode
                if self._inference_mode == "auto" and self._client:
                    logger.warning("modal_inference_failed_fallback_local", error=str(e))
                    return await self._local_completion(prompt, model, temperature, json_schema)
                else:
                    raise

        else:
            # Use local Ollama
            text = await self._local_completion(prompt, model, temperature, json_schema)

        # Process response
        if json_schema:
            parsed = _extract_json_blob(text)
            if not parsed:
                logger.warning("json_parse_error",
                             text_preview=text[:200] if text else "",
                             model=model)
            return parsed

        return {"text": text}

    async def _local_completion(self, prompt: str, model: str, temperature: float, json_schema: Optional[Dict] = None) -> str:
        """Handle local Ollama completion."""
        # Check if model is available locally (skip for cloud models)
        if not _is_cloud_model(model) and not self._check_model_availability(model):
            logger.warning("model_not_available", model=model)
            try:
                models_response = self._client.list()
                available_models = []
                if hasattr(models_response, 'models'):
                    for model_info in models_response.models:
                        if hasattr(model_info, 'model'):
                            available_models.append(model_info.model)

                if available_models:
                    model = available_models[0]  # Use first available model
                    logger.info("using_fallback_model", model=model)
                else:
                    raise RuntimeError(f"No Ollama models available. Please install a model first.")
            except Exception:
                raise RuntimeError(f"No Ollama models available. Please install a model first.")

        # Prepare options
        options = {'temperature': temperature}

        def _call() -> str:
            try:
                response = self._client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options=options
                )
                return response['message']['content']
            except Exception as e:
                logger.error("ollama_completion_error", error=str(e), model=model)
                raise

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _call)
        logger.info("ollama_completion_complete", model=model, mode="local")
        return result

    async def _cloud_completion(self, prompt: str, model: str, temperature: float, json_schema: Optional[Dict] = None) -> str:
        """Handle Ollama cloud completion."""
        # Prepare options
        options = {'temperature': temperature}

        def _call() -> str:
            try:
                response = self._client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options=options
                )
                return response['message']['content']
            except Exception as e:
                logger.error("ollama_cloud_completion_error", error=str(e), model=model)
                raise

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _call)
        logger.info("ollama_completion_complete", model=model, mode="cloud")
        return result

    async def _modal_completion_with_fallback(self, prompt: str, model: str, temperature: float) -> str:
        """Modal completion with automatic fallback handling."""
        try:
            text = await self._modal_completion(prompt, model, temperature)
            logger.info("ollama_completion_complete", model=model, mode="modal")
            return text
        except Exception as e:
            logger.warning("modal_completion_error", error=str(e), model=model)
            raise


_gateway: OllamaGateway | None = None


def get_ollama_gateway(host: Optional[str] = None, inference_mode: Optional[str] = None) -> OllamaGateway:
    """Get singleton Ollama gateway instance."""
    global _gateway
    if _gateway is None:
        _gateway = OllamaGateway(host=host, inference_mode=inference_mode)
    return _gateway