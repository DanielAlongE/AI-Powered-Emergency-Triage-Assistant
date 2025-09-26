"""
Modal deployment service for running Ollama with GPU acceleration.

This service provides a FastAPI endpoint that's compatible with the Ollama API,
running on Modal's GPU infrastructure for fast inference.
"""

import json
import os
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime

import modal

# Create Modal app
app = modal.App("ollama-triage")

# Docker image with Ollama pre-installed and models pre-pulled
ollama_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates")
    .run_commands(
        # Install Ollama
        "curl -fsSL https://ollama.com/install.sh | sh",
        # Verify installation
        "which ollama"
    )
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "pydantic==2.5.0"
    )
    .run_commands(
        # Create script using echo commands (avoiding heredoc parsing issues)
        "echo '#!/bin/bash' > /tmp/pull_models.sh",
        "echo 'set -e' >> /tmp/pull_models.sh",
        "echo 'echo \"Starting ollama server...\"' >> /tmp/pull_models.sh",
        "echo 'ollama serve &' >> /tmp/pull_models.sh",
        "echo 'OLLAMA_PID=$!' >> /tmp/pull_models.sh",
        "echo 'sleep 10' >> /tmp/pull_models.sh",
        "echo 'echo \"Waiting for ollama...\"' >> /tmp/pull_models.sh",
        "echo 'timeout 30 bash -c \"until ollama list >/dev/null 2>&1; do sleep 1; done\"' >> /tmp/pull_models.sh",
        "echo 'echo \"Pre-pulling models...\"' >> /tmp/pull_models.sh",
        "echo 'ollama pull qwen2.5:0.5b && echo \"✓ qwen2.5:0.5b\"' >> /tmp/pull_models.sh",
        "echo 'ollama pull qwen2.5:1.5b && echo \"✓ qwen2.5:1.5b\"' >> /tmp/pull_models.sh",
        "echo 'ollama pull llama3.2:1b && echo \"✓ llama3.2:1b\"' >> /tmp/pull_models.sh",
        "echo 'ollama pull gemma2:2b && echo \"✓ gemma2:2b\"' >> /tmp/pull_models.sh",
        "echo 'ollama pull phi3.5 && echo \"✓ phi3.5\"' >> /tmp/pull_models.sh",
        "echo 'ollama pull llama3.2 && echo \"✓ llama3.2\"' >> /tmp/pull_models.sh",
        "echo 'ollama pull llama3.1 && echo \"✓ llama3.1\"' >> /tmp/pull_models.sh",
        "echo 'ollama pull llama2 && echo \"✓ llama2\"' >> /tmp/pull_models.sh",
        "echo 'ollama pull qwen2:7b-instruct && echo \"✓ qwen2:7b-instruct\"' >> /tmp/pull_models.sh",
        "echo 'ollama pull gpt-oss:20b && echo \"✓ gpt-oss:20b\"' >> /tmp/pull_models.sh",
        "echo 'kill $OLLAMA_PID 2>/dev/null || true' >> /tmp/pull_models.sh",
        "echo 'wait $OLLAMA_PID 2>/dev/null || true' >> /tmp/pull_models.sh",
        "echo 'echo \"All models pre-pulled!\"' >> /tmp/pull_models.sh",
        # Make executable and run
        "chmod +x /tmp/pull_models.sh",
        "/tmp/pull_models.sh"
    )
)


@app.function(
    image=ollama_image,
    gpu=modal.gpu.T4(count=1),  # Single T4 GPU
    memory=8192,  # 8GB RAM
    timeout=600,  # 10 minute timeout
    concurrency_limit=1,  # One request at a time for simplicity
    keep_warm=0,  # Scale to zero when idle
)
@modal.web_endpoint(method="POST", label="ollama-chat")
def ollama_chat_endpoint(request_data: Dict[str, Any]):
    """
    Chat endpoint that mimics Ollama's API format.

    Expected format:
    {
        "model": "qwen2.5:0.5b",
        "messages": [{"role": "user", "content": "prompt"}],
        "options": {"temperature": 0.3}
    }
    """
    import subprocess
    import json
    import time

    # Start Ollama server in background
    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for Ollama to be ready
    time.sleep(2)

    try:
        model = request_data.get("model", "qwen2.5:0.5b")
        messages = request_data.get("messages", [])
        options = request_data.get("options", {})

        # Check if model is available, pull if needed (most common models are pre-pulled)
        try:
            # First check if model is already available
            list_result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if model not in list_result.stdout:
                # Model not found, try to pull it
                subprocess.run(["ollama", "pull", model], check=True, capture_output=True, timeout=300)
        except subprocess.CalledProcessError as e:
            return {
                "error": f"Failed to pull model {model}: {e.stderr.decode() if e.stderr else str(e)}"
            }
        except subprocess.TimeoutExpired:
            return {
                "error": f"Timeout pulling model {model}. Try a pre-pulled model: qwen2.5:0.5b, qwen2.5:1.5b, llama3.2:1b, gemma2:2b, phi3.5, llama3.2, llama3.1, llama2, qwen2:7b-instruct, gpt-oss:20b"
            }

        # Format prompt from messages
        if messages:
            prompt = messages[-1].get("content", "")
        else:
            prompt = request_data.get("prompt", "")

        # Build ollama command - use environment variable for temperature
        ollama_cmd = ["ollama", "run", model]

        # Set temperature via environment variable (more reliable than --parameter)
        env = {}
        if options.get("temperature") is not None:
            env["OLLAMA_TEMPERATURE"] = str(options['temperature'])

        # Run inference with environment variables
        import os
        current_env = os.environ.copy()
        current_env.update(env)

        result = subprocess.run(
            ollama_cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=300,  # 5 minute timeout for inference
            env=current_env
        )

        if result.returncode != 0:
            return {
                "error": f"Ollama inference failed: {result.stderr}"
            }

        response_text = result.stdout.strip()

        # Return in Ollama-compatible format
        return {
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "model": model,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "done": True
        }

    except Exception as e:
        return {
            "error": f"Inference error: {str(e)}"
        }

    finally:
        # Clean up Ollama process
        try:
            ollama_process.terminate()
            ollama_process.wait(timeout=5)
        except:
            ollama_process.kill()


@app.function(
    image=ollama_image,
    gpu=modal.gpu.T4(count=1),
    memory=8192,
    timeout=300,
    concurrency_limit=1,
)
@modal.web_endpoint(method="GET", label="ollama-list")
def ollama_list_endpoint():
    """
    List available models endpoint that mimics Ollama's API.
    """
    import subprocess
    import time

    # Start Ollama server
    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(2)

    try:
        # Get list of models
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return {"models": []}

        # Parse ollama list output
        models = []
        lines = result.stdout.strip().split('\n')[1:]  # Skip header

        for line in lines:
            parts = line.split()
            if parts:
                model_name = parts[0]
                models.append({
                    "name": model_name,
                    "model": model_name,  # For compatibility
                    "size": parts[2] if len(parts) > 2 else "unknown",
                    "modified_at": datetime.utcnow().isoformat() + "Z"
                })

        return {"models": models}

    except Exception as e:
        return {"error": f"Failed to list models: {str(e)}"}

    finally:
        try:
            ollama_process.terminate()
            ollama_process.wait(timeout=5)
        except:
            ollama_process.kill()


@app.function(
    image=ollama_image,
    gpu=modal.gpu.T4(count=1),
    timeout=600,
)
@modal.web_endpoint(method="POST", label="ollama-pull")
def ollama_pull_endpoint(request_data: Dict[str, Any]):
    """
    Pull model endpoint that mimics Ollama's API.
    """
    import subprocess
    import time

    model = request_data.get("name", request_data.get("model"))
    if not model:
        return {"error": "No model specified"}

    # Start Ollama server
    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(2)

    try:
        # Pull the model
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes for download
        )

        if result.returncode == 0:
            return {"status": "success", "model": model}
        else:
            return {"error": f"Failed to pull {model}: {result.stderr}"}

    except Exception as e:
        return {"error": f"Pull error: {str(e)}"}

    finally:
        try:
            ollama_process.terminate()
            ollama_process.wait(timeout=5)
        except:
            ollama_process.kill()


# Health check endpoint
@app.function(image=ollama_image)
@modal.web_endpoint(method="GET", label="health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "modal-ollama", "gpu": "T4"}


if __name__ == "__main__":
    # This allows running `modal serve modal_ollama_service.py` for development
    pass