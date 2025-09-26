import os, subprocess, time, requests, modal
from typing import Dict, Any

app = modal.App("ollama-triage")

# Persistent cache for models so you don’t re-download every cold start
OLLAMA_VOL = modal.Volume.from_name("ollama-cache", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install(
        "fastapi>=0.111.0", 
        "uvicorn>=0.30.0",
        "requests>=2.32.0",
        "pydantic>=2.5.0")
)

@app.cls(
    image=base_image,
    # gpu=modal.gpu.A100(size="40GB"),    # A100 40GB for large models like gpt-oss:20b
    gpu="B200",    # B200 for large models like gpt-oss:20b
    memory=12288,                       # a bit more headroom than 8GB
    concurrency_limit=1,                # one gen per GPU
    keep_warm=0,                        # keep daemon + cache warm
    timeout=600,
    container_idle_timeout=300,         # Scale down after 5 minutes of inactivity
    volumes={"/root/.ollama": OLLAMA_VOL},
    secrets=[],                         # add secrets if you need them
)
class OllamaServer:
    def __init__(self):
        self.base_url = "http://127.0.0.1:11434"

    @modal.enter()
    def start_daemon(self):
        # Boot once per container
        self.proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        # Wait until healthy
        for _ in range(60):
            try:
                requests.get(self.base_url)  # 200 when up
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("Ollama failed to start")

        # (Optional) Pre-warm frequently used models to avoid first-hit latency
        for m in os.getenv("PREPULL_MODELS", "qwen2.5:1.5b,phi3.5:3.8b-mini-instruct,llama3.2:3b").split(","):
            m = m.strip()
            if m:
                try:
                    requests.post(f"{self.base_url}/api/pull", json={"name": m}, timeout=600)
                except Exception:
                    pass  # don’t fail container on pre-pull issues

    @modal.exit()
    def stop_daemon(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    @modal.method()
    def chat(self, body: Dict[str, Any]):
        """
        body: { "model": "qwen2.5:1.5b", "messages": [...], "options": {...}, "stream": false }
        """
        # Ensure model exists (lazy pull)
        model = body.get("model", "qwen2.5:1.5b")
        try:
            requests.post(f"{self.base_url}/api/pull", json={"name": model}, timeout=600)
        except Exception as e:
            return {"error": f"pull failed: {e}"}

        # Forward to local Ollama API
        payload = {
            "model": model,
            "messages": body.get("messages") or [{"role":"user","content": body.get("prompt","")}],
            "options": body.get("options", {}),
            "stream": body.get("stream", False),
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=600)
        return r.json()

    @modal.method()
    def list_models(self):
        r = requests.get(f"{self.base_url}/api/tags", timeout=30)
        return r.json()

    @modal.method()
    def pull(self, name: str):
        r = requests.post(f"{self.base_url}/api/pull", json={"name": name}, timeout=600)
        return r.json()

# Web endpoints
@app.function(image=base_image)
@modal.web_endpoint(method="GET")
def health():
    return {"status": "healthy", "gpu": "T4"}

@app.function(image=base_image)
@modal.web_endpoint(method="POST", label="ollama-chat")
def chat(body: Dict[str, Any]):
    return OllamaServer().chat.remote(body)

@app.function(image=base_image)
@modal.web_endpoint(method="GET", label="ollama-list")
def list_models():
    return OllamaServer().list_models.remote()

@app.function(image=base_image)
@modal.web_endpoint(method="POST", label="ollama-pull")
def pull(body: Dict[str, Any]):
    name = body.get("name") or body.get("model")
    if not name:
        return {"error": "missing 'name' or 'model'"}
    return OllamaServer().pull.remote(name)
