"""Small, explicit I/O and execution records shared by this standalone demo."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]

def save_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (left != right)))
        previous = current
    return previous[-1]

def words(text):
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)

def normalized(text):
    return " ".join(words(text))

def json_response(text):
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value)
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from the model")
    return result

class NIMClient:
    """NVIDIA NIM chat client; every attempted HTTP request consumes budget.

    No automatic retries and no fixture fallback. Keys never enter logs.
    """
    def __init__(self, config, output):
        self.config = config
        self.path = Path(output) / "nim-attempts.jsonl"
        self.key = os.environ.get("NVIDIA_API_KEY")
        if not self.key:
            raise RuntimeError("Set NVIDIA_API_KEY in your environment or Colab Secrets")
        self.limit = int(config.get("max_requests", 64))
        self.count = len(read_jsonl(self.path)) if self.path.exists() else 0
        self.model = config["nim_model"]

    def chat(self, messages, max_tokens=512):
        if self.count >= self.limit:
            raise RuntimeError("NIM attempt budget exhausted")
        self.count += 1
        record = {"attempt": self.count, "model": self.model, "status": "started",
                  "input_sha256": fingerprint(messages)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        payload = {"model": self.model, "messages": messages, "temperature": 0,
                   "max_tokens": max_tokens, "stream": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        endpoint = self.config.get("nim_base_url", "https://integrate.api.nvidia.com/v1")
        if not endpoint.startswith("https://") and not endpoint.startswith("http://localhost"):
            raise ValueError("Use HTTPS or localhost for NIM")
        req = Request(endpoint.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        start = time.monotonic()
        try:
            with urlopen(req, timeout=int(self.config.get("timeout_seconds", 120))) as response:
                data = json.load(response)
        except HTTPError as error:
            save_json(self.path.parent / f"attempt-{self.count:04d}-error.json",
                      {"attempt": self.count, "status": "error", "http_status": error.code})
            raise RuntimeError(f"NVIDIA NIM HTTP {error.code}; no fallback was used") from None
        save_json(self.path.parent / f"attempt-{self.count:04d}.json",
                  {"attempt": self.count, "elapsed_seconds": time.monotonic()-start,
                   "model": self.model, "response": data})
        return data["choices"][0]["message"]["content"]

def ensure_disjoint(splits, key="id"):
    seen = {}
    for name, rows in splits.items():
        for row in rows:
            identity = str(row[key])
            if identity in seen:
                raise ValueError(f"Duplicate/leaked {key} {identity}: {seen[identity]} and {name}")
            seen[identity] = name

def package_versions():
    from importlib.metadata import version, PackageNotFoundError
    result = {}
    for name in ("torch", "transformers", "nemo-automodel", "nemo_toolkit", "data-designer", "huggingface-hub"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result
