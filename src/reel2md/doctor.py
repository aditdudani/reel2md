from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from typing import List

@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_doctor(ollama_host: str, vision_model: str) -> int:
    checks: List[CheckResult] = []
    checks.extend(check_python_modules())
    checks.extend(check_commands())
    checks.append(check_ollama(ollama_host, vision_model))

    for check in checks:
        status = "OK" if check.ok else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")

    return 0 if all(check.ok for check in checks) else 1


def check_python_modules() -> List[CheckResult]:
    modules = {
        "yt_dlp": "yt-dlp importable",
        "whisper": "openai-whisper importable",
        "requests": "requests importable",
        "pytesseract": "pytesseract importable",
        "PIL": "Pillow importable",
    }
    results = []
    for module_name, detail in modules.items():
        found = importlib.util.find_spec(module_name) is not None
        results.append(CheckResult(name=module_name, ok=found, detail=detail if found else "missing"))
    return results


def check_commands() -> List[CheckResult]:
    commands = ["ffmpeg", "tesseract", "ollama"]
    results = []
    for command in commands:
        location = shutil.which(command)
        results.append(
            CheckResult(
                name=command,
                ok=location is not None,
                detail=location or "not found on PATH",
            )
        )
    return results


def check_ollama(ollama_host: str, vision_model: str) -> CheckResult:
    requests_spec = importlib.util.find_spec("requests")
    if requests_spec is None:
        return CheckResult(
            name="ollama-api",
            ok=False,
            detail="cannot check because Python package 'requests' is missing",
        )

    import requests

    try:
        response = requests.get(f"{ollama_host.rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
    except Exception as exc:
        return CheckResult(
            name="ollama-api",
            ok=False,
            detail=f"unreachable at {ollama_host}: {exc}",
        )

    payload = response.json()
    names = [model.get("name", "") for model in payload.get("models", [])]
    normalized = {name.split(":")[0] for name in names} | set(names)
    if vision_model in normalized:
        return CheckResult(
            name="ollama-api",
            ok=True,
            detail=f"reachable, model '{vision_model}' available",
        )
    return CheckResult(
        name="ollama-api",
        ok=False,
        detail=f"reachable, but model '{vision_model}' is missing",
    )
