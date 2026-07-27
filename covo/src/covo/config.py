"""Config loading and command rendering for reproducible pipelines."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML configs require PyYAML. Install it with `pip install pyyaml`.") from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config extension: {config_path.suffix}")
    if not isinstance(data, dict):
        raise ValueError("Pipeline config must be a mapping/object")
    return data


def _stringify_value(value: Any) -> List[str]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def build_command(step: Dict[str, Any]) -> List[str]:
    script = step.get("script")
    if not script:
        raise ValueError(f"Step is missing script: {step}")
    python = str(step.get("python", sys.executable))
    command = [python, str(script)]
    args = step.get("args", {}) or {}
    if not isinstance(args, dict):
        raise ValueError("step.args must be a mapping")
    for key, value in args.items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        command.append(flag)
        command.extend(_stringify_value(value))
    return command


def command_to_string(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_run_log(path: str | Path, payload: Dict[str, Any]) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_pipeline(
    config: Dict[str, Any], *, dry_run: bool = False, log_file: str | Path | None = None
) -> List[Dict[str, Any]]:
    steps = config.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("config.steps must be a list")
    env = os.environ.copy()
    env_updates = config.get("env", {}) or {}
    if not isinstance(env_updates, dict):
        raise ValueError("config.env must be a mapping")
    env.update({str(k): str(v) for k, v in env_updates.items()})
    log_path = log_file or config.get("log_file")

    results = []
    run_started_at = _utc_now()
    for idx, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            raise ValueError(f"step {idx} must be a mapping")
        name = str(step.get("name", f"step_{idx}"))
        command = build_command(step)
        entry = {"name": name, "command": command, "command_text": command_to_string(command)}
        print(f"[{idx}/{len(steps)}] {name}: {entry['command_text']}", flush=True)
        started_at = _utc_now()
        start_timer = time.perf_counter()
        entry["started_at"] = started_at
        if dry_run:
            entry["status"] = "dry_run"
            entry["returncode"] = None
        else:
            completed = subprocess.run(command, env=env)
            entry["returncode"] = int(completed.returncode)
            entry["status"] = "success" if completed.returncode == 0 else "failed"
        entry["finished_at"] = _utc_now()
        entry["duration_seconds"] = round(time.perf_counter() - start_timer, 3)
        results.append(entry)
        if log_path:
            _write_run_log(
                log_path,
                {
                    "name": config.get("name"),
                    "dry_run": dry_run,
                    "started_at": run_started_at,
                    "updated_at": _utc_now(),
                    "steps": results,
                },
            )
        if entry["returncode"] not in (0, None):
            raise subprocess.CalledProcessError(int(entry["returncode"]), command)
    return results
