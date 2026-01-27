# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def utcnow_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class StepResult:
    ok: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None


class RunState:
    """Lightweight JSON state store for resumable per-interview processing.

    Design goals:
    - Append/overwrite safe: write atomically.
    - Resume: skip steps marked success.
    - Continue-on-failure: record error per interview+step and move on.
    """

    def __init__(self, state_path: Path):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = {
            "version": 1,
            "created_at": utcnow_iso(),
            "updated_at": utcnow_iso(),
            "interviews": {},
        }

    @staticmethod
    def default_path(project_root: Path) -> Path:
        return Path(project_root) / ".qualivault" / "run_state.json"

    def load(self) -> "RunState":
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        return self

    def save(self) -> None:
        self.data["updated_at"] = utcnow_iso()
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.state_path)

    def _ensure_interview(self, interview_key: str) -> Dict[str, Any]:
        interviews = self.data.setdefault("interviews", {})
        iv = interviews.get(interview_key)
        if iv is None:
            iv = {
                "created_at": utcnow_iso(),
                "steps": {},
            }
            interviews[interview_key] = iv
        iv.setdefault("steps", {})
        return iv

    def get_step_status(self, interview_key: str, step: str) -> str:
        iv = self._ensure_interview(interview_key)
        return iv.get("steps", {}).get(step, {}).get("status", "pending")

    def mark_step_running(self, interview_key: str, step: str) -> None:
        iv = self._ensure_interview(interview_key)
        iv["steps"].setdefault(step, {})
        iv["steps"][step].update({"status": "running", "started_at": utcnow_iso()})
        self.save()

    def mark_step_success(self, interview_key: str, step: str, **meta: Any) -> None:
        iv = self._ensure_interview(interview_key)
        iv["steps"].setdefault(step, {})
        iv["steps"][step].update({"status": "success", "finished_at": utcnow_iso(), "error": None})
        if meta:
            iv["steps"][step].setdefault("meta", {})
            iv["steps"][step]["meta"].update(meta)
        self.save()

    def mark_step_failed(self, interview_key: str, step: str, err: BaseException, **meta: Any) -> None:
        iv = self._ensure_interview(interview_key)
        iv["steps"].setdefault(step, {})
        iv["steps"][step].update(
            {
                "status": "failed",
                "finished_at": utcnow_iso(),
                "error": {
                    "type": type(err).__name__,
                    "message": str(err),
                    "traceback": traceback.format_exc(limit=30),
                },
            }
        )
        if meta:
            iv["steps"][step].setdefault("meta", {})
            iv["steps"][step]["meta"].update(meta)
        self.save()

    def should_skip(self, interview_key: str, step: str, *, resume: bool = True) -> bool:
        if not resume:
            return False
        return self.get_step_status(interview_key, step) == "success"
