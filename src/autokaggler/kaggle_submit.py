"""Safe Kaggle submission helper."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

SUBMIT_CONFIRMATION = "I_UNDERSTAND_THIS_SUBMITS_TO_KAGGLE"


def validate_submission_csv(path: Path, expected_rows: int) -> None:
    """Validate the minimal Titanic submission contract before any submit."""

    if not path.exists():
        raise FileNotFoundError(f"Submission file does not exist: {path}")
    df = pd.read_csv(path)
    if list(df.columns) != ["PassengerId", "Survived"]:
        raise ValueError("Submission must contain exactly PassengerId,Survived columns")
    if len(df) != expected_rows:
        raise ValueError(f"Submission row count {len(df)} != expected {expected_rows}")
    if not set(df["Survived"]).issubset({0, 1}):
        raise ValueError("Survived predictions must be binary 0/1 labels")


def submit_to_kaggle_if_requested(
    *,
    submit: bool,
    confirm_submit: str | None,
    competition: str,
    submission_path: Path,
    expected_rows: int,
    message: str,
    log_dir: Path,
) -> Dict[str, Any]:
    """Submit only when explicit safety gates are satisfied."""

    validate_submission_csv(submission_path, expected_rows)
    payload: Dict[str, Any] = {
        "requested": submit,
        "competition": competition,
        "submission_path": str(submission_path),
        "submitted": False,
        "status": "dry_run",
        "safety_gate": SUBMIT_CONFIRMATION,
    }
    if not submit:
        return payload
    if competition != "titanic":
        payload.update({"status": "blocked", "reason": "only titanic submission is currently allowed"})
        return payload
    if confirm_submit != SUBMIT_CONFIRMATION:
        payload.update({"status": "blocked", "reason": "missing explicit submit confirmation"})
        return payload

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        payload.update({"status": "blocked", "reason": "kaggle package is not installed"})
        return payload

    try:
        api = KaggleApi()
        api.authenticate()
        api.competition_submit(str(submission_path), message, competition)
    except Exception as exc:  # Kaggle API can fail on auth, network, or submission.
        payload.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
        return payload

    payload.update({"submitted": True, "status": "submitted", "message": message})
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"submit-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["submit_log_path"] = str(log_path)
    return payload


__all__ = ["SUBMIT_CONFIRMATION", "submit_to_kaggle_if_requested", "validate_submission_csv"]


