"""Optional bridge for NVIDIA/nvidia-kaggle style research workflows."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class NvidiaKaggleContext:
    """Metadata returned by the optional NVIDIA research bridge."""

    enabled: bool
    mode: str
    competition: str
    backend: str
    status: str
    details: Dict[str, Any]


class NvidiaKaggleAdapter:
    """Expose nvidia-kaggle as an optional lower-level research capability.

    The core Autokaggler runtime stays dependency-free. When
    ``AUTOKAGGLER_NVIDIA_KAGGLE_CMD`` is set, this adapter can call a local
    command that wraps NVIDIA/nvidia-kaggle. Otherwise it reports a structured
    dry-run context that downstream agents can use.
    """

    BACKEND_NAME = "nvidia-kaggle"

    def __init__(self, command: Optional[str] = None) -> None:
        self.command = command or os.environ.get("AUTOKAGGLER_NVIDIA_KAGGLE_CMD")

    def prepare(
        self,
        *,
        mode: str,
        competition: str,
        profile: str,
        dry_run: bool = True,
    ) -> NvidiaKaggleContext:
        """Return optional research context without submitting or uploading."""

        mode = (mode or "off").lower()
        if mode in {"off", "none", "false"}:
            return NvidiaKaggleContext(
                enabled=False,
                mode="off",
                competition=competition,
                backend=self.BACKEND_NAME,
                status="disabled",
                details={},
            )

        details: Dict[str, Any] = {
            "role": "lower-level Kaggle research/plugin capability",
            "profile": profile,
            "dry_run": dry_run,
            "command_configured": bool(self.command),
            "submission_upload": False,
            "dataset_upload": False,
        }

        if not self.command or dry_run:
            return NvidiaKaggleContext(
                enabled=True,
                mode=mode,
                competition=competition,
                backend=self.BACKEND_NAME,
                status="dry_run",
                details=details,
            )

        args = shlex.split(self.command) + [
            "--competition",
            competition,
            "--profile",
            profile,
            "--mode",
            mode,
            "--no-submit",
            "--no-upload",
        ]
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        details.update(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        )
        return NvidiaKaggleContext(
            enabled=True,
            mode=mode,
            competition=competition,
            backend=self.BACKEND_NAME,
            status="completed" if completed.returncode == 0 else "failed",
            details=details,
        )


__all__ = ["NvidiaKaggleAdapter", "NvidiaKaggleContext"]
