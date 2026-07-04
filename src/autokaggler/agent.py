"""Core agent implementation orchestrating the AutoKaggler pipeline."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .data_manager import DataManager
from .kaggle_submit import SUBMIT_CONFIRMATION, submit_to_kaggle_if_requested
from .nvidia_adapter import NvidiaKaggleAdapter, NvidiaKaggleContext
from .pipeline import TitanicPipeline, TitanicPipelineResult
from .reporting import write_titanic_mastery_report

RUNTIME_DIRS = [Path(".agent_tmp"), Path(".agent_logs")]
REPORT_DIR = RUNTIME_DIRS[0] / "reports"
DEFAULT_PROFILE = "fast"
VALID_PROFILES = {"fast", "tree", "boosting", "ensemble", "power", "linear", "auto"}
PROFILE_ALIASES = {"linear": "fast", "power": "tree"}
TAG = "#KGNINJA"


@dataclass
class TaskInput:
    """Declarative configuration for a pipeline execution."""

    profile: Optional[str] = None
    competition: str = "titanic"
    force_download: bool = False
    data_source: str = "auto"
    random_seed: int = 42
    submission_name: Optional[str] = None
    notes: Optional[str] = None
    report: bool = True
    submit: bool = False
    confirm_submit: Optional[str] = None
    submit_message: str = "KaggleDance autonomous Titanic submission"
    nvidia_mode: str = "off"
    nvidia_dry_run: bool = True

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Structured response produced by the agent."""

    ok: bool
    meta: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def bootstrap() -> None:
    """Prepare runtime directories and default environment."""

    for directory in RUNTIME_DIRS:
        directory.mkdir(exist_ok=True)
    os.environ.setdefault("PROFILE", DEFAULT_PROFILE)


def load_task_input(raw: str) -> TaskInput:
    """Deserialize ``TaskInput`` from raw JSON."""

    if not raw.strip():
        payload: Dict[str, Any] = {}
    else:
        payload = json.loads(raw)
    extra = {k: v for k, v in payload.items() if k not in TaskInput.__dataclass_fields__}
    params = {
        k: payload[k]
        for k in TaskInput.__dataclass_fields__
        if k != "extra" and k in payload
    }
    task_input = TaskInput(**params)  # type: ignore[arg-type]
    task_input.extra.update(extra)
    return task_input


def configure_logging(run_id: str) -> Path:
    """Initialise logging infrastructure for the current run."""

    log_path = RUNTIME_DIRS[1] / f"run-{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )
    logging.info("Logging initialised for run %s", run_id)
    return log_path


def resolve_profile(task_input: TaskInput) -> str:
    """Determine which profile to run based on input and environment."""

    profile = (task_input.profile or os.environ.get("PROFILE", DEFAULT_PROFILE)).lower()
    if profile not in VALID_PROFILES:
        logging.warning(
            "Unknown profile '%s'; falling back to default '%s'",
            profile,
            DEFAULT_PROFILE,
        )
        profile = DEFAULT_PROFILE
    profile = PROFILE_ALIASES.get(profile, profile)
    os.environ["PROFILE"] = profile
    return profile


def prepare_nvidia_context(
    task_input: TaskInput, profile: str
) -> NvidiaKaggleContext:
    """Prepare optional NVIDIA/nvidia-kaggle research context."""

    adapter = NvidiaKaggleAdapter()
    return adapter.prepare(
        mode=task_input.nvidia_mode,
        competition=task_input.competition,
        profile=profile,
        dry_run=task_input.nvidia_dry_run,
    )


def run_agent(
    task_input: TaskInput, run_id: str
) -> tuple[TitanicPipelineResult, NvidiaKaggleContext]:
    """Execute the Titanic pipeline."""

    started = time.perf_counter()
    if task_input.competition != "titanic":
        raise ValueError("Only competition='titanic' is implemented in this release")

    data_manager = DataManager(cache_dir=RUNTIME_DIRS[0])
    train_df, test_df, data_meta = data_manager.prepare_datasets(
        prefer_source=task_input.data_source,
        force_download=task_input.force_download,
    )

    profile = resolve_profile(task_input)
    nvidia_context = prepare_nvidia_context(task_input, profile)
    submission_name = task_input.submission_name or f"submission-{run_id}.csv"
    pipeline = TitanicPipeline(profile=profile, random_seed=task_input.random_seed)
    result = pipeline.run(
        train_df=train_df,
        test_df=test_df,
        submission_name=submission_name,
        output_dir=data_manager.submission_dir,
        notes=task_input.notes,
        data_meta=data_meta,
    )

    submit_result = submit_to_kaggle_if_requested(
        submit=task_input.submit,
        confirm_submit=task_input.confirm_submit,
        competition=task_input.competition,
        submission_path=Path(result.submission_path),
        expected_rows=len(test_df),
        message=task_input.submit_message,
        log_dir=RUNTIME_DIRS[1],
    )
    result.submit_result = submit_result

    if task_input.report:
        report_path = write_titanic_mastery_report(
            result=result,
            run_id=run_id,
            report_dir=REPORT_DIR,
            elapsed_seconds=time.perf_counter() - started,
            competition=task_input.competition,
            submit_status=submit_result,
        )
        result.report_path = str(report_path)
    return result, nvidia_context


def build_success_result(
    run_id: str,
    log_path: Path,
    result: TitanicPipelineResult,
    profile: str,
    nvidia_context: Optional[NvidiaKaggleContext] = None,
) -> AgentResult:
    """Construct a success payload."""

    meta = {
        "profile": profile,
        "run_id": run_id,
        "tags": [TAG],
        "log_file": str(log_path),
        "cache_dir": str(RUNTIME_DIRS[0]),
    }
    payload = {
        "cv_mean_accuracy": result.cv_mean,
        "cv_std": result.cv_std,
        "model_name": result.model_name,
        "requested_profile": result.requested_profile,
        "selected_profile": result.selected_profile,
        "cv_results": result.cv_results,
        "feature_columns": result.feature_columns,
        "feature_importance": result.feature_importance,
        "submission_path": result.submission_path,
        "report_path": result.report_path,
        "submit_result": result.submit_result,
        "data_source": result.data_source,
        "notes": result.notes,
        "achievement": "Titanic Mastery Achieved",
        "submit_confirmation_required": SUBMIT_CONFIRMATION,
    }
    if nvidia_context is not None:
        payload["nvidia_kaggle"] = {
            "enabled": nvidia_context.enabled,
            "mode": nvidia_context.mode,
            "competition": nvidia_context.competition,
            "backend": nvidia_context.backend,
            "status": nvidia_context.status,
            "details": nvidia_context.details,
        }
    return AgentResult(ok=True, meta=meta, result=payload)


def build_failure_result(run_id: str, log_path: Path, exc: BaseException) -> AgentResult:
    """Construct a failure payload with diagnostic information."""

    meta = {
        "profile": os.environ.get("PROFILE", DEFAULT_PROFILE),
        "run_id": run_id,
        "tags": [TAG],
        "log_file": str(log_path),
        "cache_dir": str(RUNTIME_DIRS[0]),
    }
    error = "{}: {}".format(type(exc).__name__, exc)
    logging.error("Pipeline failed: %s", error)
    logging.debug("Traceback:\n%s", traceback.format_exc())
    return AgentResult(ok=False, meta=meta, error=error)


def serialise_result(result: AgentResult) -> str:
    """Serialise the agent result to JSON."""

    return json.dumps(result.__dict__, ensure_ascii=False, default=str)


def main() -> None:
    """Entry point for executing the AutoKaggler agent."""

    bootstrap()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_path = configure_logging(run_id)

    raw_input = sys.stdin.read()
    try:
        task_input = load_task_input(raw_input)
    except json.JSONDecodeError as exc:
        result = build_failure_result(run_id, log_path, exc)
        print(serialise_result(result))
        return

    try:
        profile = resolve_profile(task_input)
        pipeline_result, nvidia_context = run_agent(task_input, run_id)
        result = build_success_result(
            run_id, log_path, pipeline_result, profile, nvidia_context
        )
    except Exception as exc:  # pylint: disable=broad-except
        result = build_failure_result(run_id, log_path, exc)
    print(serialise_result(result))


if __name__ == "__main__":  # pragma: no cover
    main()
