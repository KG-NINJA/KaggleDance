"""Markdown report generation for Autokaggler runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .pipeline import TitanicPipelineResult


def write_titanic_mastery_report(
    *,
    result: TitanicPipelineResult,
    run_id: str,
    report_dir: Path,
    elapsed_seconds: float,
    competition: str,
    submit_status: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a concise achievement report for the current run."""

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "titanic-mastery-report.md"
    cv_rows = []
    for profile, metrics in sorted(result.cv_results.items()):
        cv_rows.append(
            f"| `{profile}` | {metrics['mean_accuracy']:.4f} | {metrics['std']:.4f} |"
        )
    if not cv_rows:
        cv_rows.append(f"| `{result.selected_profile}` | {result.cv_mean:.4f} | {result.cv_std:.4f} |")

    importance_rows = []
    for row in result.feature_importance[:10]:
        importance_rows.append(f"| `{row['feature']}` | {float(row['importance']):.6f} |")
    if not importance_rows:
        importance_rows.append("| n/a | Model does not expose direct importance |")

    submit_status = submit_status or {"status": "not_requested"}
    generated_at = datetime.now(timezone.utc).isoformat()
    body = f"""# Titanic Mastery Achieved

Run ID: `{run_id}`  
Generated: `{generated_at}`  
Competition: `{competition}`  
Elapsed seconds: `{elapsed_seconds:.2f}`

## Achievement

KaggleDance completed the Titanic automation loop locally: data loading, feature engineering, model selection, cross-validation, training, and `submission.csv` generation.

This is a practical capability proof: the same pattern maps to other table-data problems where an agent must acquire data, build features, compare models, produce an artifact, and explain the result.

## Selected Model

- Requested profile: `{result.requested_profile}`
- Selected profile: `{result.selected_profile}`
- Model: `{result.model_name}`
- CV accuracy mean: `{result.cv_mean:.4f}`
- CV accuracy std: `{result.cv_std:.4f}`
- Data source: `{result.data_source}`
- Submission: `{result.submission_path}`

## CV Comparison

| Profile | Mean accuracy | Std |
| --- | ---: | ---: |
{chr(10).join(cv_rows)}

## Engineered Features

{', '.join(f'`{name}`' for name in result.feature_columns)}

## Feature Signal

| Feature | Importance |
| --- | ---: |
{chr(10).join(importance_rows)}

## Kaggle Submit Status

```json
{_json_block(submit_status)}
```

## Code Path

- `src/autokaggler/agent.py`: JSON stdin orchestration and safety gates
- `src/autokaggler/data_manager.py`: Kaggle download or synthetic sample fallback
- `src/autokaggler/pipeline.py`: Titanic features, model profiles, auto selection, submission generation
- `src/autokaggler/reporting.py`: this achievement report

## Capability Statement

Titanic Mastery is not just a tutorial pass. It demonstrates a repeatable autonomous workflow for structured data problems: inspect the task, obtain usable data, construct predictive features, select a model, create a deliverable, and report the evidence.
"""
    report_path.write_text(body, encoding="utf-8")
    return report_path


def _json_block(payload: Dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = ["write_titanic_mastery_report"]
