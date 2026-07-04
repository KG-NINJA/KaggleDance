"""Tests for the AutoKaggler Titanic pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from autokaggler.agent import build_success_result, load_task_input, main
from autokaggler.kaggle_submit import SUBMIT_CONFIRMATION, submit_to_kaggle_if_requested
from autokaggler.data_manager import DataManager
from autokaggler.nvidia_adapter import NvidiaKaggleAdapter
from autokaggler.pipeline import TitanicPipeline, TitanicPipelineResult


EXPECTED_MODELS = {
    "fast": "LogisticRegression",
    "tree": "RandomForestClassifier",
    "boosting": "HistGradientBoostingClassifier",
    "ensemble": "VotingClassifier",
}


def test_agent_payment_metadata_is_safe_discovery_only():
    payment_path = Path(__file__).resolve().parents[1] / "agent-payment.json"
    payload = json.loads(payment_path.read_text(encoding="utf-8"))

    assert payload["usage_policy"]["human_manual_use"]["payment_required"] is False
    assert payload["usage_policy"]["human_manual_use"]["price"] == "free"
    assert payload["usage_policy"]["ai_agent_reuse_or_adaptation"]["payment_required"] is True
    assert payload["usage_policy"]["ai_agent_reuse_or_adaptation"]["payment_via"] == "KGSTACK x402"
    assert payload["payment"]["provider"] == "KGSTACK"
    assert payload["payment"]["protocol"] == "x402"
    assert payload["payment"]["routes"]["paid_endpoint"] == "/agent-visibility-report"
    assert payload["payment"]["pay_to"] == "0x4D7d842536De9Eb491AE2300126B3CDdE7B0aDE3"
    assert payload["payment"]["network"] == "eip155:8453"
    assert payload["payment"]["asset"]["contract"] == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
    assert payload["payment"]["amount"]["atomic_units"] == "10000"
    assert payload["payment"]["safety"]["repository_contains_public_pay_to_address"] is True
    assert payload["payment"]["safety"]["repository_contains_tokens"] is False
    assert payload["payment"]["safety"]["repository_contains_settlement_logic"] is False
    assert payload["payment"]["safety"]["human_manual_use_payment_required"] is False


def test_json_input_parsing_preserves_known_and_extra_fields():
    task_input = load_task_input(
        '{"profile":"tree","data_source":"sample","random_seed":7,'
        '"nvidia_mode":"research","custom":true}'
    )

    assert task_input.profile == "tree"
    assert task_input.data_source == "sample"
    assert task_input.random_seed == 7
    assert task_input.nvidia_mode == "research"
    assert task_input.extra == {"custom": True}


def test_json_input_uses_task_defaults_for_missing_fields():
    task_input = load_task_input('{"profile":"fast"}')

    assert task_input.competition == "titanic"
    assert task_input.data_source == "auto"
    assert task_input.nvidia_mode == "off"
    assert task_input.nvidia_dry_run is True
    assert task_input.report is True
    assert task_input.submit is False


def test_nvidia_adapter_reports_dry_run_without_external_command():
    context = NvidiaKaggleAdapter(command=None).prepare(
        mode="research",
        competition="titanic",
        profile="fast",
        dry_run=True,
    )

    assert context.enabled is True
    assert context.backend == "nvidia-kaggle"
    assert context.status == "dry_run"
    assert context.details["submission_upload"] is False
    assert context.details["dataset_upload"] is False


def test_sample_dataset_fallback_for_auto_source(tmp_path):
    manager = DataManager(cache_dir=tmp_path)
    train_df, test_df, meta = manager.prepare_datasets(prefer_source="auto")

    assert meta.source == "sample"
    assert not train_df.empty
    assert not test_df.empty


def test_documented_titanic_features_are_created():
    manager = DataManager(cache_dir=Path(".agent_tmp/test-features"))
    train_df, _, _ = manager.prepare_datasets(prefer_source="sample")
    features = TitanicPipeline._add_titanic_features(train_df)

    assert features.loc[0, "Title"] == "Mr"
    assert features.loc[0, "FamilySize"] == 2
    assert features.loc[1, "HasCabin"] == 1
    assert features.loc[0, "AgePclass"] == 66
    assert features.loc[0, "FarePerPerson"] == 3.625


def test_each_model_profile_runs_on_sample_data(tmp_path):
    manager = DataManager(cache_dir=tmp_path)
    train_df, test_df, meta = manager.prepare_datasets(prefer_source="sample")

    for profile, expected_model in EXPECTED_MODELS.items():
        pipeline = TitanicPipeline(profile=profile, random_seed=7)
        result = pipeline.run(
            train_df=train_df,
            test_df=test_df,
            submission_name=f"{profile}.csv",
            output_dir=manager.submission_dir,
            notes="pytest",
            data_meta=meta,
        )

        assert 0.0 <= result.cv_mean <= 1.0
        assert result.model_name == expected_model
        assert Path(result.submission_path).exists()


def test_auto_profile_selects_model_and_records_cv_results(tmp_path):
    manager = DataManager(cache_dir=tmp_path)
    train_df, test_df, meta = manager.prepare_datasets(prefer_source="sample")
    pipeline = TitanicPipeline(profile="auto", random_seed=7)
    result = pipeline.run(
        train_df=train_df,
        test_df=test_df,
        submission_name="auto.csv",
        output_dir=manager.submission_dir,
        notes="pytest",
        data_meta=meta,
    )

    assert result.requested_profile == "auto"
    assert result.selected_profile in EXPECTED_MODELS
    assert set(result.cv_results) == set(EXPECTED_MODELS)
    assert Path(result.submission_path).exists()


def test_submit_gate_blocks_without_explicit_confirmation(tmp_path):
    path = tmp_path / "submission.csv"
    pd.DataFrame({"PassengerId": [1, 2], "Survived": [0, 1]}).to_csv(path, index=False)

    result = submit_to_kaggle_if_requested(
        submit=True,
        confirm_submit=None,
        competition="titanic",
        submission_path=path,
        expected_rows=2,
        message="pytest",
        log_dir=tmp_path,
    )

    assert result["submitted"] is False
    assert result["status"] == "blocked"
    assert result["safety_gate"] == SUBMIT_CONFIRMATION


def test_submission_csv_generation(tmp_path):
    manager = DataManager(cache_dir=tmp_path)
    train_df, test_df, meta = manager.prepare_datasets(prefer_source="sample")
    pipeline = TitanicPipeline(profile="fast", random_seed=7)
    result = pipeline.run(
        train_df=train_df,
        test_df=test_df,
        submission_name="submission.csv",
        output_dir=manager.submission_dir,
        notes=None,
        data_meta=meta,
    )

    submission = pd.read_csv(result.submission_path)
    assert list(submission.columns) == ["PassengerId", "Survived"]
    assert len(submission) == len(test_df)
    assert set(submission["Survived"]).issubset({0, 1})


def test_module_main_accepts_stdin_json_and_writes_submission(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.stdin",
        type(
            "Stdin",
            (),
            {
                "read": lambda self: (
                    '{"profile":"auto","data_source":"sample",'
                    '"nvidia_mode":"research","nvidia_dry_run":true,'
                    '"submit":false,"report":true}'
                )
            },
        )(),
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["meta"]["profile"] == "auto"
    assert payload["result"]["nvidia_kaggle"]["status"] == "dry_run"
    assert payload["result"]["nvidia_kaggle"]["competition"] == "titanic"
    assert payload["result"]["nvidia_kaggle"]["details"]["dry_run"] is True
    assert payload["result"]["achievement"] == "Titanic Mastery Achieved"
    assert payload["result"]["selected_profile"] in EXPECTED_MODELS
    assert Path(payload["result"]["submission_path"]).exists()
    assert Path(payload["result"]["report_path"]).exists()
    assert payload["result"]["submit_result"]["status"] == "dry_run"


def test_success_result_contains_required_metadata(tmp_path):
    dummy_result = TitanicPipelineResult(
        cv_mean=0.5,
        cv_std=0.1,
        model_name="LogisticRegression",
        selected_profile="fast",
        requested_profile="fast",
        submission_path=str(tmp_path / "submission.csv"),
        data_source="sample",
        notes=None,
    )
    agent_result = build_success_result(
        run_id="test-run",
        log_path=tmp_path / "log.txt",
        result=dummy_result,
        profile="fast",
    )
    assert agent_result.ok is True
    assert "#KGNINJA" in agent_result.meta["tags"]
