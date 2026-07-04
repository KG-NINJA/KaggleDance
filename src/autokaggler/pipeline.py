"""Machine learning pipeline implementations for the Kaggle Titanic challenge."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Survived"
MODEL_PROFILES = ("fast", "tree", "boosting", "ensemble")


@dataclass
class TitanicPipelineResult:
    """Summary of a Titanic pipeline execution."""

    cv_mean: float
    cv_std: float
    model_name: str
    submission_path: str
    data_source: str
    notes: Optional[str]
    selected_profile: str = "fast"
    requested_profile: str = "fast"
    cv_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    feature_importance: List[Dict[str, float | str]] = field(default_factory=list)
    report_path: Optional[str] = None
    submit_result: Optional[Dict[str, object]] = None


class TitanicPipeline:
    """Configurable Titanic modelling pipeline."""

    def __init__(self, profile: str = "fast", random_seed: int = 42) -> None:
        self.profile = profile
        self.random_seed = random_seed
        self.numeric_features = [
            "Age",
            "Fare",
            "Pclass",
            "SibSp",
            "Parch",
            "FamilySize",
            "HasCabin",
            "AgePclass",
            "FarePerPerson",
        ]
        self.categorical_features = ["Sex", "Embarked", "Title"]
        self.feature_columns = self.numeric_features + self.categorical_features
        self.preprocessor = self._build_preprocessor()
        self.model, self._model_name = self._build_model(profile)

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        submission_name: str,
        output_dir: Path,
        notes: Optional[str],
        data_meta,
    ) -> TitanicPipelineResult:
        """Train, evaluate and export predictions for the Titanic task."""

        logging.info("Starting Titanic pipeline with profile '%s'", self.profile)
        np.random.seed(self.random_seed)
        self._validate_dataframe(train_df, is_train=True)
        self._validate_dataframe(test_df, is_train=False)

        train_features = self._add_titanic_features(train_df)
        test_features = self._add_titanic_features(test_df)
        X = train_features[self.feature_columns]
        y = train_df[TARGET_COLUMN]

        selected_profile, pipeline, scores, cv_results = self._select_pipeline(X, y)
        self.model = pipeline.named_steps["model"]
        self._model_name = self._model_display_name(selected_profile, self.model)
        logging.info("Selected profile '%s' with CV scores: %s", selected_profile, scores)

        pipeline.fit(X, y)
        logging.info("Model '%s' fitted on %d samples", self._model_name, len(train_df))

        submission_df = self._build_submission(pipeline, test_df, test_features)
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / submission_name
        submission_df.to_csv(submission_path, index=False)
        logging.info("Submission saved to %s", submission_path)

        note_lines = []
        if notes:
            note_lines.append(notes)
        if data_meta.source == "sample":
            note_lines.append("Using bundled sample dataset")
        compiled_notes = " | ".join(note_lines) if note_lines else None

        return TitanicPipelineResult(
            cv_mean=float(scores.mean()),
            cv_std=float(scores.std()),
            model_name=self._model_name,
            submission_path=str(submission_path),
            data_source=data_meta.source,
            notes=compiled_notes,
            selected_profile=selected_profile,
            requested_profile=self.profile,
            cv_results=cv_results,
            feature_columns=list(self.feature_columns),
            feature_importance=self._extract_feature_importance(pipeline),
        )

    def _select_pipeline(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[str, Pipeline, np.ndarray, Dict[str, Dict[str, float]]]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        candidates = MODEL_PROFILES if (self.profile or "fast").lower() == "auto" else ((self.profile or "fast").lower(),)
        cv_results: Dict[str, Dict[str, float]] = {}
        best_profile = "fast"
        best_score = -1.0
        best_scores: Optional[np.ndarray] = None

        for profile in candidates:
            model, _ = self._build_model(profile)
            pipeline = Pipeline(
                steps=[("preprocessor", self._build_preprocessor()), ("model", model)]
            )
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
            cv_results[profile] = {
                "mean_accuracy": float(scores.mean()),
                "std": float(scores.std()),
            }
            if float(scores.mean()) > best_score:
                best_score = float(scores.mean())
                best_profile = profile
                best_scores = scores

        if best_scores is None:  # pragma: no cover
            raise RuntimeError("No model profiles were evaluated")
        model, _ = self._build_model(best_profile)
        pipeline = Pipeline(
            steps=[("preprocessor", self._build_preprocessor()), ("model", model)]
        )
        return best_profile, pipeline, best_scores, cv_results

    def _build_submission(
        self, pipeline: Pipeline, test_df: pd.DataFrame, test_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate submission dataframe with PassengerId and predictions."""
        X_test = test_features[self.feature_columns]
        preds = pipeline.predict(X_test).astype(int)
        return pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": preds})

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        try:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # scikit-learn < 1.2
            onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", onehot),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

    def _build_model(self, profile: str):
        profile = (profile or "fast").lower()
        if profile == "auto":
            return (
                LogisticRegression(max_iter=1000, random_state=self.random_seed),
                "AutoProfileSelector",
            )
        if profile in {"fast", "linear"}:
            return (
                LogisticRegression(max_iter=1000, random_state=self.random_seed),
                "LogisticRegression",
            )
        if profile in {"tree", "power"}:
            return (
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=5,
                    min_samples_leaf=2,
                    random_state=self.random_seed,
                ),
                "RandomForestClassifier",
            )
        if profile == "boosting":
            try:
                return (
                    HistGradientBoostingClassifier(
                        max_iter=120,
                        learning_rate=0.07,
                        random_state=self.random_seed,
                    ),
                    "HistGradientBoostingClassifier",
                )
            except Exception:  # pragma: no cover
                return (
                    GradientBoostingClassifier(random_state=self.random_seed),
                    "GradientBoostingClassifier",
                )
        if profile == "ensemble":
            return (
                VotingClassifier(
                    estimators=[
                        (
                            "fast",
                            LogisticRegression(
                                max_iter=1000, random_state=self.random_seed
                            ),
                        ),
                        (
                            "tree",
                            RandomForestClassifier(
                                n_estimators=120,
                                max_depth=5,
                                min_samples_leaf=2,
                                random_state=self.random_seed,
                            ),
                        ),
                        (
                            "boosting",
                            HistGradientBoostingClassifier(
                                max_iter=120,
                                learning_rate=0.07,
                                random_state=self.random_seed,
                            ),
                        ),
                    ],
                    voting="soft",
                ),
                "VotingClassifier",
            )
        logging.warning("Unknown profile '%s'; using fast profile", profile)
        return (
            LogisticRegression(max_iter=1000, random_state=self.random_seed),
            "LogisticRegression",
        )

    def _extract_feature_importance(self, pipeline: Pipeline) -> List[Dict[str, float | str]]:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = list(self.feature_columns)

        values = None
        if hasattr(model, "feature_importances_"):
            values = getattr(model, "feature_importances_")
        elif hasattr(model, "coef_"):
            values = np.abs(getattr(model, "coef_")).ravel()
        elif hasattr(model, "estimators_"):
            importances = []
            for estimator in getattr(model, "estimators_", []):
                candidate = estimator
                if isinstance(estimator, tuple):
                    candidate = estimator[1]
                if hasattr(candidate, "feature_importances_"):
                    importances.append(getattr(candidate, "feature_importances_"))
                elif hasattr(candidate, "coef_"):
                    importances.append(np.abs(getattr(candidate, "coef_")).ravel())
            if importances:
                values = np.mean(importances, axis=0)

        if values is None:
            return []
        rows = []
        for name, value in zip(feature_names, values):
            rows.append({"feature": str(name), "importance": float(value)})
        rows.sort(key=lambda item: float(item["importance"]), reverse=True)
        return rows[:15]

    @staticmethod
    def _model_display_name(profile: str, model) -> str:
        if profile == "fast":
            return "LogisticRegression"
        if profile == "tree":
            return "RandomForestClassifier"
        if profile == "boosting":
            return type(model).__name__
        if profile == "ensemble":
            return "VotingClassifier"
        return type(model).__name__

    @staticmethod
    def _add_titanic_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add documented Titanic feature engineering columns."""
        features = df.copy()
        title = (
            features["Name"]
            .fillna("")
            .str.extract(r",\s*([^\.]+)\.", expand=False)
            .fillna("Unknown")
        )
        title = title.replace(
            {
                "Mlle": "Miss",
                "Ms": "Miss",
                "Mme": "Mrs",
                "Lady": "Rare",
                "Countess": "Rare",
                "Capt": "Rare",
                "Col": "Rare",
                "Don": "Rare",
                "Dr": "Rare",
                "Major": "Rare",
                "Rev": "Rare",
                "Sir": "Rare",
                "Jonkheer": "Rare",
                "Dona": "Rare",
            }
        )
        features["Title"] = title.where(
            title.isin({"Mr", "Mrs", "Miss", "Master", "Rare"}), "Rare"
        )
        features["FamilySize"] = (
            features["SibSp"].fillna(0) + features["Parch"].fillna(0) + 1
        )
        features["HasCabin"] = features["Cabin"].notna().astype(int)
        features["AgePclass"] = features["Age"] * features["Pclass"]
        features["FarePerPerson"] = features["Fare"] / features["FamilySize"].replace(0, 1)
        return features

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, *, is_train: bool) -> None:
        """Ensure required columns exist."""
        required_columns = [
            "PassengerId",
            "Pclass",
            "Name",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Cabin",
            "Embarked",
        ]
        if is_train:
            required_columns.append(TARGET_COLUMN)

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")


__all__ = ["TitanicPipeline", "TitanicPipelineResult", "MODEL_PROFILES"]
