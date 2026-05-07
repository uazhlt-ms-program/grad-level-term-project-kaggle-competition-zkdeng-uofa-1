from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from .data import LABELS, load_competition_data, write_submission


def build_classical_models(seed: int) -> dict[str, Pipeline]:
    word_features = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=200_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    return {
        "word_logreg": Pipeline(
            [
                ("features", clone(word_features)),
                (
                    "clf",
                    OneVsRestClassifier(
                        LogisticRegression(
                            C=4.0,
                            max_iter=2000,
                            class_weight="balanced",
                            solver="liblinear",
                            random_state=seed,
                        ),
                    ),
                ),
            ]
        ),
        "word_linear_svc": Pipeline(
            [
                ("features", clone(word_features)),
                ("clf", LinearSVC(C=0.75, class_weight="balanced", random_state=seed)),
            ]
        ),
        "word_char_linear_svc": Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", clone(word_features)),
                            (
                                "char",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    min_df=2,
                                    max_features=150_000,
                                    sublinear_tf=True,
                                ),
                            ),
                        ]
                    ),
                ),
                ("clf", LinearSVC(C=0.75, class_weight="balanced", random_state=seed)),
            ]
        ),
        "word_complement_nb": Pipeline(
            [
                ("features", clone(word_features)),
                ("clf", ComplementNB(alpha=0.25)),
            ]
        ),
    }


def evaluate_estimator(name: str, estimator, X, y, cv: StratifiedKFold) -> dict:
    oof = np.empty(len(y), dtype=int)
    scores = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[valid_idx])
        oof[valid_idx] = preds
        scores.append(f1_score(y[valid_idx], preds, average="macro"))
        print(f"{name} fold {fold}: {scores[-1]:.4f}", flush=True)

    return summarize_predictions(name, y, oof, scores)


def summarize_predictions(name: str, y_true, y_pred, scores: list[float]) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)

    return {
        "name": name,
        "fold_scores": scores,
        "mean_f1_macro": float(np.mean(scores)),
        "std_f1_macro": float(np.std(scores)),
        "report": report,
        "confusion_matrix": matrix.tolist(),
    }


def choose_final_model(classical: dict) -> dict:
    return {"kind": "classical", "reason": "best classical model selected", **classical}


def train_final_model(selected: dict, classical_models: dict, data):
    X_train = data.train["TEXT"].to_numpy()
    y = data.train["LABEL"].to_numpy()
    X_test = data.test["TEXT"].to_numpy()

    model = clone(classical_models[selected["name"]])
    model.fit(X_train, y)
    preds = model.predict(X_test)
    bundle = {"kind": "classical", "model": model, "selected": selected}

    return bundle, preds


def write_outputs(args, data, classical_results, selected, bundle, preds) -> None:
    reports_dir = Path(args.reports_dir)
    models_dir = Path(args.models_dir)
    submissions_dir = Path(args.submissions_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "model": result["name"],
            "mean_f1_macro": result["mean_f1_macro"],
            "std_f1_macro": result["std_f1_macro"],
            "fold_scores": " ".join(f"{score:.5f}" for score in result["fold_scores"]),
        }
        for result in classical_results
    ]
    pd.DataFrame(rows).sort_values("mean_f1_macro", ascending=False).to_csv(
        reports_dir / "model_results.csv",
        index=False,
    )
    pd.DataFrame(
        selected["confusion_matrix"],
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    ).to_csv(reports_dir / "confusion_matrix.csv")
    pd.DataFrame(selected["report"]).transpose().to_csv(reports_dir / "classification_report.csv")

    metrics = {
        "mode": "classical",
        "selected": selected,
        "classical_results": classical_results,
    }
    (reports_dir / "selected_metrics.json").write_text(json.dumps(to_builtin(metrics), indent=2))

    joblib.dump(bundle, models_dir / "final_model.joblib")
    write_submission(data.test, preds, submissions_dir / "submission.csv")

    print(f"selected model: {selected['name']} ({selected['kind']})", flush=True)
    print(f"validation macro F1: {selected['mean_f1_macro']:.4f}", flush=True)
    print(f"submission written to {submissions_dir / 'submission.csv'}", flush=True)


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--submissions-dir", default="submissions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_competition_data(args.data_dir)

    X = data.train["TEXT"].to_numpy()
    y = data.train["LABEL"].to_numpy()
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    classical_models = build_classical_models(args.seed)
    classical_results = []
    for name, model in classical_models.items():
        print(f"\nEvaluating {name}", flush=True)
        classical_results.append(evaluate_estimator(name, model, X, y, cv))

    best_classical = max(classical_results, key=lambda result: result["mean_f1_macro"])
    selected = choose_final_model(best_classical)
    bundle, preds = train_final_model(selected, classical_models, data)
    write_outputs(args, data, classical_results, selected, bundle, preds)


if __name__ == "__main__":
    main()
