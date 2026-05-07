from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LABELS = [0, 1, 2]
TRAIN_COLUMNS = {"ID", "TEXT", "LABEL"}
TEST_COLUMNS = {"ID", "TEXT"}
SAMPLE_COLUMNS = {"ID", "LABEL"}


@dataclass
class CompetitionData:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


def load_competition_data(data_dir: str | Path = "data/raw") -> CompetitionData:
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"

    missing_files = [path for path in [train_path, test_path, sample_path] if not path.exists()]
    if missing_files:
        names = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing data file(s): {names}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    _check_columns(train, TRAIN_COLUMNS, "train.csv")
    _check_columns(test, TEST_COLUMNS, "test.csv")
    _check_columns(sample, SAMPLE_COLUMNS, "sample_submission.csv")
    _check_labels(train)
    _check_ids(test, sample)

    train = train.copy()
    test = test.copy()
    train["TEXT"] = train["TEXT"].fillna("").astype(str)
    test["TEXT"] = test["TEXT"].fillna("").astype(str)
    train["LABEL"] = train["LABEL"].astype(int)

    return CompetitionData(train=train, test=test, sample_submission=sample)


def write_submission(test: pd.DataFrame, labels, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({"ID": test["ID"], "LABEL": labels})
    if list(submission.columns) != ["ID", "LABEL"]:
        raise ValueError("Submission must contain exactly ID and LABEL columns")
    if not set(submission["LABEL"]).issubset(set(LABELS)):
        raise ValueError("Submission contains a label outside {0, 1, 2}")

    submission.to_csv(output_path, index=False)
    return output_path


def _check_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")


def _check_labels(train: pd.DataFrame) -> None:
    labels = set(train["LABEL"].dropna().astype(int))
    bad_labels = sorted(labels - set(LABELS))
    if bad_labels:
        raise ValueError(f"train.csv contains labels outside {{0, 1, 2}}: {bad_labels}")


def _check_ids(test: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(test["ID"]) != list(sample["ID"]):
        raise ValueError("test.csv IDs do not match sample_submission.csv IDs in the same order")
