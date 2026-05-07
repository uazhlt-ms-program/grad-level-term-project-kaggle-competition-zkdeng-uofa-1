from pathlib import Path

import pandas as pd
import pytest

from ling539_competition.data import load_competition_data, write_submission


def write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def make_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    write_csv(
        data_dir / "train.csv",
        [
            {"ID": 1, "TEXT": "great movie", "LABEL": 1},
            {"ID": 2, "TEXT": "bad movie", "LABEL": 2},
            {"ID": 3, "TEXT": "weather report", "LABEL": 0},
        ],
    )
    write_csv(
        data_dir / "test.csv",
        [
            {"ID": 10, "TEXT": "new review"},
            {"ID": 11, "TEXT": None},
        ],
    )
    write_csv(
        data_dir / "sample_submission.csv",
        [
            {"ID": 10, "LABEL": 0},
            {"ID": 11, "LABEL": 0},
        ],
    )
    return data_dir


def test_load_competition_data_validates_and_fills_text(tmp_path):
    data = load_competition_data(make_data_dir(tmp_path))

    assert list(data.train.columns) == ["ID", "TEXT", "LABEL"]
    assert data.test.loc[1, "TEXT"] == ""


def test_load_competition_data_rejects_bad_labels(tmp_path):
    data_dir = make_data_dir(tmp_path)
    train = pd.read_csv(data_dir / "train.csv")
    train.loc[0, "LABEL"] = 9
    train.to_csv(data_dir / "train.csv", index=False)

    with pytest.raises(ValueError, match="outside"):
        load_competition_data(data_dir)


def test_load_competition_data_rejects_sample_id_mismatch(tmp_path):
    data_dir = make_data_dir(tmp_path)
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    sample.loc[0, "ID"] = 99
    sample.to_csv(data_dir / "sample_submission.csv", index=False)

    with pytest.raises(ValueError, match="do not match"):
        load_competition_data(data_dir)


def test_write_submission_uses_required_columns(tmp_path):
    data = load_competition_data(make_data_dir(tmp_path))
    output_path = write_submission(data.test, [1, 2], tmp_path / "submission.csv")
    submission = pd.read_csv(output_path)

    assert list(submission.columns) == ["ID", "LABEL"]
    assert submission.to_dict("records") == [{"ID": 10, "LABEL": 1}, {"ID": 11, "LABEL": 2}]
