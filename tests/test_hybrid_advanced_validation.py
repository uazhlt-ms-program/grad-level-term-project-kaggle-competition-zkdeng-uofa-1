import numpy as np

from ling539_competition.train_hybrid_advanced import (
    reduce_long_text,
    search_blend,
    softmax_temperature,
    validate_advanced_metrics,
)


def test_reduce_long_text_keeps_head_and_tail():
    text = " ".join(f"w{i}" for i in range(300))
    reduced = reduce_long_text(text, head_words=5, tail_words=3).split()

    assert reduced == ["w0", "w1", "w2", "w3", "w4", "w297", "w298", "w299"]


def test_reduce_long_text_leaves_short_text_alone():
    assert reduce_long_text("a short review", head_words=5, tail_words=3) == "a short review"


def test_softmax_temperature_returns_probabilities():
    probs = softmax_temperature(np.array([[1.0, 2.0, 3.0]]), temperature=1.0)

    assert probs.shape == (1, 3)
    assert np.allclose(probs.sum(axis=1), [1.0])


def test_search_blend_returns_valid_predictions():
    y_valid = np.array([0, 1, 2, 1])
    transformer_valid = np.array([[4, 1, 0], [0, 4, 1], [0, 1, 4], [0, 3, 1]], dtype=float)
    transformer_test = np.array([[3, 1, 0], [0, 0, 3]], dtype=float)
    svm_valid = np.array([[3, 0, 0], [0, 2, 1], [0, 1, 2], [0, 3, 0]], dtype=float)
    svm_test = np.array([[2, 1, 0], [0, 0, 2]], dtype=float)

    result = search_blend(
        y_valid,
        transformer_valid,
        transformer_test,
        svm_valid,
        svm_test,
        weights=np.array([0.0, 0.5, 1.0]),
        temperatures=[1.0],
        bias_values=[0.0],
    )

    assert 0 <= result.weight <= 1
    assert result.validation_predictions.shape == (4,)
    assert result.test_predictions.shape == (2,)
    assert set(result.test_predictions).issubset({0, 1, 2})


def test_validate_advanced_metrics_requires_main_fields():
    assert validate_advanced_metrics(
        {
            "model_name": "distilbert",
            "blend_validation_f1": 0.95,
            "selected_blend": {},
            "device": "mps",
        }
    )
