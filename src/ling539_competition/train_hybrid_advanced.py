from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from .data import LABELS, load_competition_data, write_submission
from .train_classic import build_classical_models, to_builtin


BASELINE_MACRO_F1 = 0.9262817722601703
BIAS_VALUES = [-0.20, -0.10, 0.0, 0.10, 0.20]
TEMPERATURES = [0.75, 1.0, 1.25, 1.5, 2.0]


@dataclass
class BlendResult:
    f1_macro: float
    weight: float
    transformer_temperature: float
    svm_temperature: float
    class_bias: list[float]
    validation_predictions: np.ndarray
    test_predictions: np.ndarray


def reduce_long_text(text: str, head_words: int = 200, tail_words: int = 80) -> str:
    words = str(text).split()
    keep_words = head_words + tail_words
    if len(words) <= keep_words:
        return " ".join(words)
    return " ".join(words[:head_words] + words[-tail_words:])


def softmax_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scores = np.exp(scaled)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def search_blend(
    y_valid: np.ndarray,
    transformer_valid_logits: np.ndarray,
    transformer_test_logits: np.ndarray,
    svm_valid_scores: np.ndarray,
    svm_test_scores: np.ndarray,
    weights: np.ndarray | None = None,
    temperatures: list[float] | None = None,
    bias_values: list[float] | None = None,
) -> BlendResult:
    weights = np.arange(0, 1.0001, 0.05) if weights is None else weights
    temperatures = TEMPERATURES if temperatures is None else temperatures
    bias_values = BIAS_VALUES if bias_values is None else bias_values
    best = None

    bias_grid = []
    for bias in itertools.product(bias_values, repeat=len(LABELS)):
        bias_array = np.array(bias, dtype=float)
        bias_grid.append(bias_array - bias_array.mean())

    for transformer_temp in temperatures:
        transformer_valid_probs = softmax_temperature(transformer_valid_logits, transformer_temp)
        transformer_test_probs = softmax_temperature(transformer_test_logits, transformer_temp)
        for svm_temp in temperatures:
            svm_valid_probs = softmax_temperature(svm_valid_scores, svm_temp)
            svm_test_probs = softmax_temperature(svm_test_scores, svm_temp)
            for weight in weights:
                valid_probs = weight * transformer_valid_probs + (1 - weight) * svm_valid_probs
                test_probs = weight * transformer_test_probs + (1 - weight) * svm_test_probs
                valid_log_probs = np.log(valid_probs + 1e-12)
                test_log_probs = np.log(test_probs + 1e-12)
                for bias in bias_grid:
                    valid_preds = (valid_log_probs + bias).argmax(axis=1)
                    score = f1_score(y_valid, valid_preds, average="macro")
                    if best is None or score > best.f1_macro:
                        best = BlendResult(
                            f1_macro=float(score),
                            weight=float(weight),
                            transformer_temperature=float(transformer_temp),
                            svm_temperature=float(svm_temp),
                            class_bias=bias.tolist(),
                            validation_predictions=valid_preds,
                            test_predictions=(test_log_probs + bias).argmax(axis=1),
                        )

    return best


def sample_stratified(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df

    parts = []
    per_label = max(2, sample_size // len(LABELS))
    for label in LABELS:
        label_rows = df[df["LABEL"] == label]
        n = min(per_label, len(label_rows))
        parts.append(label_rows.sample(n=n, random_state=seed + label))
    sampled = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled


def train_tfidf_svm(train_texts, train_labels, valid_texts, test_texts, seed: int):
    model = clone(build_classical_models(seed)["word_char_linear_svc"])
    model.fit(train_texts, train_labels)
    valid_scores = model.decision_function(valid_texts)
    test_scores = model.decision_function(test_texts)
    return model, valid_scores, test_scores


def choose_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(tokenizer, texts, labels, args, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    tensors = [encoded["input_ids"], encoded["attention_mask"]]
    if labels is not None:
        tensors.append(torch.tensor(labels, dtype=torch.long))
    return DataLoader(TensorDataset(*tensors), batch_size=args.batch_size, shuffle=shuffle)


def run_transformer(train_texts, train_labels, valid_texts, valid_labels, test_texts, args):
    import torch
    from torch.optim import AdamW
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(args.seed)
    device = choose_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = len(LABELS)
    config.id2label = {i: str(i) for i in LABELS}
    config.label2id = {str(i): i for i in LABELS}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_loader = make_loader(tokenizer, train_texts, train_labels, args, shuffle=True)
    valid_loader = make_loader(tokenizer, valid_texts, valid_labels, args, shuffle=False)
    test_loader = make_loader(tokenizer, test_texts, None, args, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_steps = max(1, update_steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = -1.0
    best_state = None
    completed_steps = 0
    started_at = time.monotonic()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss / args.gradient_accumulation
            loss.backward()

            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1

                if completed_steps % args.log_every == 0:
                    elapsed = (time.monotonic() - started_at) / 60
                    print(f"transformer step {completed_steps}/{total_steps} after {elapsed:.1f} min", flush=True)

                if args.max_steps and completed_steps >= args.max_steps:
                    break
                if (time.monotonic() - started_at) / 60 >= args.max_train_minutes:
                    break

        valid_logits = predict_logits(model, valid_loader, device)
        valid_preds = valid_logits.argmax(axis=1)
        valid_f1 = f1_score(valid_labels, valid_preds, average="macro")
        print(f"transformer epoch {epoch + 1}: {valid_f1:.4f}", flush=True)

        if valid_f1 > best_f1:
            best_f1 = valid_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if args.max_steps and completed_steps >= args.max_steps:
            break
        if (time.monotonic() - started_at) / 60 >= args.max_train_minutes:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    valid_logits = predict_logits(model, valid_loader, device)
    test_logits = predict_logits(model, test_loader, device)

    transformer_dir = Path(args.models_dir) / "advanced_model" / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(transformer_dir)
    tokenizer.save_pretrained(transformer_dir)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": str(device),
        "best_f1_macro": float(best_f1),
        "completed_steps": completed_steps,
        "validation_logits": valid_logits,
        "test_logits": test_logits,
    }


def predict_logits(model, loader, device) -> np.ndarray:
    import torch

    model.eval()
    logits = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits.append(output.logits.detach().cpu().numpy())
    return np.vstack(logits)


def build_metrics(args, data, split_info, transformer_info, svm_valid_scores, blend: BlendResult):
    y_valid = split_info["y_valid"]
    svm_valid_preds = svm_valid_scores.argmax(axis=1)
    transformer_valid_preds = transformer_info["validation_logits"].argmax(axis=1)
    report = classification_report(y_valid, blend.validation_predictions, labels=LABELS, output_dict=True, zero_division=0)

    return {
        "model_name": args.model_name,
        "device": transformer_info["device"],
        "baseline_macro_f1": BASELINE_MACRO_F1,
        "train_rows": int(len(split_info["train_idx"])),
        "validation_rows": int(len(split_info["valid_idx"])),
        "test_rows": int(len(data.test)),
        "transformer_validation_f1": float(f1_score(y_valid, transformer_valid_preds, average="macro")),
        "tfidf_svm_validation_f1": float(f1_score(y_valid, svm_valid_preds, average="macro")),
        "blend_validation_f1": blend.f1_macro,
        "target_met": bool(blend.f1_macro >= 0.94),
        "selected_blend": {
            "transformer_weight": blend.weight,
            "svm_weight": 1 - blend.weight,
            "transformer_temperature": blend.transformer_temperature,
            "svm_temperature": blend.svm_temperature,
            "class_bias": blend.class_bias,
        },
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_valid, blend.validation_predictions, labels=LABELS).tolist(),
        "transformer_completed_steps": transformer_info["completed_steps"],
        "train_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "validation_size": args.validation_size,
            "sample_size": args.sample_size,
            "log_every": args.log_every,
        },
    }


def write_advanced_outputs(args, data, metrics, blend: BlendResult, svm_model) -> None:
    reports_dir = Path(args.reports_dir)
    models_dir = Path(args.models_dir) / "advanced_model"
    submissions_dir = Path(args.submissions_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "advanced_metrics.json").write_text(json.dumps(to_builtin(metrics), indent=2))
    pd.DataFrame(metrics["classification_report"]).transpose().to_csv(reports_dir / "advanced_classification_report.csv")
    pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    ).to_csv(reports_dir / "advanced_confusion_matrix.csv")
    joblib.dump({"svm_model": svm_model, "metrics": metrics}, models_dir / "blend_bundle.joblib")
    write_submission(data.test, blend.test_predictions, submissions_dir / "advanced_submission.csv")


def validate_advanced_metrics(metrics: dict) -> bool:
    required = {"model_name", "blend_validation_f1", "selected_blend", "device"}
    return required.issubset(metrics)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--submissions-dir", default="submissions")
    parser.add_argument("--model-name", default="distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-train-minutes", type=float, default=55)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    data = load_competition_data(args.data_dir)
    train_df = sample_stratified(data.train, args.sample_size, args.seed)

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=args.validation_size,
        stratify=train_df["LABEL"],
        random_state=args.seed,
    )

    full_texts = train_df["TEXT"].to_numpy()
    y = train_df["LABEL"].to_numpy()
    test_texts = data.test["TEXT"].to_numpy()

    reduced_texts = np.array([reduce_long_text(text) for text in full_texts])
    reduced_test_texts = np.array([reduce_long_text(text) for text in test_texts])

    X_train_full = full_texts[train_idx]
    X_valid_full = full_texts[valid_idx]
    y_train = y[train_idx]
    y_valid = y[valid_idx]

    X_train_reduced = reduced_texts[train_idx]
    X_valid_reduced = reduced_texts[valid_idx]

    print("training TF-IDF SVM", flush=True)
    svm_model, svm_valid_scores, svm_test_scores = train_tfidf_svm(
        X_train_full,
        y_train,
        X_valid_full,
        test_texts,
        args.seed,
    )

    print("training transformer", flush=True)
    transformer_info = run_transformer(
        X_train_reduced,
        y_train,
        X_valid_reduced,
        y_valid,
        reduced_test_texts,
        args,
    )

    print("searching blend", flush=True)
    blend = search_blend(
        y_valid,
        transformer_info["validation_logits"],
        transformer_info["test_logits"],
        svm_valid_scores,
        svm_test_scores,
    )

    split_info = {"train_idx": train_idx, "valid_idx": valid_idx, "y_valid": y_valid}
    metrics = build_metrics(args, data, split_info, transformer_info, svm_valid_scores, blend)
    write_advanced_outputs(args, data, metrics, blend, svm_model)

    print(f"TF-IDF SVM validation macro F1: {metrics['tfidf_svm_validation_f1']:.4f}", flush=True)
    print(f"Transformer validation macro F1: {metrics['transformer_validation_f1']:.4f}", flush=True)
    print(f"Blend validation macro F1: {metrics['blend_validation_f1']:.4f}", flush=True)
    print(f"submission written to {Path(args.submissions_dir) / 'advanced_submission.csv'}", flush=True)


if __name__ == "__main__":
    main()
