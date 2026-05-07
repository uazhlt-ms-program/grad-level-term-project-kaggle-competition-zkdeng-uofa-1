# LING 539 Class Competition

This repository contains my code for the Spring 2026 LING 539 Kaggle class competition.

The task is a three-way text classification problem:

- `0`: not a movie or TV review
- `1`: positive movie or TV review
- `2`: negative movie or TV review

Kaggle evaluates submissions with macro F1, so the model selection code uses macro F1 on local validation data.

## Final Submission

The final submitted method is a hybrid of:

- a word and character n-gram TF-IDF `LinearSVC`
- a fine-tuned `distilbert-base-uncased-finetuned-sst-2-english`
- a validation-tuned blend of SVM decision scores and DistilBERT logits

The `LinearSVC` component is the course-covered classification method in the final model. DistilBERT is used as an additional pretrained text model to improve sentiment classification.

Current results:

| model | validation macro F1 |
| --- | ---: |
| TF-IDF `LinearSVC` | 0.9252 |
| DistilBERT | 0.9324 |
| Hybrid blend | 0.9468 |

Kaggle public score for `advanced_submission.csv`: `0.94900`.

## Setup

Install [pixi](https://pixi.sh/), then run:

```bash
pixi install
```

Download the Kaggle data:

```bash
pixi run download-data
```

This expects Kaggle credentials at:

```text
~/.kaggle/kaggle.json
```

Expected data layout:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

## Training

Run the final hybrid method:

```bash
pixi run train-hybrid-advanced
```

The command writes:

```text
reports/advanced_metrics.json
reports/advanced_classification_report.csv
reports/advanced_confusion_matrix.csv
models/advanced_model/
submissions/advanced_submission.csv
```

For a quick pipeline check:

```bash
pixi run train-hybrid-advanced-smoke
```

The classical baseline can be reproduced with:

```bash
pixi run train-classic
```

## Submit

Submit the final hybrid file:

```bash
pixi run submit
```

Equivalent Kaggle command:

```bash
kaggle competitions submit \
  -c ling-539-competition-2026 \
  -f submissions/advanced_submission.csv \
  -m "advanced DistilBERT TF-IDF LinearSVC blend"
```

## Tests

```bash
pixi run test
```

The tests cover data validation, submission shape, long-text reduction, probability normalization, blend search, and advanced metrics structure.

## Notes

Generated data, model checkpoints, and submission files are ignored by Git. The code can rebuild the submission from the Kaggle data files.
