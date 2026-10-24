# Baseline Model Training Report

## Project
Malayalam Fake Review Detection System

## Objective
Train and compare TF-IDF with Linear Support Vector Machine (Linear SVM) configurations for three-class review classification:
- `0`: Human Real
- `1`: Human Fake
- `2`: AI Fake (AI Generated)

## Dataset
- **Training samples:** 2,074 (Human Real: 985, Human Fake: 987, AI Generated: 102)
- **Validation samples:** 519 (Human Real: 247, Human Fake: 247, AI Generated: 25)
- **Test samples:** 649 (Human Real: 308, Human Fake: 309, AI Generated: 32)
- **Training, validation, and test text overlap:** 0 (Verified zero data leakage)

## Experiments
Four configurations were evaluated on the validation set using Macro F1:
1. Character TF-IDF, n-grams (2, 5), C = 0.5 (Balanced)
2. Character TF-IDF, n-grams (2, 5), C = 1.0 (Balanced)
3. Character TF-IDF, n-grams (2, 5), C = 2.0 (Balanced)
4. Word TF-IDF, n-grams (1, 2), C = 1.0 (Balanced)

## Validation Results

| Configuration | Validation Accuracy | Validation Macro F1 | AI Fake F1 | AI Fake Precision | AI Fake Recall |
|---|---:|---:|---:|---:|---:|
| **Character (2–5), C = 0.5** | **71.87%** | **75.45%** | **83.72%** | **100.00%** | **72.00%** |
| Character (2–5), C = 1.0 | 71.48% | 75.19% | 83.72% | 100.00% | 72.00% |
| Character (2–5), C = 2.0 | 71.10% | 74.90% | 83.72% | 100.00% | 72.00% |
| Word (1–2), C = 1.0 | 64.16% | 59.78% | 50.00% | 81.82% | 36.00% |

## Selected Baseline
- **Selected Model:** Character-level TF-IDF (2–5 n-grams) with Linear SVM (`C = 0.5`, `class_weight='balanced'`).
- **Validation Macro F1:** 75.45%
- **Validation Accuracy:** 71.87%

## Independent Test Set Evaluation (Unseen 649 Reviews)
Evaluated on `data/processed/test_final.csv`:
- **Overall Test Accuracy:** **69.18%**
- **Test Macro F1:** **74.33%**
- **AI Fake Precision:** **96.15%**
- **AI Fake Recall:** **78.12%**
- **AI Fake F1:** **86.21%**

## Saved Artifacts
- **Model:** `models/tfidf_linear_svm.joblib`
- **Validation Metrics:** `results/baseline_model_performance.csv`
- **Test Evaluation Metrics:** `results/test_evaluation_results.csv`
- **Training Script:** `src/train_baseline.py`
- **Evaluation Script:** `src/evaluate_test.py`
- **Prediction CLI:** `src/predict_review.py`
