# Trains and evaluates the baseline TF-IDF + Linear SVM model.
from pathlib import Path
import warnings

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TRAIN_PATH = DATA_DIR / "train_final.csv"
VAL_PATH = DATA_DIR / "validation_final.csv"

TEXT_COL = "review_text"
LABEL_COL = "label"
LABELS = [0, 1]
CLASS_NAMES = ["Human", "AI"]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path):
    df = pd.read_csv(path)

    required = {TEXT_COL, LABEL_COL}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    if df[[TEXT_COL, LABEL_COL]].isna().any().any():
        raise ValueError(f"{path.name} contains missing text or labels.")

    # Clean text column
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    # Ensure label column is integer
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="raise").astype(int)

    # Map any non-binary AI label (e.g., 2) to binary label 1 (AI)
    df[LABEL_COL] = df[LABEL_COL].replace({2: 1})

    df = df[df[TEXT_COL] != ""].copy()

    # After mapping, the only valid labels should be 0 (Human) and 1 (AI)
    unexpected = set(df[LABEL_COL].unique()) - set(LABELS)
    if unexpected:
        raise ValueError(f"Unexpected labels in {path.name} after mapping: {unexpected}")

    if set(df[LABEL_COL].unique()) != set(LABELS):
        warnings.warn(f"{path.name} does not contain both required classes 0 and 1.")

    return df


def main():
    train_df = load_dataset(TRAIN_PATH)
    val_df = load_dataset(VAL_PATH)

    # Check for exact review overlap between training and validation.
    train_texts = set(train_df[TEXT_COL].str.strip())
    val_texts = set(val_df[TEXT_COL].str.strip())
    overlap = train_texts & val_texts

    print("Training rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("\nTraining label counts:")
    print(train_df[LABEL_COL].value_counts().sort_index())
    print("\nValidation label counts:")
    print(val_df[LABEL_COL].value_counts().sort_index())
    print("\nExact text overlaps:", len(overlap))

    if overlap:
        warnings.warn(
            "Exact reviews occur in both splits. Investigate before "
            "trusting validation metrics."
        )

    X_train = train_df[TEXT_COL]
    y_train = train_df[LABEL_COL]
    X_val = val_df[TEXT_COL]
    y_val = val_df[LABEL_COL]

    # Controlled baseline experiments.
    configs = [
        {
            "name": "char_2_5_C1",
            "analyzer": "char",
            "ngram_range": (2, 5),
            "C": 1.0,
        },
        {
            "name": "char_2_5_C0.5",
            "analyzer": "char",
            "ngram_range": (2, 5),
            "C": 0.5,
        },
        {
            "name": "char_2_5_C2",
            "analyzer": "char",
            "ngram_range": (2, 5),
            "C": 2.0,
        },
        {
            "name": "word_1_2_C1",
            "analyzer": "word",
            "ngram_range": (1, 2),
            "C": 1.0,
        },
    ]

    experiment_results = []
    best_model = None
    best_config = None
    best_score = -1.0

    for config in configs:
        print(f"\nTraining: {config['name']}")

        model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=config["analyzer"],
                    ngram_range=config["ngram_range"],
                    sublinear_tf=True,
                    max_features=200000,
                    min_df=1,
                ),
            ),
            (
                "svm",
                LinearSVC(
                    C=config["C"],
                    class_weight="balanced",
                    random_state=42,
                    max_iter=10000,
                ),
            ),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        report = classification_report(
            y_val,
            predictions,
            labels=LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(
            classification_report(
                y_val,
                predictions,
                labels=LABELS,
                target_names=CLASS_NAMES,
                zero_division=0,
            )
        )

        experiment_results.append({
            "configuration": config["name"],
            "analyzer": config["analyzer"],
            "ngram_range": str(config["ngram_range"]),
            "C": config["C"],
            "class_weight": "balanced",
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "human_f1": report["Human"]["f1-score"],
            "ai_f1": report["AI"]["f1-score"],
            "ai_recall": report["AI"]["recall"],
        })

        # Select using validation macro F1 only.
        if macro_f1 > best_score:
            best_score = macro_f1
            best_model = model
            best_config = config

    results_df = pd.DataFrame(experiment_results)
    results_df = results_df.sort_values(
        "macro_f1", ascending=False
    ).reset_index(drop=True)

    results_df.to_csv(
        RESULTS_DIR / "baseline_model_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_path = MODEL_DIR / "tfidf_linear_svm.joblib"
    joblib.dump(best_model, model_path)

    print("\nBest configuration:", best_config["name"])
    print(f"Best validation macro F1: {best_score:.4f}")
    print("Saved model:", model_path)
    print(
        "Saved results:",
        RESULTS_DIR / "baseline_model_performance.csv",
    )


if __name__ == "__main__":
    main()
