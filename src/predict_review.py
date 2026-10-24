"""
Utility script to test and predict individual Malayalam reviews
using the trained TF-IDF + Linear SVM model.
"""

from pathlib import Path
import sys
import joblib
import numpy as np

# Ensure Malayalam text displays correctly in the terminal.
if sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding.lower().replace("-", "") != "utf8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tfidf_linear_svm.joblib"

LABEL_MAP = {
    0: "Human",
    1: "AI",
}


def predict(review_text: str):
    """Predict the class of one Malayalam review."""

    review_text = review_text.strip()

    if not review_text:
        raise ValueError("Review text cannot be empty. Please enter a review.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please check that the trained model file exists."
        )

    model = joblib.load(MODEL_PATH)
    prediction = model.predict([review_text])[0]
    decision_scores = model.decision_function([review_text])[0]

    print("\n" + "=" * 60)
    print(f'Review: "{review_text}"')
    # Binary classification output
    binary_label = "Human" if prediction in (0, 1) else "AI"
    print(f"Predicted Class: {prediction} -> {binary_label}")
    # Extract decision scores depending on model type (binary or multiclass)
    if hasattr(decision_scores, "__len__") and len(decision_scores) >= 2:
        # Multiclass (original 3‑class) – combine human scores (class 0 & 1) and AI (class 2)
        human_score = decision_scores[0] + decision_scores[1]
        ai_score = decision_scores[2]
        scores = [human_score.round(3), ai_score.round(3)]
    else:
        # Binary model – decision_scores is a scalar distance to hyperplane
        human_score = decision_scores
        ai_score = -decision_scores
        scores = [human_score.round(3), ai_score.round(3)]
    print("Decision scores (Human, AI):", scores)
    print("Note: Decision scores are not probabilities.")
    print("=" * 60 + "\n")

    return prediction


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter a Malayalam review: ")

    try:
        predict(text)
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}")
        sys.exit(1)