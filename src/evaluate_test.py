"""
Script to evaluate the trained TF-IDF + Linear SVM baseline model
on the unseen, held-out test dataset (test_final.csv).
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tfidf_linear_svm.joblib"
TEST_PATH = ROOT / "data" / "processed" / "test_final.csv"
REPORT_OUTPUT = ROOT / "results" / "test_evaluation_results.csv"

CLASS_NAMES = ["Human (0)", "AI (1)"]

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run 'python src/train_baseline.py' first.")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found at {TEST_PATH}. Run 'python src/split_final_dataset.py' first.")

    print("=" * 65)
    print("Evaluating TF-IDF Model on Held-out Test Set (test_final.csv)")
    print("=" * 65)

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")

    # Map original labels to binary (Human=0, AI=1). Include AI label=2 mapping.
    label_map = {0: 0, 1: 1, 2: 1}
    test_df["label"] = test_df["label"].map(label_map).astype(int)
    X_test = test_df["review_text"].astype(str).str.strip()
    y_test = test_df["label"].astype(int)

    print(f"Total Test Samples: {len(test_df)}")
    print(f"Test Label Distribution:")
    print(y_test.value_counts().sort_index().rename(index={0: '0 (Human)', 1: '1 (AI)'}))

    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 65)
    print(f"OVERALL TEST ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 65)

    print("\nDetailed Classification Report:")
    report_str = classification_report(y_test, y_pred, labels=[0,1], target_names=CLASS_NAMES, digits=4)
    print(report_str)

    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    print("Confusion Matrix:")
    print("Confusion Matrix (rows: actual, cols: predicted):")
    print(cm)

    # Save to CSV
    report_dict = classification_report(y_test, y_pred, labels=[0,1], target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(REPORT_OUTPUT, encoding="utf-8-sig")
    print(f"\nSaved evaluation metrics to: {REPORT_OUTPUT}")

if __name__ == "__main__":
    main()
