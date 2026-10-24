import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tfidf_linear_svm.joblib"
TEST_PATH = ROOT / "data" / "processed" / "test_final.csv"
ERRORS_OUTPUT = ROOT / "results" / "baseline_error_analysis.csv"

# Map original labels to binary (Human=0, AI=1). Include AI label=2 mapping.
LABEL_MAP = {0: 0, 1: 1, 2: 1}
CLASS_NAMES = ["Human (0)", "AI (1)"]

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run 'python src/train_baseline.py' first.")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found at {TEST_PATH}. Run 'python src/split_final_dataset.py' first.")

    # Load model and test data
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    df["label"] = df["label"].map(LABEL_MAP).astype(int)
    X_test = df["review_text"].astype(str).str.strip()
    y_true = df["label"].astype(int)

    # Predictions
    y_pred = model.predict(X_test)

    # Overall metrics
    acc = (y_pred == y_true).mean()
    print(f"Overall accuracy: {acc * 100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    # Identify misclassifications
    mis_idx = (y_pred != y_true)
    mis_df = df.loc[mis_idx, ["review_text", "label"]].copy()
    mis_df["pred_label"] = y_pred[mis_idx]
    # Replace numeric labels with readable names
    label_to_name = {0: "Human", 1: "AI"}
    mis_df["true_label_name"] = mis_df["label"].map(label_to_name)
    mis_df["pred_label_name"] = mis_df["pred_label"].map(label_to_name)
    # Reorder columns
    mis_df = mis_df[["review_text", "true_label_name", "pred_label_name"]]
    # Save errors to CSV
    mis_df.to_csv(ERRORS_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Saved {len(mis_df)} misclassified samples to {ERRORS_OUTPUT}")

if __name__ == "__main__":
    main()
