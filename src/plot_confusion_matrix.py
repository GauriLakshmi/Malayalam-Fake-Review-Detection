import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tfidf_linear_svm.joblib"
TEST_PATH = ROOT / "data" / "processed" / "test_final.csv"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
if not TEST_PATH.exists():
    raise FileNotFoundError(f"Test data not found at {TEST_PATH}")

# Load model and test data
model = joblib.load(MODEL_PATH)
test_df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
# Map labels to binary (Human=0, AI=1)
label_map = {0: 0, 1: 1}
test_df["label"] = test_df["label"].map(label_map).astype(int)
X_test = test_df["review_text"].astype(str).str.strip()
y_true = test_df["label"].astype(int)

# Predictions
y_pred = model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

class_names = ["Human (0)", "AI (1)"]

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap='viridis')
ax.figure.colorbar(im, ax=ax)

# Ticks
ax.set_xticks(np.arange(cm.shape[1]))
ax.set_yticks(np.arange(cm.shape[0]))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix (Test Set)')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Text annotations
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')

fig.tight_layout()
# Save image
output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'confusion_matrix.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Confusion matrix image saved to {output_path}')
