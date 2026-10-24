#!/usr/bin/env python
import pandas as pd
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
REPORT_PATH = RESULTS_DIR / "full_error_report.pdf"

# Load evaluation metrics and error analysis data
TEST_EVAL_PATH = RESULTS_DIR / "test_evaluation_results.csv"
ERROR_ANALYSIS_PATH = RESULTS_DIR / "baseline_error_analysis.csv"

if not TEST_EVAL_PATH.exists():
    raise FileNotFoundError(f"Test evaluation results not found at {TEST_EVAL_PATH}")
if not ERROR_ANALYSIS_PATH.exists():
    raise FileNotFoundError(f"Error analysis CSV not found at {ERROR_ANALYSIS_PATH}")

test_df = pd.read_csv(TEST_EVAL_PATH)
error_df = pd.read_csv(ERROR_ANALYSIS_PATH)

styles = getSampleStyleSheet()
story = []
story.append(Paragraph("<b>Malayalam Fake Review Detection – Full Error Analysis Report</b>", styles["Title"]))
story.append(Spacer(1, 12))

# Summary metrics table (extract from test_df)
summary_rows = []
for label in ["Human (0)", "AI (1)", "accuracy", "macro avg", "weighted avg"]:
    if label in test_df.index:
        row = test_df.loc[label]
        summary_rows.append([
            label,
            f"{row['precision']:.4f}" if 'precision' in row else "",
            f"{row['recall']:.4f}" if 'recall' in row else "",
            f"{row['f1-score']:.4f}" if 'f1-score' in row else "",
            int(row['support']) if 'support' in row and not pd.isna(row['support']) else ""
        ])

if summary_rows:
    table = Table([['Class', 'Precision', 'Recall', 'F1-Score', 'Support']] + summary_rows, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

# Confusion matrix (hard‑coded from previous results)
conf_mat = pd.DataFrame([[160, 71], [65, 191]], index=["Actual Human", "Actual AI"], columns=["Pred Human", "Pred AI"]).astype(int)
story.append(Paragraph("<b>Confusion Matrix (rows = actual, columns = predicted)</b>", styles["Heading2"]))
cm_table = Table([[''] + list(conf_mat.columns)] + [[idx] + list(conf_mat.loc[idx]) for idx in conf_mat.index], hAlign='LEFT')
cm_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
    ('BACKGROUND', (0,1), (0,-1), colors.lightgrey),
    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
    ('ALIGN', (1,1), (-1,-1), 'CENTER'),
]))
story.append(cm_table)
story.append(Spacer(1, 12))

# Sample misclassifications (first 5)
story.append(Paragraph("<b>Sample Misclassifications (first 5)</b>", styles["Heading2"]))
sample = error_df.head(5)
sample_data = [["Review Text", "True Label", "Predicted Label"]]
for _, row in sample.iterrows():
    txt = row['review_text']
    txt = txt[:200] + ('...' if len(txt) > 200 else '')
    sample_data.append([txt, row['true_label_name'], row['pred_label_name']])
sample_table = Table(sample_data, colWidths=[300, 80, 80], hAlign='LEFT')
sample_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
]))
story.append(sample_table)
story.append(Spacer(1, 12))

# Build PDF
SimpleDocTemplate(str(REPORT_PATH), pagesize=A4).build(story)
print(f"PDF report saved to {REPORT_PATH}")
