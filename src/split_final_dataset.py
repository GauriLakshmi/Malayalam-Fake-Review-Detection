"""
Script to create leakage-safe stratified splits (train, validation, test)
for the final 3-class Malayalam Fake Review dataset.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "final_dataset.csv"

TRAIN_PATH = ROOT / "data" / "processed" / "train_final.csv"
VAL_PATH = ROOT / "data" / "processed" / "validation_final.csv"
TEST_PATH = ROOT / "data" / "processed" / "test_final.csv"

SPLIT_SUMMARY_PATH = ROOT / "results" / "final_split_summary.csv"

RANDOM_STATE = 42

def main():
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    print(f"Loaded final dataset: {len(df)} rows.")

    # Create binary_label for stratification: map label 2 (AI) to 1, keep 0 as Human
    df["binary_label"] = df["label"].replace({2: 1})

    # 1. Stratified Train (70%) vs Temp (30% for val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["binary_label"]
    )

    # 2. Split Temp into Validation (15%) and Test (15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,  # 0.5 * 30% = 15% total
        random_state=RANDOM_STATE,
        stratify=temp_df["binary_label"]
    )

    # test_df assigned above

    print(f"Train split: {len(train_df)} rows")
    print(f"Validation split: {len(val_df)} rows")
    print(f"Test split: {len(test_df)} rows")

    # Verification of Data Leakage (Overlap check between review texts across splits)
    train_texts = set(train_df["review_text"])
    val_texts = set(val_df["review_text"])
    test_texts = set(test_df["review_text"])

    tv_overlap = len(train_texts.intersection(val_texts))
    tt_overlap = len(train_texts.intersection(test_texts))
    vt_overlap = len(val_texts.intersection(test_texts))

    print(f"\n--- Leakage Verification ---")
    print(f"Train vs Val text overlap: {tv_overlap}")
    print(f"Train vs Test text overlap: {tt_overlap}")
    print(f"Val vs Test text overlap: {vt_overlap}")

    if tv_overlap + tt_overlap + vt_overlap > 0:
        # Remove overlapping rows from validation and test sets to avoid leakage
        val_df = val_df[~val_df["review_text"].isin(train_texts)]
        # Update validation text set after removal
        val_texts = set(val_df["review_text"])
        # Remove test rows that appear in training or validation
        combined_texts = train_texts.union(val_texts)
        test_df = test_df[~test_df["review_text"].isin(combined_texts)]

        # Recalculate overlaps for reporting
        tv_overlap = len(train_texts.intersection(val_texts))
        tt_overlap = len(train_texts.intersection(set(test_df["review_text"])) )
        vt_overlap = len(val_texts.intersection(set(test_df["review_text"])) )
        print("After removing overlaps:")
        print(f"Train vs Val text overlap: {tv_overlap}")
        print(f"Train vs Test text overlap: {tt_overlap}")
        print(f"Val vs Test text overlap: {vt_overlap}")

        # Ensure validation set contains a minimum number of AI examples (binary_label == 1)
        MIN_AI = 50
        val_ai_count = (val_df["binary_label"] == 1).sum()
        if val_ai_count < MIN_AI:
            needed = MIN_AI - val_ai_count
            # Sample AI rows from training set based on original label == 2
            train_ai = train_df[train_df["label"] == 2]
            move_df = train_ai.sample(n=min(needed, len(train_ai)), random_state=RANDOM_STATE)
            train_df = train_df.drop(move_df.index)
            # Ensure binary_label column added to moved rows
            move_df["binary_label"] = 1
            val_df = pd.concat([val_df, move_df], ignore_index=True)
            print(f"Moved {len(move_df)} AI rows from train to validation to ensure minimum AI examples.")




        # Ensure test set also contains a minimum number of AI examples (binary_label == 1)
        MIN_AI = 50
        test_ai_count = (test_df["binary_label"] == 1).sum()
        if test_ai_count < MIN_AI:
            needed = MIN_AI - test_ai_count
            # First try to pull from training set where original label == 2
            train_ai = train_df[train_df["label"] == 2]
            available = len(train_ai)
            if available >= needed:
                move_df = train_ai.sample(n=needed, random_state=RANDOM_STATE)
                train_df = train_df.drop(move_df.index)
                move_df["binary_label"] = 1
                test_df = pd.concat([test_df, move_df], ignore_index=True)
                print(f"Moved {len(move_df)} AI rows from train to test to ensure minimum AI examples.")
            else:
                # Take all from train and then supplement from validation if possible
                move_df_train = train_ai
                train_df = train_df.drop(move_df_train.index)
                move_df_train["binary_label"] = 1
                remaining = needed - len(move_df_train)
                move_df_val = pd.DataFrame()
                if remaining > 0:
                    val_ai = val_df[val_df["label"] == 2]
                    take = min(remaining, len(val_ai))
                    if take > 0:
                        move_df_val = val_ai.sample(n=take, random_state=RANDOM_STATE)
                        move_df_val["binary_label"] = 1
                        val_df = val_df.drop(move_df_val.index)
                test_df = pd.concat([test_df, move_df_train, move_df_val], ignore_index=True)
                print(f"Moved {len(move_df_train)} AI rows from train and {len(move_df_val)} from validation to test.")
# Save final split files
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    val_df.to_csv(VAL_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")

    print(f"\nSaved split files to {TRAIN_PATH.parent}")

    # Build summary table across splits and classes
    summary_list = []
    for split_name, split_data in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        human_count = (split_data["binary_label"] == 0).sum()
        ai_count = (split_data["binary_label"] == 1).sum()
        summary_list.append({
            "Split": split_name,
            "Total Rows": len(split_data),
            "Human (0)": int(human_count),
            "AI (1)": int(ai_count)
        })

    summary_df = pd.DataFrame(summary_list)
    SPLIT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SPLIT_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print("\n--- Split Class Breakdown ---")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
