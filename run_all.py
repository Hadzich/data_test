#!/usr/bin/env python3
# run_all.py

from task_1      import tag_sentences
from task_1_v2          import tag_sentences_v2
from task_2   import tag_sentences_ml

# ─── configure your inputs ──────────────────────────────────
SENTENCES_FILE = "data/sentences.txt"
TAGS_FILE      = "data/tags.csv"

# ─── (optional) common per-tag thresholds for ML task ───────
PER_TAG_THRESHOLDS = {
    "Marital Status": 0.97,
    "Online Accounts": 0.4,
    "Personal Loans":   0.5,
    "Home Loans":       0.55,
    "Vehicle Loan":     0.85,
}

def main():
    # Task 1 exact‐match this is more strct matching as described in task 1
    tag_sentences(
        sentences_path=SENTENCES_FILE,
        tags_csv_path=TAGS_FILE,
        output_tsv_path="task_1_output.tsv"
    )
    print("✅ task_1_output.tsv complete")

    # Task 1 this is more advance version
    tag_sentences_v2(
        sentences_path=SENTENCES_FILE,
        tags_csv_path=TAGS_FILE,
        output_tsv_path="task_1_output_v2.tsv"
    )
    print("✅ task_1_output_v2.tsv complete")

    # Task 2 ML
    tag_sentences_ml(
        sentences_path=SENTENCES_FILE,
        tags_csv_path=TAGS_FILE,
        output_path="task_2_output.tsv",
        semantic_threshold     = 0.7,
        fuzzy_single_threshold = 90,
        fuzzy_multi_threshold  = 85,
        min_fuzzy_token_len    = 5,
        per_tag_thresholds     = PER_TAG_THRESHOLDS
    )
    print("✅ task_2_output.tsv complete")

if __name__ == "__main__":
    main()
