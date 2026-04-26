# merge_scores.py

import json
import pandas as pd
from pathlib import Path


INPUT_JSON = Path("intermediate_batch_output/master_intermediate.json")
INPUT_CSV = Path("score_ocr_results.csv")
OUTPUT_JSON = Path("intermediate_batch_output/master_intermediate_reviewed.json")


def pick_score(row):
    # priority: manual_score > final_score
    if pd.notna(row.get("manual_score")) and str(row.get("manual_score")).strip() != "":
        return int(row["manual_score"]), "manual"
    if pd.notna(row.get("final_score")):
        return int(row["final_score"]), "auto"
    return None, "none"


def main():
    print("Loading files...")

    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    df = pd.read_csv(INPUT_CSV)

    print(f"JSON records: {len(data)}")
    print(f"CSV rows: {len(df)}")

    # สร้าง key map
    score_map = {}

    for _, row in df.iterrows():
        key = f"{row['record_id']}_{int(row['row_index'])}"
        score, source = pick_score(row)
        score_map[key] = (score, source)

    updated_count = 0

    for rec in data:
        record_id = rec.get("record_id")

        if rec.get("form_type") == "constituency":
            rows = rec.get("candidates", [])
        else:
            rows = rec.get("party_results", [])

        for row in rows:
            key = f"{record_id}_{row.get('row_index')}"

            if key in score_map:
                score, source = score_map[key]

                if score is not None:
                    row["score"] = score
                    row["score_source"] = source
                    row["score_confidence"] = 1.0
                    row["review_required"] = False
                    updated_count += 1

    OUTPUT_JSON.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("DONE MERGE")
    print("updated rows:", updated_count)
    print("output:", OUTPUT_JSON)


if __name__ == "__main__":
    main()