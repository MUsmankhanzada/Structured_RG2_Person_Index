"""
reorder_raw.py

Restores printed reading order in a raw extraction CSV.

Why it is needed
----------------
grab_person_vol2.py appends rows as each column finishes, and
ThreadPoolExecutor completes work out of submission order. The result is that
the sequence INSIDE each column block is correct, but the blocks themselves
appear shuffled:

    page 0014 left
    page 0015 right     <- out of order
    page 0016 right
    page 0014 right
    ...

This script sorts by (page number, left before right, position within column).
Position is taken from 'entry_no' when present; otherwise it is rebuilt from
the stored row order, which is reliable because only the blocks are shuffled,
never the rows inside them.

Run it from the folder holding the CSV:

    python reorder_raw.py
    python reorder_raw.py some_other_file.csv
"""

import os
import re
import sys

import pandas as pd

INPUT_CSV = "personen_vol2_raw_final.csv"
OUTPUT_CSV = "personen_vol2_raw_final_ordered.csv"

PAGE_NO_RE = re.compile(r"He[\s_]*6481.*?(\d+)\s*$", re.IGNORECASE)


def page_number(path):
    stem = os.path.basename(str(path)).rsplit(".", 1)[0]
    m = PAGE_NO_RE.search(stem)
    return int(m.group(1)) if m else 10**9


def reorder(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    if not os.path.exists(input_csv):
        print(f"Not found: {os.path.abspath(input_csv)}")
        return

    df = pd.read_csv(input_csv, encoding="utf-8-sig", dtype=str).fillna("")
    print(f"Read {len(df)} rows from {input_csv}")

    before = df[["source_image", "column"]].drop_duplicates()
    print("\nBlock order BEFORE:")
    for _, r in before.iterrows():
        print(f"   page {page_number(r['source_image']):>4}  {r['column']}")

    if "entry_no" in df.columns and df["entry_no"].str.strip().ne("").all():
        pos = pd.to_numeric(df["entry_no"], errors="coerce").fillna(0).astype(int)
    else:
        # rebuild from stored order: correct within each block
        pos = df.groupby(["source_image", "column"], sort=False).cumcount() + 1
        df["entry_no"] = pos

    df["_page"] = df["source_image"].map(page_number)
    df["_col"] = df["column"].map({"left": 0, "right": 1}).fillna(2).astype(int)
    df["_pos"] = pos

    df = df.sort_values(["_page", "_col", "_pos"], kind="stable")
    df = df.drop(columns=["_page", "_col", "_pos"]).reset_index(drop=True)

    after = df[["source_image", "column"]].drop_duplicates()
    print("\nBlock order AFTER:")
    for _, r in after.iterrows():
        print(f"   page {page_number(r['source_image']):>4}  {r['column']}")

    cols = [c for c in ["source_image", "column", "index_column",
                        "entry_no", "raw_entry"] if c in df.columns]
    df = df[cols + [c for c in df.columns if c not in cols]]

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(df)} rows to {os.path.abspath(output_csv)}")

    print("\nFirst 5 entries in reading order:")
    for v in df["raw_entry"].head(5):
        print("   ", v)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else INPUT_CSV
    out = (target.rsplit(".", 1)[0] + "_ordered.csv") if len(sys.argv) > 1 else OUTPUT_CSV
    reorder(target, out)