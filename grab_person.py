"""
grab_person_vol2.py

Raw transcription of the Repertorium Germanicum Vol. 2 Personenverzeichnis.

Reads double-column scans, splits each page into its two printed columns, and
sends each column to a vision LLM. Output is ONE ROW PER PRINTED ENTRY, kept
verbatim - no field splitting, no expansion, no correction. All structuring
happens in later scripts.

Entry model
-----------
A printed entry is either a headword line or a line beginning with a dash:

    Abraham Johannis 37 542 963.        <- headword entry
    - prep. eccl. Lancicien. 1157.      <- dash entry (its own person)
    - de Monaco 1432.                   <- dash entry (its own person)

The dash means "repeat the headword forename" (Abraham). It is kept VERBATIM
here and expanded deterministically downstream, because expansion is derivable
from position and every synthesised token is a token a model can get wrong.

Two kinds of line continuation must be distinguished:

  wrap            indented, previous line does NOT end in '-'  -> join with ONE space
  word division   previous line ends in '-'                    -> join with NO space,
                                                                   drop the hyphen

    - d. Nemersze de Novacuria 156 229      ->  - d. Nemersze de Novacuria 156 229 542.
          542.

    Achacius (Achatius) Frederici de Fyer-  ->  Achacius (Achatius) Frederici de
          naw al. de Zydendorf 37.              Fyernaw al. de Zydendorf 37.

Usage
-----
    python grab_person_vol2.py
"""

import base64
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, RateLimitError


# ============================================================
# API CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("ACADEMIC_CLOUD_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("ACADEMIC_CLOUD_BASE_URL", "https://chat-ai.academiccloud.de/v1")
MODEL = os.getenv("MODEL_NAME", "qwen3.8-27b")

if not API_KEY:
    raise ValueError(
        "API key not found. Set 'ACADEMIC_CLOUD_API_KEY' or 'OPENAI_API_KEY' "
        "in your .env file or environment."
    )

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=180.0)


# ------------------------------------------------------------
# Thinking / reasoning control
# ------------------------------------------------------------
# Verbatim transcription is the one task where reasoning HURTS: a thinking
# model will "helpfully" regularise historical spellings the prompt forbids it
# to touch, and a plausible normalisation is far harder to catch downstream
# than an obvious misread.
#
# Each family takes its own key in its own place, and only Mistral rejects
# wrong arguments - the rest ignore them silently. So the switch is built per
# family rather than sent blindly.

def thinking_kwargs(model_name):
    """Return the request kwargs that turn thinking OFF for this model."""
    name = model_name.lower()

    # longest family prefix wins: qwen3.8 is not a qwen3
    if name.startswith("qwen3.8"):
        return {
            "reasoning_effort": "low",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    if name.startswith("deepseek-v4"):
        return {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
    if name.startswith(("qwen3", "gemma-4", "glm-4", "glm-5")):
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if name.startswith("mistral-"):
        return {"reasoning_effort": "none"}
    if name.startswith("openai-gpt-oss"):
        return {"reasoning_effort": "low"}     # no off level exists
    return {}                                   # apertus, llama, devstral


THINKING_KWARGS = thinking_kwargs(MODEL)


# ============================================================
# PATHS AND IMAGE RANGE
# ============================================================

# Scans live one level up from the experiments folder this script runs in:
#   W:\ZZ_HisQu\RG Volumes\Volume 2\JPEG            <- images
#   W:\ZZ_HisQu\RG Volumes\Volume 2\JPEG\experiments <- this script
# ".." keeps it portable; set an absolute path instead if you prefer.
IMAGE_DIR = os.getenv("RG_IMAGE_DIR", "..")

# Files are named like:  He 6481 (2,2_3_0014.jpg
START_PAGE = 14
END_PAGE = 167


# ============================================================
# FILES
# ============================================================

CHECKPOINT_CSV = "personen_vol2_raw_checkpoint.csv"
FINAL_CSV = "personen_vol2_raw_final.csv"


# ============================================================
# SPEED CONFIGURATION
# ============================================================

MAX_WORKERS = 4
REQUEST_DELAY = 0.5
MAX_RETRIES = 5


# ============================================================
# IMAGE HANDLING
# ============================================================

def encode_and_resize(img_array, max_width=1600, jpeg_quality=90):
    """Encode a column image as base64 JPEG.

    Wider and higher quality than the Vol. 3 settings: this index sets long
    runs of 3-4 digit column numbers in a small face, and downscaling too far
    is where digit misreads come from.
    """
    h, w = img_array.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img_array = cv2.resize(
            img_array, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    ok, buffer = cv2.imencode(
        ".jpg", img_array, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    if not ok:
        raise RuntimeError("Could not encode image.")
    return base64.b64encode(buffer).decode("utf-8")


def detect_body_top(gray):
    """Y of the horizontal rule under the running header, +1 line of padding.

    Measured across the samples the rule sits at 5.4-6.5% of page height, but
    the very first page of the index has no running header and therefore no
    rule, so a fallback is needed.
    """
    h, w = gray.shape
    frac = (gray < 200).mean(axis=1)
    band = range(int(h * 0.03), int(h * 0.16))
    y = max(band, key=lambda i: frac[i])
    if frac[y] > 0.25:                 # a real full-width rule
        return y + int(h * 0.006)
    return int(h * 0.05)               # no header rule on this page


def detect_split_x(gray):
    """X of the gutter between the two printed columns.

    A fixed w//2 is WRONG here: the scans are not consistently centred and the
    true gutter ranges from 45.8% to 52.5% of page width across the samples.
    On page 0015 the midpoint falls inside the left column and clips the ends
    of its lines. The gutter is found as the widest near-empty vertical band
    in the central third; the printed rule sits inside it.
    """
    h, w = gray.shape
    frac = (gray < 160).mean(axis=h and 0)
    lo, hi = int(w * 0.35), int(w * 0.65)

    best_len, best_end, run = 0, None, 0
    for x in range(lo, hi):
        if frac[x] < 0.02:
            run += 1
            if run > best_len:
                best_len, best_end = run, x
        else:
            run = 0

    if best_end is None or best_len < 8:
        return w // 2
    return best_end - best_len // 2


def split_columns(image_path):
    """Crop header/footer, then split into left and right printed columns."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    top = detect_body_top(gray)
    split = detect_split_x(gray)

    h = img.shape[0]
    body = img[top: int(h * 0.97), :]
    return body[:, :split], body[:, split:]


def header_strip(image_path):
    """The top band, holding '<left col>  Headword - Headword  <right col>'."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img[: detect_body_top(gray), :]


def get_image_files(start_num=START_PAGE, end_num=END_PAGE):
    """Find 'He_6481__2_2_3_0014.jpg' style scans in the working directory."""
    valid = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    # matches "He 6481 (2,2_3_0014" and "He_6481__2_2_3_0014" alike
    pattern = re.compile(r"He[\s_]*6481.*?(\d+)\s*$", re.IGNORECASE)

    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(
            f"Image directory not found: {os.path.abspath(IMAGE_DIR)}"
        )

    matched = []
    for fname in os.listdir(IMAGE_DIR):
        if not fname.lower().endswith(valid):
            continue
        stem = fname.rsplit(".", 1)[0]
        m = pattern.search(stem)
        if m and start_num <= int(m.group(1)) <= end_num:
            matched.append((int(m.group(1)), os.path.join(IMAGE_DIR, fname)))

    matched.sort(key=lambda x: x[0])
    return [f for _, f in matched]


# ============================================================
# PROMPTS
# ============================================================

PERSON_PROMPT = r"""
Transcribe this column from a historical PERSON INDEX (Personenverzeichnis)
into a JSON list of entries.

WHAT COUNTS AS ONE ENTRY

1. A line starting at the normal left margin begins a new entry.

2. A line starting with a dash ("-" or "—") is ALSO its own separate entry.
   The dash stands for the repeated headword name. KEEP THE DASH. Do NOT
   replace it with the name it stands for. Do not merge dash lines into the
   entry above.

3. Return each entry as exactly ONE string.

JOINING CONTINUATION LINES

4. If an entry runs over several physical lines, join them into one string.

5. If a line ends with a hyphen "-", the word is split across the line break.
   Join with NO space and DELETE the hyphen.
       "Frederici de Fyer-" + "naw al. de Zydendorf 37."
       -> "Frederici de Fyernaw al. de Zydendorf 37."

6. Otherwise join continuation lines with exactly ONE space.
       "d. Nemersze de Novacuria 156 229" + "542."
       -> "d. Nemersze de Novacuria 156 229 542."

TRANSCRIBE EXACTLY AS PRINTED

7. Do NOT modernise, correct, normalise or expand anything. Historical and
   inconsistent spellings must be preserved character for character.

8. Preserve every abbreviation exactly: ep. aep. abb. prep. can. com. dux
   tit. card. mon. eccl. al. d. v. cf. etc.

9. Preserve ALL punctuation: periods, commas, parentheses, square brackets,
   hyphens inside words. Square brackets are editorial and are part of the
   text: "[III.] dux Austrie", "Gra(n)den.", "Culm[u]ach".

10. Preserve every number exactly, in the printed order, separated by single
    spaces. Do not merge, reorder, drop or invent numbers.

11. Preserve German characters: Ä Ö Ü ä ö ü ß.

12. Ignore bold or spaced type. Transcribe bold numbers as plain numbers.

13. Do not invent text. If something is genuinely illegible, transcribe what
    is visible and nothing more.

IGNORE THESE

14. The running header line at the top of the page.

15. Page/column numbers printed in the top corners.

16. Single-letter or short alphabetical section dividers standing alone on
    their own line, such as "A", "B", "C Ch K". NOTE: these can appear in the
    MIDDLE of a column, not only at the top. Skip the divider itself but keep
    transcribing the entries that follow it.

OUTPUT

Return ONLY valid JSON, no commentary and no markdown fences:

{
    "entries": [
        "Entry 1",
        "Entry 2"
    ]
}
"""

HEADER_PROMPT = r"""
This is the top header strip of a two-page spread from a printed index.
It looks like:   7    Agnes - Albertus de Chuden    8

Report the number printed in the far LEFT corner and the number printed in the
far RIGHT corner. These are column numbers. If a number is missing or
unreadable, use null.

Return ONLY valid JSON, no commentary and no markdown fences:

{"left": 7, "right": 8}
"""


# ============================================================
# API CALLS
# ============================================================

def strip_fences(text):
    """Remove ```json ... ``` wrappers if the model adds them anyway."""
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


def call_model(prompt, b64_img, label):
    """One vision call with retry/backoff. Returns parsed JSON or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_DELAY)
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    ],
                }],
                temperature=0.0,
                **THINKING_KWARGS,
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(strip_fences(raw))

        except RateLimitError:
            wait = 15 * attempt
            print(f"[RATE LIMIT] {label} - waiting {wait}s")
            time.sleep(wait)

        except (APIStatusError, json.JSONDecodeError, Exception) as exc:
            if attempt >= MAX_RETRIES:
                print(f"[FAILED] {label}: {exc}")
                return None
            wait = 3 * attempt
            print(f"[ERROR] {label} attempt {attempt}/{MAX_RETRIES} "
                  f"-> retry in {wait}s ({exc})")
            time.sleep(wait)

    return None


def extract_column(column_img, image_name, column_name):
    """Transcribe one printed column."""
    label = f"{image_name} [{column_name}]"
    data = call_model(PERSON_PROMPT, encode_and_resize(column_img), label)

    if data is None:
        return {"success": False, "image": image_name, "column": column_name,
                "entries": [], "error": "call failed"}

    entries = data.get("entries", [])
    if not isinstance(entries, list):
        return {"success": False, "image": image_name, "column": column_name,
                "entries": [], "error": "'entries' is not a list"}

    cleaned = []
    for entry in entries:
        if not isinstance(entry, str):
            continue
        entry = entry.strip().strip('"').strip()
        if entry:
            cleaned.append(entry)

    return {"success": True, "image": image_name, "column": column_name,
            "entries": cleaned, "error": None}


def read_header_columns(image_path):
    """Read the printed column numbers from the header strip."""
    try:
        strip = header_strip(image_path)
    except FileNotFoundError:
        return None, None

    data = call_model(HEADER_PROMPT, encode_and_resize(strip, max_width=1600),
                      f"{image_path} [header]")
    if not isinstance(data, dict):
        return None, None
    return data.get("left"), data.get("right")


# ============================================================
# CHECKPOINTING
# ============================================================

COLUMNS = ["source_image", "column", "index_column", "entry_no", "raw_entry"]


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_CSV):
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(CHECKPOINT_CSV, encoding="utf-8-sig", dtype=str)
        if not {"source_image", "column", "raw_entry"}.issubset(df.columns):
            print("Checkpoint format invalid - starting fresh.")
            return pd.DataFrame(columns=COLUMNS)
        if "index_column" not in df.columns:
            df["index_column"] = ""
        if "entry_no" not in df.columns:
            # older checkpoint: rebuild position from the stored row order,
            # which is correct WITHIN a column even though the column blocks
            # themselves were written in completion order.
            df["entry_no"] = df.groupby(
                ["source_image", "column"], sort=False
            ).cumcount() + 1
        return df
    except Exception as exc:
        print(f"Could not read checkpoint: {exc}")
        return pd.DataFrame(columns=COLUMNS)


def already_processed(df, image_name, column_name):
    if df.empty:
        return False
    return ((df["source_image"] == image_name) & (df["column"] == column_name)).any()


def add_results(df, image_name, column_name, index_column, entries):
    """Append rows. A column with zero entries still gets one blank marker row
    so an empty column is not re-processed on every run."""
    if entries:
        rows = pd.DataFrame({
            "source_image": [image_name] * len(entries),
            "column": [column_name] * len(entries),
            "index_column": [index_column if index_column is not None else ""] * len(entries),
            # position of the entry inside its printed column, 1-based
            "entry_no": list(range(1, len(entries) + 1)),
            "raw_entry": entries,
        })
    else:
        rows = pd.DataFrame([{
            "source_image": image_name, "column": column_name,
            "index_column": index_column if index_column is not None else "",
            "entry_no": 1, "raw_entry": "",
        }])
    return pd.concat([df, rows], ignore_index=True)


def save_checkpoint(df):
    df.to_csv(CHECKPOINT_CSV, index=False, encoding="utf-8-sig")


# ============================================================
# FINAL EXPORT
# ============================================================

MOJIBAKE_MARKERS = ("Ã", "â€", "Â", "Å")


def fix_mojibake(val):
    """UTF-8 bytes decoded as cp1252. cp1252 first, NOT latin-1: 'â€”' holds
    '€', which latin-1 cannot encode, so a latin-1 round-trip silently fails
    on every em-dash."""
    if not isinstance(val, str) or not any(m in val for m in MOJIBAKE_MARKERS):
        return val
    for enc in ("cp1252", "latin1"):
        try:
            return val.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return val


PAGE_NO_RE = re.compile(r"He[\s_]*6481.*?(\d+)\s*$", re.IGNORECASE)


def page_number(path):
    """Scan number from the file name, for sorting."""
    stem = os.path.basename(str(path)).rsplit(".", 1)[0]
    m = PAGE_NO_RE.search(stem)
    return int(m.group(1)) if m else 10**9


def build_final(df):
    """Clean, then restore PRINTED READING ORDER.

    Rows are appended as columns finish, and ThreadPoolExecutor completes them
    out of order, so the checkpoint holds correct sequences inside each column
    block but the blocks themselves are shuffled. Sorting by
    (page, left-before-right, position) puts the volume back in reading order.
    """
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)

    out = df.copy()
    out["raw_entry"] = out["raw_entry"].astype(str).map(fix_mojibake)
    out["raw_entry"] = out["raw_entry"].str.strip().str.strip('"').str.strip()
    out = out[out["raw_entry"].notna()
              & (out["raw_entry"] != "")
              & (out["raw_entry"] != "nan")]

    out["_page"] = out["source_image"].map(page_number)
    out["_col"] = out["column"].map({"left": 0, "right": 1}).fillna(2).astype(int)
    out["_pos"] = pd.to_numeric(out["entry_no"], errors="coerce").fillna(0).astype(int)

    out = out.sort_values(["_page", "_col", "_pos"], kind="stable")
    out = out.drop(columns=["_page", "_col", "_pos"])
    return out.reset_index(drop=True)


def save_final(df):
    final = build_final(df)
    final.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig",
                 quoting=csv.QUOTE_ALL)
    return final


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()

    print("=" * 70)
    print("PERSONENVERZEICHNIS VOL. 2 - RAW EXTRACTION")
    print("=" * 70)
    print(f"Image folder    : {os.path.abspath(IMAGE_DIR)}")
    print(f"Pages           : {START_PAGE} -> {END_PAGE}")
    print(f"Model           : {MODEL}")
    print(f"Thinking kwargs : {THINKING_KWARGS or 'none for this family'}")
    print(f"Parallel workers: {MAX_WORKERS}")

    images = get_image_files()
    print(f"\nFound {len(images)} images.")
    expected = END_PAGE - START_PAGE + 1
    if len(images) != expected:
        print(f"WARNING: expected {expected} images, found {len(images)}.")
    if not images:
        return

    checkpoint_df = load_checkpoint()
    print(f"Checkpoint rows : {len(checkpoint_df)}")

    tasks, skipped = [], 0
    header_cache = {}

    for img_path in images:
        try:
            left_col, right_col = split_columns(img_path)
        except Exception as exc:
            print(f"Could not read {img_path}: {exc}")
            continue

        for column_name, column_img in (("left", left_col), ("right", right_col)):
            if already_processed(checkpoint_df, img_path, column_name):
                skipped += 1
            else:
                tasks.append((img_path, column_name, column_img))

    print(f"\nAlready processed: {skipped} columns")
    print(f"Remaining        : {len(tasks)} columns")

    if not tasks:
        final = save_final(checkpoint_df)
        print(f"\nEverything already processed. Final entries: {len(final)}")
        return

    # printed column numbers, one header call per page, run in parallel -
    # serially this is 150+ round trips before transcription even starts
    pages_needed = sorted({t[0] for t in tasks})
    print(f"\nReading column numbers from {len(pages_needed)} page headers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        header_futures = {
            executor.submit(read_header_columns, page): page
            for page in pages_needed
        }
        done = 0
        for future in as_completed(header_futures):
            page = header_futures[future]
            try:
                header_cache[page] = future.result()
            except Exception as exc:
                print(f"  [header failed] {page}: {exc}")
                header_cache[page] = (None, None)
            done += 1
            if done % 25 == 0 or done == len(pages_needed):
                print(f"  {done}/{len(pages_needed)} headers read")

    print(f"\nStarting {len(tasks)} transcription calls "
          f"with {MAX_WORKERS} workers...\n")

    completed = failed = 0
    failures = []
    total_tasks = len(tasks)
    run_started = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(extract_column, img, name, col): (name, col)
            for name, col, img in tasks
        }

        for future in as_completed(future_map):
            image_name, column_name = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                failed += 1
                failures.append((image_name, column_name, str(exc)))
                print(f"[WORKER ERROR] {image_name} [{column_name}]: {exc}")
                continue

            if not result["success"]:
                failed += 1
                failures.append((image_name, column_name, result["error"]))
                print(f"[FAILED] {image_name} [{column_name}]: {result['error']}")
                continue

            left_no, right_no = header_cache.get(image_name, (None, None))
            index_column = left_no if column_name == "left" else right_no

            checkpoint_df = add_results(
                checkpoint_df, image_name, column_name,
                index_column, result["entries"],
            )
            save_checkpoint(checkpoint_df)

            completed += 1
            elapsed = time.time() - run_started
            rate = elapsed / max(completed, 1)
            remaining = (total_tasks - completed - failed) * rate / 60
            print(f"[{completed}/{total_tasks}] {os.path.basename(image_name)} "
                  f"[{column_name}] col {index_column} -> "
                  f"{len(result['entries'])} entries  (~{remaining:.0f} min left)")

    print("\n" + "=" * 70)
    print("CREATING FINAL CSV")
    print("=" * 70)

    final = save_final(checkpoint_df)
    minutes = (time.time() - start_time) / 60

    print(f"\nCompleted columns: {completed}")
    print(f"Failed columns   : {failed}")
    if failures:
        print("\nFAILED COLUMNS - re-run the script to retry only these:")
        for image_name, column_name, err in failures:
            print(f"   {os.path.basename(image_name)} [{column_name}]: {err}")
    # NB: computed outside the f-string - backslashes are not allowed inside
    # f-string expressions before Python 3.12.
    dash_pattern = r"^\s*[-\u2013\u2014]"
    dash_count = int(final["raw_entry"].str.match(dash_pattern).sum()) if len(final) else 0

    print(f"Total entries    : {len(final)}")
    print(f"Dash entries     : {dash_count}")
    print(f"Time elapsed     : {minutes:.2f} minutes")
    print(f"\nFinal CSV     : {os.path.abspath(FINAL_CSV)}")
    print(f"Checkpoint CSV: {os.path.abspath(CHECKPOINT_CSV)}")

    print("\nFirst 15 entries:")
    print(final.head(15).to_string(index=False))

    print("\n" + "=" * 70)
    print("FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()