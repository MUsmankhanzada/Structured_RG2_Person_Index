"""
Prototype splitter for the RG Vol. 2 Personenverzeichnis.

Turns one row per printed entry into one row per person, with the name and
its qualifiers split into columns.

THREE CHANGES IN THIS VERSION
----------------------------
1. NAME INHERITANCE IS UNCHANGED. A dash entry inherits only the FORENAME of
   the headword above it, which is what the printed running headers show:

       running header p9/10 : "Albertus Kunow - Albertus de Potenstete"
       headword             : "Albertus Hese 54."
       last entry           : "- de Potenstete (Putenstete) 1192 1339."
       header expands it as : "Albertus de Potenstete"   NOT "Albertus Hese de Potenstete"

2. HEADWORD BRACKETS ARE CLASSIFIED BEFORE BEING INHERITED. A bracket after
   the forename is only a name variant when it actually resembles the name:

       Achacius (Achatius)    -> variant, inherited by sub-entries
       Agnes (Hospes)         -> zusatz,  NOT inherited
       Affra 913 (v. Nicolaus Meltzer) -> reference, NOT inherited

3. REFERENCES. "(v. X)" is pulled out wherever it sits, and the digits around
   it stay as column numbers:

       Affra 913 (v. Nicolaus Meltzer).      -> reference X, column 913
       - 285 (v. Fredericus com. de Bichelingen). -> reference X, column 285
       - (v. Fridericus baro de Gundelfragen) 292. -> reference X, column 292

   REFERENCE AND ZUSATZ ARE NEVER INHERITED - only name variants are.

Also fixes four defects seen in the first preview:
  * "[III.] dux Austrie"        - regnal numeral was landing in familyname
  * "baro de Litaw al. de Sterenberg" - alias after an office was missed
  * "de Loden al. ten Antbade"  - alias starting with a particle was swallowed
  * "com. de Lyssenik (Lisnig)" - variant was left inline in office
"""

import difflib
import re

import pandas as pd

SIMILARITY_THRESHOLD = 0.4

OFFICE = (
    "prep.", "ep.", "aep.", "abb.", "abba", "dec.", "decan.", "can.", "com.",
    "comitissa", "dux", "ducissa", "rex", "regina", "marchio", "baro",
    "burggravius", "magistra", "precept.", "tit.", "presb.", "card.", "diac.",
    "prior", "priorissa", "custos", "scolast.", "pincerna", "vic.", "mag.",
    "el.", "papa", "imperator", "princeps", "lantgravius", "dominus",
    "monach.", "conv.", "capit.", "cantor", "thesaur.", "archidiac.",
    "archipresb.", "offic.", "cler.", "miles", "civis", "prefectus", "pp.",
)
INSTITUTION = ("eccl.", "mon.", "dom.", "hosp.", "capel.", "par.", "abbatia",
               "priorat.", "studium")
ALIAS_MARKERS = ("al.", "d.", "dict.", "sive", "seu")
PARTICLES = ("de", "van", "von", "ten", "der", "den", "di", "del", "in")

ROMAN_RE = re.compile(r"^\[?[IVXLCDM]+\.?\]?$")


def similarity(a, b):
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def looks_like_variant(content, name):
    """Is this bracket a spelling variant of `name`, or something else?"""
    if not content or not name:
        return False
    for piece in [p.strip() for p in content.split(",") if p.strip()]:
        if piece.startswith("-"):                    # suffix variant, -berg
            return True
        if any(piece.lower().startswith(m) for m in ALIAS_MARKERS):
            return False                             # (d. Vogel) is an alias
        if similarity(piece, name) >= SIMILARITY_THRESHOLD:
            return True
    return False


# ----------------------------------------------------------------------
# tokenising
# ----------------------------------------------------------------------

def tokenise(text):
    out = []
    for seg in [s for s in re.split(r"(\([^)]*\))", text) if s]:
        if seg.startswith("(") and seg.endswith(")"):
            out.append(("bracket", seg[1:-1].strip()))
        else:
            for w in seg.split():
                if w.strip():
                    out.append(("word", w.strip()))
    return out


def txt(tokens):
    return " ".join(v if t == "word" else f"({v})" for t, v in tokens).strip()


def find_office(tokens, start=0):
    """First lowercase office word. Capitalised forms are surnames."""
    for i in range(start, len(tokens)):
        t, v = tokens[i]
        if t == "word" and v[:1].islower() and v.lower().rstrip(",") in OFFICE:
            return i
    return None


def find_inst(tokens, start=0):
    for i in range(start, len(tokens)):
        t, v = tokens[i]
        if t == "word" and v.lower().rstrip(",") in INSTITUTION:
            return i
    return None


# ----------------------------------------------------------------------
# field extraction
# ----------------------------------------------------------------------

def extract_reference(text):
    """Pull out '(v. X)' anywhere, or a trailing bare 'v. X'."""
    m = re.search(r"\(\s*v\.\s*(.+?)\s*\)", text)
    if m:
        rest = (text[:m.start()] + " " + text[m.end():])
        return m.group(1).strip().rstrip("."), re.sub(r"\s+", " ", rest).strip()
    m = re.search(r"(?:^|\s)v\.\s+(.+?)\s*$", text)
    if m:
        # kept verbatim: the final dot may be the abbreviation's own
        # ("v. A. ep. Spiren." -> "A. ep. Spiren.")
        return m.group(1).strip(), text[:m.start()].strip()
    return "", text


def extract_pages(text):
    m = re.search(r"((?:\d+)(?:\s+\d+)*)\s*\.?\s*$", text)
    if m:
        return m.group(1).strip(), text[:m.start()].strip()
    # no page numbers: keep the trailing dot, it is the abbreviation's own
    # ("aep. Magunt." not "aep. Magunt")
    return "", text.strip()


def split_alias(tokens):
    """Pull an 'al.' / 'd.' byname out of a token run.

    A 'de ...' AFTER alias content is the territorial surname and goes back
    to the name; directly after the marker it belongs to the alias.
    """
    idx = next((i for i, (t, v) in enumerate(tokens)
                if t == "word" and v.lower() in ALIAS_MARKERS), None)
    if idx is None:
        return tokens, ""

    head, after = tokens[:idx], tokens[idx + 1:]

    cut = len(after)
    for j, (t, v) in enumerate(after):
        if t == "word" and v.lower() in PARTICLES and j > 0:
            cut = j
            break
    return head + after[cut:], txt(after[:cut])


def classify_brackets(tokens, out, fallback_base):
    """Sort every bracket in `tokens` into variant / alias / zusatz.

    The bracket is compared against the word IMMEDIATELY BEFORE IT, not the
    last word of the run. "Hebbonis (Hobbonis) de Catwyc" is a variant of
    Hebbonis; comparing it against Catwyc calls it a zusatz.
    """
    prev_word = ""
    for t, v in tokens:
        if t == "word":
            prev_word = v.strip(",.")
            continue
        base = prev_word or fallback_base
        # "ep. (el.) Poznan." - the bracket is a second office term (electus),
        # not a spelling of the bishop's name
        if v.lower().strip(",") in OFFICE or v.lower().strip(",") in INSTITUTION:
            out["office"] = " ".join(
                x for x in (out["office"], f"({v})") if x).strip()
            continue
        if looks_like_variant(v, base):
            out["familyname_variant"] = "; ".join(
                x for x in (out["familyname_variant"], v) if x)
        elif any(v.lower().startswith(m) for m in ALIAS_MARKERS):
            out["alias"] = "; ".join(x for x in (out["alias"], v) if x)
        else:
            out["zusatz"] = "; ".join(x for x in (out["zusatz"], v) if x)


def parse_name_body(body, own_name_for_variants=""):
    """Split the name part of an entry into its columns."""
    out = dict(firstname="", firstname_variant="", familyname="",
               familyname_variant="", alias="", office="", institution="",
               zusatz="", numeral="")

    tokens = tokenise(body)
    if not tokens:
        return out

    oi = find_office(tokens)

    # A POSTPOSED TITLE. "Cliven. et de Marka com." is "count of Cleves and
    # Mark" - the territory belongs to the office, not to a family name. The
    # giveaway is that the office word is the LAST word of the run and the
    # words before it are coordinated with "et". Without the "et" test a
    # plain surname such as "de Monte com." would be swallowed whole.
    if oi is not None:
        word_idx = [i for i, (t, _) in enumerate(tokens) if t == "word"]
        if word_idx and oi == word_idx[-1] and oi > 0:
            if any(v.lower() == "et" for t, v in tokens[:oi] if t == "word"):
                oi = 0

    name_tokens = tokens[:oi] if oi is not None else tokens
    role_tokens = tokens[oi:] if oi is not None else []

    # an alias can sit inside either part
    name_tokens, alias_a = split_alias(name_tokens)
    role_tokens, alias_b = split_alias(role_tokens)
    out["alias"] = "; ".join(x for x in (alias_a, alias_b) if x)

    # office / institution, with any bracket inside pulled out separately
    if role_tokens:
        role_words = [tk for tk in role_tokens if tk[0] == "word"]
        ii = find_inst(role_words, 1)
        if ii is None:
            out["office"] = txt(role_words)
        else:
            out["office"] = txt(role_words[:ii])
            out["institution"] = txt(role_words[ii:])
        classify_brackets(role_tokens, out, "")

    # the name itself
    words = [v for t, v in name_tokens if t == "word"]

    # a regnal numeral belongs to the forename, not the surname:
    # "Albertus [III.] dux Austrie" is Albertus III, of no family name
    lead = 0
    while lead < len(words) and ROMAN_RE.match(words[lead]):
        lead += 1
    if lead:
        out["numeral"] = " ".join(words[:lead])
        words = words[lead:]

    if words:
        out["familyname"] = " ".join(words).strip().rstrip(",")

    classify_brackets(name_tokens, out, own_name_for_variants)
    return out


def parse_headword(raw):
    """Split a headword row into forename, its variant, and the rest.

    The bracket after the forename is only inherited when it is a genuine
    spelling variant.
    """
    text = raw.strip()
    m = re.match(r"^([A-ZÄÖÜ][\wäöüß'\[\]]*)\s*(\(([^)]*)\))?\s*(.*)$", text, re.S)
    if not m:
        return text.split()[0] if text.split() else "", "", "", ""

    forename = m.group(1)
    bracket = (m.group(3) or "").strip()
    rest = m.group(4)

    variant, zusatz = "", ""
    if bracket:
        if looks_like_variant(bracket, forename):
            variant = bracket
        else:
            zusatz = bracket
    return forename, variant, zusatz, rest


def build_row(raw, after_name, is_dash, hw_first, hw_first_var):
    """Parse one printed entry into one output row.

    `after_name` is the text still to be parsed: for a dash entry that is
    everything after the dash; for a headword it is everything after the
    forename and its bracket, so the forename is not re-read as a surname.
    """
    reference, text = extract_reference(after_name)
    pages, body = extract_pages(text)

    row = dict(is_dash=int(is_dash),
               firstname=hw_first,
               firstname_variant=hw_first_var,
               familyname="", familyname_variant="", alias="",
               office="", institution="", reference=reference,
               zusatz="", column_num=pages, raw_entry=raw)

    parsed = parse_name_body(body, hw_first)
    numeral = parsed.pop("numeral", "")
    row.update(parsed)

    # the forename is always the headword's; a numeral qualifies it
    row["firstname"] = (hw_first + " " + numeral).strip() if numeral else hw_first
    row["firstname_variant"] = hw_first_var
    return row


def run(in_csv, out_csv):
    df = pd.read_csv(in_csv, encoding="utf-8-sig", dtype=str).fillna("")
    rows = []
    hw_first = hw_first_var = ""

    for _, r in df.iterrows():
        raw = r["raw_entry"].strip()
        if not raw:
            continue

        if re.match(r"^\s*[—–-]", raw):
            after = re.sub(r"^\s*[—–-]\s*", "", raw)
            row = build_row(raw, after, True, hw_first, hw_first_var)
        else:
            forename, variant, hzusatz, rest = parse_headword(raw)
            hw_first, hw_first_var = forename, variant

            row = build_row(raw, rest, False, hw_first, hw_first_var)
            # a headword bracket that was not a variant is this row's zusatz
            if hzusatz:
                row["zusatz"] = "; ".join(x for x in (hzusatz, row["zusatz"]) if x)

        row["source_image"] = r.get("source_image", "")
        row["page_column"] = r.get("column", "")
        row["index_column"] = r.get("index_column", "")
        row["entry_no"] = r.get("entry_no", "")
        rows.append(row)

    cols = ["source_image", "page_column", "index_column", "entry_no", "is_dash",
            "firstname", "firstname_variant",
            "familyname", "familyname_variant", "alias",
            "office", "institution", "reference", "zusatz",
            "column_num", "raw_entry"]
    out = pd.DataFrame(rows)[cols]
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "personen_vol2_raw_final_ordered.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "vol2_split_preview.csv"
    out = run(src, dst)
    print(f"{len(out)} rows -> {dst}")