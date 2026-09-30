"""Prototype splitter for the Vol. 2 Personenverzeichnis - for review only."""

import re
import pandas as pd

OFFICE = (
    "prep.", "ep.", "aep.", "abb.", "abba", "dec.", "decan.", "can.", "com.",
    "comitissa", "dux", "ducissa", "rex", "regina", "marchio", "baro",
    "burggravius", "magistra", "precept.", "tit.", "presb.", "card.", "diac.",
    "prior", "priorissa", "custos", "scolast.", "pincerna", "vic.", "mag.",
    "el.", "papa", "imperator", "princeps", "lantgravius", "dominus",
    "monach.", "conv.", "capit.", "cantor", "thesaur.", "archidiac.",
    "archipresb.", "offic.", "cler.", "miles", "civis", "prefectus",
)
INSTITUTION = ("eccl.", "mon.", "dom.", "hosp.", "capel.", "par.", "abbatia",
               "priorat.", "studium")
ALIAS = ("d.", "al.")
PARTICLES = ("de", "van", "von", "ten", "in")


def longest_first(seq):
    return sorted(seq, key=lambda s: (-len(s.split()), -len(s)))


OFFICE_S = longest_first(OFFICE)
INST_S = longest_first(INSTITUTION)


def tokenise(text):
    """('word', w) / ('bracket', content), keeping order."""
    out = []
    for seg in [s for s in re.split(r"(\([^)]*\))", text) if s]:
        if seg.startswith("(") and seg.endswith(")"):
            out.append(("bracket", seg[1:-1].strip()))
        else:
            for w in seg.split():
                w = w.strip()
                if w:
                    out.append(("word", w))
    return out


def find_office(tokens, start=0):
    for i in range(start, len(tokens)):
        t, v = tokens[i]
        if t != "word":
            continue
        if v.lower().rstrip(",") in OFFICE and v[:1].islower():
            return i
    return None


def find_inst(tokens, start=0):
    for i in range(start, len(tokens)):
        t, v = tokens[i]
        if t == "word" and v.lower().rstrip(",") in INSTITUTION:
            return i
    return None


def txt(tokens):
    return " ".join(v if t == "word" else f"({v})" for t, v in tokens).strip()


def extract_pages(text):
    """Trailing run of column numbers."""
    m = re.search(r"((?:\d+)(?:\s+\d+)*)\s*\.?\s*$", text)
    if m:
        return m.group(1).strip(), text[: m.start()].strip()
    return "", text.rstrip(".").strip()


def extract_reference(text):
    """(v. X) anywhere, or a bare 'v. X' tail."""
    m = re.search(r"\(\s*v\.\s*(.+?)\s*\)", text)
    if m:
        return m.group(1).strip(), (text[: m.start()] + " " + text[m.end():]).strip()
    m = re.search(r"\bv\.\s+(.+?)\.?\s*$", text)
    if m:
        return m.group(1).strip().rstrip("."), text[: m.start()].strip()
    return "", text


def parse_body(body):
    """Split the part of an entry that follows the forename."""
    out = dict(familyname="", familyname_variant="", alias="",
               office="", institution="", zusatz="")
    tokens = tokenise(body)
    if not tokens:
        return out

    # alias: d. / al. introduce a byname
    alias_i = next((i for i, (t, v) in enumerate(tokens)
                    if t == "word" and v.lower() in ALIAS), None)

    oi = find_office(tokens)

    if oi is not None:
        name_tokens, role_tokens = tokens[:oi], tokens[oi:]
    else:
        name_tokens, role_tokens = tokens, []

    if role_tokens:
        ii = find_inst(role_tokens, 1)
        if ii is None:
            out["office"] = txt(role_tokens)
        else:
            out["office"] = txt(role_tokens[:ii])
            out["institution"] = txt(role_tokens[ii:])

    if alias_i is not None and (oi is None or alias_i < oi):
        before = name_tokens[:alias_i]
        after = name_tokens[alias_i + 1:]
        cut = next((i for i, (t, v) in enumerate(after)
                    if t == "word" and v.lower() in PARTICLES), len(after))
        out["alias"] = txt(after[:cut])
        name_tokens = before + after[cut:]

    words = [v for t, v in name_tokens if t == "word"]
    brackets = [v for t, v in name_tokens if t == "bracket"]
    out["familyname"] = " ".join(words).strip().rstrip(",")
    if brackets:
        out["familyname_variant"] = "; ".join(brackets)
    return out


def parse_entry(raw, forename, forename_variant):
    text = raw.strip()
    is_dash = bool(re.match(r"^\s*[—–-]", text))
    if is_dash:
        text = re.sub(r"^\s*[—–-]\s*", "", text)

    reference, text = extract_reference(text)
    pages, body = extract_pages(text)

    row = dict(firstname=forename, firstname_variant=forename_variant,
               familyname="", familyname_variant="", alias="",
               office="", institution="", reference=reference,
               zusatz="", column_num=pages, is_dash=int(is_dash),
               raw_entry=raw)
    row.update(parse_body(body))
    return row


def headword_forename(raw):
    """Forename = first token, plus any bracket that follows it."""
    text = raw.strip()
    m = re.match(r"^([A-ZÄÖÜ][\wäöüß]*)\s*(\(([^)]*)\))?\s*(.*)$", text, re.S)
    if not m:
        return text.split()[0], "", ""
    return m.group(1), (m.group(3) or "").strip(), m.group(4)


def run(in_csv, out_csv):
    df = pd.read_csv(in_csv, encoding="utf-8-sig", dtype=str).fillna("")
    rows, forename, fvar = [], "", ""

    for _, r in df.iterrows():
        raw = r["raw_entry"]
        if re.match(r"^\s*[—–-]", raw):
            row = parse_entry(raw, forename, fvar)
        else:
            forename, fvar, rest = headword_forename(raw)
            row = parse_entry(raw, forename, fvar)
            # headword: the forename itself is not part of the body
            row.update(parse_body(
                extract_pages(extract_reference(rest)[1])[1]))
            ref, _ = extract_reference(rest)
            pages, _ = extract_pages(extract_reference(rest)[1])
            row["reference"], row["column_num"] = ref, pages
        row["source_image"] = r.get("source_image", "")
        row["entry_no"] = r.get("entry_no", "")
        rows.append(row)

    cols = ["source_image", "entry_no", "is_dash", "firstname",
            "firstname_variant", "familyname", "familyname_variant", "alias",
            "office", "institution", "reference", "zusatz", "column_num",
            "raw_entry"]
    out = pd.DataFrame(rows)[cols]
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    run("/mnt/user-data/uploads/personen_vol2_raw_final_ordered.csv",
        "vol2_split_preview.csv")