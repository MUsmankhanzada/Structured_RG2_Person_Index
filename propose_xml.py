"""Build an XML proposal grouping dash entries under each main entry."""

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

INPUT_DEFAULT = "vol2_split_preview.csv"
OUTPUT_DEFAULT = "vol2_persons_proposal.xml"
FIELDS = (
    "source_image",
    "entry_no",
    "is_dash",
    "firstname",
    "firstname_variant",
    "familyname",
    "familyname_variant",
    "alias",
    "office",
    "institution",
    "reference",
    "zusatz",
    "column_num",
    "raw_entry",
)


def read_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = [field for field in FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        return list(reader)


def group_people(rows):
    people = []
    for row in rows:
        is_dash = (row.get("is_dash") or "").strip()
        if is_dash == "0":
            people.append([row])
        elif is_dash == "1":
            if not people:
                raise ValueError("The first CSV row is a dash continuation with no main entry")
            people[-1].append(row)
        else:
            raise ValueError(f"Unexpected is_dash value: {is_dash!r}")
    return people


def build_entry(row, entry_type):
    entry = ET.Element("entry", {"type": entry_type})
    for field in FIELDS:
        value = (row.get(field) or "").strip()
        if value:
            ET.SubElement(entry, field).text = value
    return entry


def build_tree(rows):
    root = ET.Element("personenverzeichnis", {"volume": "2"})
    for person_id, entries in enumerate(group_people(rows), start=1):
        person = ET.SubElement(root, "person", {"id": f"p{person_id:04d}"})
        for entry_index, row in enumerate(entries):
            entry_type = "main" if entry_index == 0 else "continuation"
            person.append(build_entry(row, entry_type))
    return ET.ElementTree(root)


def verify_xml(path, rows):
    root = ET.parse(path).getroot()
    if root.tag != "personenverzeichnis" or root.get("volume") != "2":
        return ["Unexpected XML root or volume"]

    people = root.findall("./person")
    expected_groups = group_people(rows)
    if len(people) != len(expected_groups):
        return [f"Expected {len(expected_groups)} people, found {len(people)}"]

    problems = []
    row_number = 0
    for person_id, (person, expected_entries) in enumerate(
        zip(people, expected_groups), start=1
    ):
        expected_id = f"p{person_id:04d}"
        if person.get("id") != expected_id:
            problems.append(f"Person {person_id}: expected id {expected_id}")
        entries = person.findall("./entry")
        if len(entries) != len(expected_entries):
            problems.append(f"Person {expected_id}: entry count did not round-trip")
            continue
        for entry_index, (entry, row) in enumerate(zip(entries, expected_entries)):
            expected_type = "main" if entry_index == 0 else "continuation"
            if entry.get("type") != expected_type:
                problems.append(f"Person {expected_id}: entry type did not round-trip")
            for field in FIELDS:
                expected = (row.get(field) or "").strip()
                actual = entry.findtext(field, default="")
                if actual != expected:
                    problems.append(f"CSV row {row_number + 1}: {field} did not round-trip")
            row_number += 1
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=INPUT_DEFAULT)
    parser.add_argument("-o", "--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--preview", nargs="?", const=3, type=int)
    args = parser.parse_args()

    try:
        rows = read_rows(args.input)
        tree = build_tree(rows)
        ET.indent(tree, space="\t")
        if args.preview is not None:
            root = tree.getroot()
            preview = ET.ElementTree(
                ET.Element(root.tag, root.attrib)
            )
            preview.getroot().extend(list(root)[:max(0, args.preview)])
            ET.indent(preview, space="\t")
            print(ET.tostring(preview.getroot(), encoding="unicode"))
            return

        tree.write(args.output, encoding="utf-8", xml_declaration=True)
        problems = verify_xml(args.output, rows)
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            raise SystemExit(1)
        print(f"CSV rows         : {len(rows)}")
        print(f"People           : {len(group_people(rows))}")
        print(f"Entries          : {len(rows)}")
        print("Checks           : all passed")
        print(f"Wrote {args.output}")
    except (OSError, ValueError, csv.Error, ET.ParseError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()