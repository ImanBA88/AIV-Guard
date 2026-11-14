#!/usr/bin/env python3
"""Validate or reconstruct the 250-case Orchid-HEval evaluation dataset.

Examples:
    python scripts/rebuild_dataset.py --verify-existing
    python scripts/rebuild_dataset.py --source path/to/Orchid-HEval/data.jsonl
    python scripts/rebuild_dataset.py --revision <commit-or-tag>

Rebuilding does not run the classifier or change saved predictions.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
CASES_FILE = DATA_DIR / "orchid_new250_cases.csv"
MANIFEST_FILE = DATA_DIR / "manifest.csv"
CEV_SAMPLE_FILE = DATA_DIR / "selected_30_cases.csv"

COLUMNS = [
    "case_id", "task_name", "entry_point", "requirement",
    "gold_label", "gold_binary", "gold_explanation",
]
LABELS = [
    ("clear", "prompt", ""),
    ("lexical", "Lexical_prompt", "Lexical_ambiguity_explanation"),
    ("semantic", "Semantic_prompt", "Semantic_ambiguity_explanation"),
    ("syntactic", "Syntactic_prompt", "Syntactic_ambiguity_explanation"),
    ("vagueness", "Vagueness_prompt", "Vagueness_ambiguity_explanation"),
]
PREVIOUS_TASKS = {
    "HumanEval/0", "HumanEval/1", "HumanEval/117", "HumanEval/118",
    "HumanEval/130", "HumanEval/135", "HumanEval/143", "HumanEval/149",
    "HumanEval/39", "HumanEval/49", "HumanEval/5", "HumanEval/50",
    "HumanEval/54", "HumanEval/58", "HumanEval/6", "HumanEval/62",
    "HumanEval/75", "HumanEval/8", "HumanEval/80", "HumanEval/81",
    "HumanEval/82", "HumanEval/86", "HumanEval/90", "HumanEval/93",
    "HumanEval/96",
}


def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_dataset(cases):
    """Check dataset structure and agreement with saved selection records."""
    if len(cases) != 250:
        raise ValueError(f"Expected 250 cases; found {len(cases)}")
    if any(not set(COLUMNS).issubset(row) for row in cases):
        raise ValueError("Dataset is missing required columns")
    if [row["case_id"] for row in cases] != [str(i) for i in range(1, 251)]:
        raise ValueError("Case identifiers must be unique and ordered from 1 to 250")

    groups = Counter(row["task_name"] for row in cases)
    labels = Counter(row["gold_label"] for row in cases)
    if len(groups) != 50 or any(count != 5 for count in groups.values()):
        raise ValueError("Expected 50 source tasks, with five cases per task")
    if labels != {label: 50 for label, _, _ in LABELS}:
        raise ValueError(f"Unexpected label distribution: {dict(labels)}")
    if set(groups) & PREVIOUS_TASKS:
        raise ValueError("The evaluation set overlaps with the previously used source tasks")

    for row in cases:
        expected_binary = "CLEAR" if row["gold_label"] == "clear" else "AMBIGUOUS"
        if row["gold_binary"] != expected_binary:
            raise ValueError(f"Binary gold label mismatch in case {row['case_id']}")
        if not row["requirement"].strip():
            raise ValueError(f"Empty requirement in case {row['case_id']}")

    manifest = load_csv(MANIFEST_FILE)
    fields = ("case_id", "task_name", "gold_label", "gold_binary")
    expected_manifest = [{key: row[key] for key in fields} for row in cases]
    if len(manifest) != 250 or any(not set(fields).issubset(row) for row in manifest):
        raise ValueError("The evaluation manifest is incomplete")
    if [{key: row[key] for key in fields} for row in manifest] != expected_manifest:
        raise ValueError("Dataset identifiers and labels differ from the saved manifest")

    selected = load_csv(CEV_SAMPLE_FILE)
    if len(selected) != 30:
        raise ValueError(f"Expected 30 selected CEV cases; found {len(selected)}")
    if len({row["case_id"] for row in selected}) != 30:
        raise ValueError("Repeated case identifier in the CEV sample")

    by_id = {row["case_id"]: row for row in cases}
    for row in selected:
        case = by_id.get(row["case_id"])
        if case is None:
            raise ValueError(f"Missing selected CEV case {row['case_id']}")
        for field in ("task_name", "gold_label", "gold_binary", "requirement"):
            if row[field] != case[field]:
                raise ValueError(f"CEV sample mismatch: case {row['case_id']}, {field}")

    return len(groups), len(selected)


def build_cases(source_rows):
    if len(source_rows) != 164 or len({row["name"] for row in source_rows}) != 164:
        raise ValueError("Expected 164 distinct Orchid-HEval source tasks")

    keys = {"name", "entry_point"} | {field for _, field, _ in LABELS}
    if any(not keys.issubset(row) for row in source_rows):
        raise ValueError("Upstream Orchid-HEval task schema differs from the expected format")

    available = sorted(
        (row for row in source_rows if row["name"] not in PREVIOUS_TASKS),
        key=lambda row: int(row["name"].split("/")[-1]),
    )
    if len(available) != 139:
        raise ValueError("Expected 139 source tasks after excluding the earlier evaluation set")
    sampled = random.Random(42).sample([row["name"] for row in available], 50)
    chosen = set(sampled)

    cases = []
    for task in source_rows:
        if task["name"] not in chosen:
            continue
        for label, text_key, explanation_key in LABELS:
            cases.append({
                "task_name": task["name"],
                "entry_point": str(task["entry_point"]),
                "requirement": task[text_key],
                "gold_label": label,
                "gold_binary": "CLEAR" if label == "clear" else "AMBIGUOUS",
                "gold_explanation": str(task.get(explanation_key, "") or ""),
            })

    label_order = {label: index for index, (label, _, _) in enumerate(LABELS)}
    cases.sort(key=lambda row: (
        int(row["task_name"].split("/")[-1]), label_order[row["gold_label"]]
    ))
    for case_id, row in enumerate(cases, 1):
        row["case_id"] = str(case_id)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-existing", action="store_true",
                        help="Validate the packaged CSV without downloading or changing it")
    parser.add_argument("--source", type=Path,
                        help="Use an existing local Orchid-HEval/data.jsonl source file")
    parser.add_argument("--revision", default="main",
                        help="Hugging Face revision for source reconstruction")
    args = parser.parse_args()

    if args.verify_existing:
        if args.source is not None:
            parser.error("--source and --verify-existing cannot be combined")
        cases = load_csv(CASES_FILE)
        tasks, cev_count = verify_dataset(cases)
        digest = sha256(CASES_FILE)
        provenance_file = DATA_DIR / "rebuild_provenance.json"
        if provenance_file.exists():
            metadata = json.loads(provenance_file.read_text(encoding="utf-8"))
            recorded = metadata.get("dataset_sha256") or metadata.get("generated_case_sha256")
            if recorded and digest != recorded:
                raise ValueError("Dataset SHA-256 differs from the recorded rebuild fingerprint")
        else:
            print("No saved full-dataset fingerprint is available for offline comparison")
        print(f"Existing dataset structure verified: {len(cases)} cases, {tasks} tasks")
        print(f"CEV input texts matched: {cev_count}")
        print(f"Dataset SHA-256: {digest}")
        print("The offline check does not establish original upstream text provenance.")
        return

    if args.source:
        source = args.source
    else:
        from huggingface_hub import hf_hub_download
        source = Path(hf_hub_download(
            repo_id="SII-YDD/Orchid", repo_type="dataset",
            filename="Orchid-HEval/data.jsonl", revision=args.revision,
        ))

    raw = source.read_bytes()
    source_rows = [
        json.loads(line)
        for line in raw.decode("utf-8-sig").splitlines()
        if line.strip()
    ]
    cases = build_cases(source_rows)
    tasks, cev_count = verify_dataset(cases)

    if CASES_FILE.exists():
        existing = load_csv(CASES_FILE)
        if existing != [{key: row[key] for key in COLUMNS} for row in cases]:
            raise ValueError("Existing dataset differs from the reconstruction; file left unchanged")
        print("Existing dataset agrees with the provided upstream source")
    else:
        with CASES_FILE.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(cases)
        print("Reconstructed dataset:", CASES_FILE)

    metadata = {
        "dataset_repo": "SII-YDD/Orchid",
        "source_file": "Orchid-HEval/data.jsonl",
        "requested_revision": args.revision if args.source is None else None,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_sha256": sha256(CASES_FILE),
        "case_count": len(cases),
        "source_task_count": tasks,
        "matched_cev_cases": cev_count,
    }
    provenance_file = DATA_DIR / "rebuild_provenance.json"
    provenance_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Source SHA-256:", metadata["source_sha256"])
    print("Dataset SHA-256:", metadata["dataset_sha256"])


if __name__ == "__main__":
    main()
