#!/usr/bin/env python3
"""Validate and summarize saved AIV-Guard classification and CEV results.

Usage: python scripts/verify_results.py

This script works offline and does not call any language model.
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_csv(relative_path):
    with (ROOT / relative_path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def as_bool(value):
    ensure(value.strip().lower() in {"true", "false"},
           f"Invalid boolean value: {value!r}")
    return value.strip().lower() == "true"


def f1_for_class(rows, label):
    tp = sum(r["gold_binary"] == r["agent1_label"] == label for r in rows)
    fp = sum(r["agent1_label"] == label and r["gold_binary"] != label for r in rows)
    fn = sum(r["gold_binary"] == label and r["agent1_label"] != label for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    predictions = load_csv("results/predictions_test250.csv")
    cev = load_csv("results/cev_predictions_30.csv")
    manifest = load_csv("dataset/manifest.csv")
    selected = load_csv("dataset/selected_30_cases.csv")

    ensure(len(predictions) == len(manifest) == 250,
           "Expected 250 prediction and manifest rows")
    ensure(len(cev) == len(selected) == 30,
           "Expected 30 CEV output and selected-case rows")
    ensure(all(row["status"] == "ok" for row in predictions + cev),
           "Saved results include incomplete or failed cases")

    expected_ids = [str(case_id) for case_id in range(1, 251)]
    prediction_ids = [row["case_id"] for row in predictions]
    ensure(prediction_ids == expected_ids,
           "Prediction case IDs must cover 1 to 250 in order, without duplicates")

    fields = ("case_id", "task_name", "gold_label", "gold_binary")
    ensure(
        [tuple(row[field] for field in fields) for row in predictions]
        == [tuple(row[field] for field in fields) for row in manifest],
        "Prediction identifiers or gold labels differ from the evaluation manifest",
    )
    ensure(Counter(row["gold_label"] for row in predictions) == {
        "clear": 50, "lexical": 50, "semantic": 50,
        "syntactic": 50, "vagueness": 50,
    }, "Incorrect gold-label distribution")
    tasks = Counter(row["task_name"] for row in predictions)
    ensure(len(tasks) == 50 and all(count == 5 for count in tasks.values()),
           "Expected five requirement variants from each of 50 source tasks")

    valid_predictions = {"CLEAR", "AMBIGUOUS", "INVALID"}
    valid_gate_actions = {"CLEAR", "AMBIGUOUS", "NEEDS_REVIEW"}
    for row in predictions:
        case_id = row["case_id"]
        ensure(row["agent1_label"] in valid_predictions,
               f"Unexpected classifier label in case {case_id}")
        ensure(row["review_action"] in valid_gate_actions,
               f"Unexpected gate action in case {case_id}")
        parsed = as_bool(row["parsed"])
        quote_valid = as_bool(row["quote_valid"])
        evidence = row["evidence"].strip()
        if not parsed:
            expected_gate = "NEEDS_REVIEW"
        elif row["agent1_label"] == "CLEAR" and not evidence:
            expected_gate = "CLEAR"
        elif row["agent1_label"] == "AMBIGUOUS" and quote_valid:
            expected_gate = "AMBIGUOUS"
        else:
            expected_gate = "NEEDS_REVIEW"
        ensure(row["review_action"] == expected_gate,
               f"Gate action contradicts saved parse/evidence flags in case {case_id}")

    by_id = {row["case_id"]: row for row in predictions}
    selected_by_id = {row["case_id"]: row for row in selected}
    ensure(len(selected_by_id) == len(selected), "Duplicate case in CEV selection")
    ensure(len({row["case_id"] for row in cev}) == len(cev),
           "Duplicate case in CEV results")
    ensure(set(selected_by_id) == {row["case_id"] for row in cev},
           "CEV results do not correspond to the saved 30-case selection")

    for row in cev:
        case_id = row["case_id"]
        ensure(case_id in by_id, f"CEV case {case_id} is absent from classification results")
        expected = by_id[case_id]
        sample = selected_by_id[case_id]
        for field in ("task_name", "gold_label", "gold_binary", "agent1_label", "evidence"):
            ensure(row[field] == expected[field] == sample[field],
                   f"CEV case {case_id} disagrees on {field}")
        ensure(row["requirement"] == sample["requirement"],
               f"CEV input requirement mismatch in case {case_id}")
        ensure(expected["agent1_label"] == "AMBIGUOUS" and
               expected["review_action"] == "AMBIGUOUS",
               f"CEV case {case_id} was not accepted as AMBIGUOUS")
        structural_pass = as_bool(row["structural_pass"])
        expected_action = "HUMAN_SEMANTIC_REVIEW" if structural_pass else "NEEDS_REVIEW"
        ensure(row["review_action"] == expected_action,
               f"CEV review action mismatch in case {case_id}")

    case_file = ROOT / "dataset" / "orchid_new250_cases.csv"
    if case_file.exists():
        cases = load_csv("dataset/orchid_new250_cases.csv")
        ensure(len(cases) == 250, "Packaged dataset must contain 250 cases")
        ensure([tuple(row[field] for field in fields) for row in cases]
               == [tuple(row[field] for field in fields) for row in manifest],
               "Packaged dataset differs from the experiment manifest")
        data_by_id = {row["case_id"]: row for row in cases}
        for row in selected:
            ensure(data_by_id[row["case_id"]]["requirement"] == row["requirement"],
                   f"Packaged dataset disagrees with selected case {row['case_id']}")
        provenance_file = ROOT / "dataset" / "rebuild_provenance.json"
        if provenance_file.exists():
            metadata = json.loads(provenance_file.read_text(encoding="utf-8"))
            recorded = metadata.get("dataset_sha256") or metadata.get("generated_case_sha256")
            if recorded:
                current_sha = hashlib.sha256(case_file.read_bytes()).hexdigest()
                ensure(current_sha == recorded,
                       "Dataset differs from the recorded rebuild fingerprint")
                print("Dataset SHA-256 matches the saved rebuild fingerprint")
        print("Packaged dataset: 250 cases and selected CEV texts verified")
    else:
        print("Packaged full dataset not found; checked manifest and 30 saved input texts")

    correct = sum(row["agent1_label"] == row["gold_binary"] for row in predictions)
    f1_scores = []
    for label in ("CLEAR", "AMBIGUOUS"):
        precision, recall, f1 = f1_for_class(predictions, label)
        f1_scores.append(f1)
        print(f"{label}: precision={precision:.2%}, recall={recall:.2%}, F1={f1:.2%}")

    gate = Counter(row["review_action"] for row in predictions)
    accepted = [row for row in predictions if row["review_action"] != "NEEDS_REVIEW"]
    accepted_correct = sum(row["review_action"] == row["gold_binary"] for row in accepted)
    passed = sum(as_bool(row["structural_pass"]) for row in cev)
    pending = sum(row["human_semantic_check"] == "PENDING" for row in cev)
    macro_f1 = sum(f1_scores) / 2

    print(f"Overall accuracy: {correct}/{len(predictions)} = {correct / len(predictions):.2%}")
    print(f"Macro-F1: {macro_f1:.2%}")
    print(f"Gate: {dict(gate)}")
    print(f"Accepted-only accuracy: {accepted_correct}/{len(accepted)} = "
          f"{accepted_correct / len(accepted):.2%}")
    print(f"CEV structural passes: {passed}/{len(cev)}")
    print(f"CEV semantic review pending: {pending}/{len(cev)}")

    ensure(correct == 171, "Classification accuracy differs from the reported evaluation")
    ensure(abs(macro_f1 - 0.6212398) < 0.001,
           "Macro-F1 differs from the reported evaluation")
    ensure(gate == {"CLEAR": 88, "AMBIGUOUS": 138, "NEEDS_REVIEW": 24},
           "Gate counts differ from the reported evaluation")
    ensure(accepted_correct == 157,
           "Gate accepted-correct count differs from the reported evaluation")
    ensure(passed == 27, "CEV structural count differs from the reported evaluation")
    ensure(pending == 30, "CEV human-review status differs from the reported pilot")
    print("All saved-result consistency checks passed.")
    print("This verification does not establish semantic validity or rerun the language model.")


if __name__ == "__main__":
    main()
