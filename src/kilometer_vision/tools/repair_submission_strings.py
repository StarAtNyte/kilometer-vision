"""Repair answer text after an answer_id-only candidate merge."""
import argparse
import json
from pathlib import Path

from kilometer_vision.submission import audit_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    predictions = json.loads(args.input.read_text())
    for video_id, rows in predictions.items():
        questions = manifest[video_id]
        for row in rows:
            question = questions.get(str(row["id"]))
            if question is None:
                question = next(q for q in questions.values() if str(q.get("id")) == str(row["id"]))
            row["answer"] = str(question["options"][row["answer_id"]])
    audit = audit_predictions(manifest, predictions)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, indent=2))
    args.output.write_text(json.dumps(predictions, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
