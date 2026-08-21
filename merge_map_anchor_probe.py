"""Merge only high-evidence, label-free map OCR/metadata winners."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    anchor = json.loads(args.anchor.read_text())
    probe = json.loads(args.probe.read_text())
    manifest = json.loads(args.manifest.read_text())
    by_key = {(str(row["video_id"]), str(row["question_id"])): row for row in probe["records"]}
    changed = []
    for video_id, questions in manifest.items():
        for key, question in questions.items():
            row = by_key.get((str(video_id), str(question.get("id", key))))
            if row is None or len(row.get("winners", [])) != 1 or int(row.get("margin", 0)) < 1:
                continue
            winner = int(row["winners"][0])
            matches = row.get("matches", [[] for _ in range(5)])[winner]
            # Require two independently visible tokens; a city name alone is
            # common to every distractor and is not enough to override a map.
            distinct = sorted({str(token).lower() for token in matches if str(token).strip()})
            if int(row.get("scores", [0] * 5)[winner]) < 2 or len(distinct) < 2:
                continue
            target = [x for x in anchor.get(video_id, []) if str(x.get("id")) == str(question.get("id", key))]
            if len(target) != 1:
                raise SystemExit(f"missing unique row {video_id}/{key}")
            old = target[0]["answer_id"]
            if old != winner:
                target[0]["answer_id"] = winner
                target[0]["answer"] = question["options"][winner]
                changed.append({"video_id": video_id, "question_id": str(question.get("id", key)), "old": old, "new": winner, "matches": distinct})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(anchor, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"changed": len(changed), "changes": changed}, indent=2))


if __name__ == "__main__":
    main()
