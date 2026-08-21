"""Merge only the opt-in straight-line distance changes into an anchor artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_straight_line(question: dict) -> bool:
    text = str(question.get("question", "")).lower()
    return any(x in text for x in ("straight-line", "straight line", "line-of-sight", "line of sight"))


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
    by_key = {(str(v), str(row["id"])): row for v, rows in probe.items() for row in rows}
    changed = []
    for video_id, questions in manifest.items():
        for key, question in questions.items():
            if not is_straight_line(question):
                continue
            qid = str(question.get("id", key))
            rows = [row for row in anchor.get(video_id, []) if str(row.get("id")) == qid]
            replacement = by_key.get((str(video_id), qid))
            if len(rows) != 1 or replacement is None:
                raise SystemExit(f"missing unique row {video_id}/{qid}")
            old = rows[0]["answer_id"]
            new = replacement["answer_id"]
            if old != new:
                rows[0]["answer_id"] = new
                rows[0]["answer"] = question["options"][new]
                changed.append((video_id, qid, old, new))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(anchor, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"changed": len(changed), "changes": changed}, indent=2))


if __name__ == "__main__":
    main()
