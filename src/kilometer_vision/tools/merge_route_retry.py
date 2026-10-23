"""Create a reversible candidate by filling only missing route-instruction VLM records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge(base_path: Path, evidence_path: Path, manifest_path: Path,
          retry_root: Path, output_path: Path) -> dict:
    base = json.loads(base_path.read_text())
    evidence = json.loads(evidence_path.read_text())["videos"]
    manifest = json.loads(manifest_path.read_text())
    changed = []
    for video_id, items in evidence.items():
        retry_path = retry_root / f"{video_id}.json"
        if not retry_path.exists() or video_id not in base:
            continue
        retry = json.loads(retry_path.read_text())
        by_id = {str(row_id): row for row_id, row in retry.items()}
        for item in items:
            if item.get("category") != "map":
                continue
            question = manifest.get(video_id, {}).get(str(item["id"]), {})
            if question.get("image_paths"):
                continue
            vlm = next((row for row in item.get("evidence", [])
                        if row.get("source") == "vlm_evidence_card"), {})
            if vlm.get("status") == "ok":
                continue
            question_id = str(item["id"])
            replacement = by_id.get(question_id, {})
            selected = replacement.get("selected_index")
            if replacement.get("status") != "ok" or not isinstance(selected, int):
                continue
            target = next((row for row in base[video_id] if str(row.get("id")) == question_id), None)
            if target is None:
                continue
            previous = target.get("answer_id")
            if previous == selected:
                continue
            target["answer_id"] = selected
            options = question.get("options", [])
            if 0 <= selected < len(options):
                target["answer"] = options[selected]
            changed.append({"video_id": video_id, "question_id": question_id,
                            "from": previous, "to": selected})
    output_path.write_text(json.dumps(base, indent=2) + "\n")
    return {"output": str(output_path), "changed": len(changed), "records": changed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.base, args.evidence, args.manifest,
                           args.retry_root, args.output), indent=2))


if __name__ == "__main__":
    main()
