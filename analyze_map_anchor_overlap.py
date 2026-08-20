"""Measure label-free overlap between public route metadata and candidate-map OCR."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


STOPWORDS = {
    "and", "around", "data", "downtown", "google", "map", "maps", "market",
    "old", "part", "street", "the", "tour", "town", "video", "walk", "walking",
    "with", "world", "city", "center", "centre", "copyright", "route",
}


def tokens(values: list[str]) -> set[str]:
    text = " ".join(values)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return {
        token for token in re.findall(r"[a-z]{4,}", text)
        if token not in STOPWORDS and not token.endswith("fps")
    }


def similar(anchor: str, candidate: str) -> bool:
    if anchor == candidate:
        return True
    if min(len(anchor), len(candidate)) < 5:
        return False
    return SequenceMatcher(None, anchor, candidate).ratio() >= 0.86


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ocr_dir", type=Path)
    parser.add_argument("--anchors", type=Path, default=Path("experiments/video-route-anchors-test.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/map-anchor-overlap-test.json"))
    args = parser.parse_args()

    anchor_records = json.loads(args.anchors.read_text())["records"]
    rows = []
    for path in sorted(args.ocr_dir.rglob("*.json")):
        payload = json.loads(path.read_text())
        for record in payload["records"]:
            metadata = anchor_records.get(record["video_id"], {})
            anchor_values = list(metadata.get("route_anchor_candidates", [])) + [metadata.get("title", "")]
            anchor_tokens = tokens(anchor_values)
            scores = []
            matches = []
            for candidate in record["candidates"]:
                candidate_tokens = tokens(candidate)
                hit = sorted({a for a in anchor_tokens if any(similar(a, c) for c in candidate_tokens)})
                scores.append(len(hit))
                matches.append(hit)
            best = max(scores, default=0)
            winners = [index for index, score in enumerate(scores) if score == best and score > 0]
            ordered = sorted(scores, reverse=True)
            rows.append({
                "video_id": record["video_id"], "question_id": record["question_id"],
                "scores": scores, "matches": matches, "winners": winners,
                "margin": best - (ordered[1] if len(ordered) > 1 else 0),
            })

    summary = {
        "ground_truth_used": False,
        "ocr_files": len(list(args.ocr_dir.rglob("*.json"))),
        "questions": len(rows),
        "candidate_image_questions": sum(len(row["scores"]) == 5 for row in rows),
        "any_match": sum(bool(row["winners"]) for row in rows),
        "unique_winner": sum(len(row["winners"]) == 1 for row in rows),
        "positive_margin": sum(row["margin"] > 0 for row in rows),
        "winner_distribution": dict(sorted(Counter(
            row["winners"][0] for row in rows if len(row["winners"]) == 1
        ).items())),
    }
    result = {"summary": summary, "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
