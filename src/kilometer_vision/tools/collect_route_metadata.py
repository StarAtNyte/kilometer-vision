"""Collect public YouTube route metadata without reading challenge answers."""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


GENERIC = {
    "Walking", "Tour", "Video", "Subscribe", "Youtube", "Google", "Ultra", "Full",
    "Around", "Copyright", "Music", "Camera", "Thanks", "Please", "Channel",
}


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "KilometerVision research/1.0"})
    return urllib.request.urlopen(request, timeout=30).read().decode("utf-8", "replace")


def anchors(text: str) -> list[str]:
    text = html.unescape(text.replace("\\n", "\n"))
    candidates: list[str] = []
    patterns = (
        r"\b[A-ZÀ-ÖØ-Þ][\wÀ-ž'’-]*(?:[ -]+[A-ZÀ-ÖØ-Þ][\wÀ-ž'’-]*){0,4}\b",
        r"\b(?:at|from|to|via|towards?|near)\s+([A-ZÀ-ÖØ-Þ][^\n,.;:()]{2,60})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = (match.group(1) if match.lastindex else match.group(0)).strip(" -–—")
            words = value.split()
            if not value or words[0] in GENERIC or len(value) < 3 or len(value) > 70:
                continue
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
                continue
            if value.casefold() not in {item.casefold() for item in candidates}:
                candidates.append(value)
    return candidates[:40]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = {}
    for video_id in manifest:
        record = {"source": "youtube_public_metadata", "status": "error"}
        try:
            oembed_url = "https://www.youtube.com/oembed?" + urllib.parse.urlencode({
                "url": f"https://www.youtube.com/watch?v={video_id}", "format": "json",
            })
            title = json.loads(fetch(oembed_url)).get("title", "")
            page = fetch(f"https://www.youtube.com/watch?v={video_id}")
            found = re.search(r'"shortDescription":"((?:\\.|[^"\\])*)"', page)
            description = json.loads('"' + found.group(1) + '"') if found else ""
            record.update({
                "status": "ok", "title": title, "description_chars": len(description),
                "route_anchor_candidates": anchors(title + "\n" + description),
            })
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        records[video_id] = record
        time.sleep(0.1)
    result = {"ground_truth_used": False, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "videos": len(records), "ok": sum(r["status"] == "ok" for r in records.values()),
        "with_anchors": sum(bool(r.get("route_anchor_candidates")) for r in records.values()),
    }))


if __name__ == "__main__":
    main()
