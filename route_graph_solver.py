"""Label-free street-graph ranking for route-instruction options.

This is deliberately independent of video trajectories: options are parsed into
ordered street/place mentions, geocoded in the public city context, and ranked
by whether successive mentions are geographically adjacent.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PREFIX = re.compile(
    r"^(?:head\s+(?:north|south|east|west|northwest|northeast|southwest|southeast)\s+on\s+|"
    r"turn\s+(?:left|right)\s+onto\s+|slight\s+(?:left|right)(?:\s+to\s+stay)?\s+on\s+|"
    r"slight\s+(?:left|right)\s+onto\s+|continue\s+(?:straight\s+to\s+stay\s+on\s+|to\s+follow\s+|onto\s+)|"
    r"turn\s+(?:left|right)\s+toward\s+|head\s+[^|]+?\s+toward\s+)", re.I)


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = value.split(" toward ", 1)[0]
    value = re.sub(r"\s+", " ", value.replace("/", " / ")).strip(" .,;:")
    return value


def option_names(option: str) -> list[str]:
    result = []
    for segment in option.split("|"):
        segment = segment.strip()
        if not segment or segment.lower() == "take the stairs":
            continue
        name = PREFIX.sub("", segment).strip()
        name = re.sub(r"\s+toward\s+.*$", "", name, flags=re.I)
        name = norm(name)
        if len(name) >= 4 and name.lower() not in {"continue straight", "there"}:
            if not result or result[-1].casefold() != name.casefold():
                result.append(name)
    return result


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    x = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(x))


def geocode(query: str, cache: dict[str, list[list[float]]], delay: float) -> list[list[float]]:
    if query in cache:
        return cache[query]
    url = "https://nominatim.openstreetmap.org/search?" + urlencode({"q": query, "format": "jsonv2", "limit": 5})
    request = Request(url, headers={"User-Agent": "KilometerVision academic route-graph audit"})
    try:
        with urlopen(request, timeout=25) as response:
            rows = json.loads(response.read().decode())
        cache[query] = [[float(row["lat"]), float(row["lon"])] for row in rows]
    except Exception:
        cache[query] = []
    time.sleep(delay)
    return cache[query]


def rank_question(question: dict, city: str, cache: dict, delay: float) -> dict:
    sequences = [option_names(str(option)) for option in question.get("options", [])]
    locations: dict[str, list[list[float]]] = {}
    for sequence in sequences:
        for name in sequence:
            if name not in locations:
                locations[name] = geocode(f"{name}, {city}", cache, delay)
    scores = []
    for sequence in sequences:
        pairs = []
        for left, right in zip(sequence, sequence[1:]):
            if locations.get(left) and locations.get(right):
                pairs.append(min(haversine(tuple(a), tuple(b)) for a in locations[left] for b in locations[right]))
        connected = sum(distance <= 300 for distance in pairs)
        # Favor connected ordered paths, mildly penalize implausible jumps, and
        # do not punish missing OSM names as if they were disconnected streets.
        score = connected - sum(min(distance, 3000.0) / 3000.0 for distance in pairs) * 0.15
        scores.append({"score": score, "connected": connected, "pairs": pairs, "sequence": sequence})
    prediction = max(range(len(scores)), key=lambda index: scores[index]["score"]) if scores else None
    return {"prediction": prediction, "scores": scores, "city": city}


def city_context(title: str) -> str:
    """Extract a compact geocoder context from varied YouTube title formats."""
    match = re.search(r"\bin\s+([^/,\-]+)", title, flags=re.I)
    if match:
        city = match.group(1).strip()
        country = re.search(r"/\s*([^🇦🇫-]+)", title)
        return f"{city}, {country.group(1).strip()}" if country else city
    return title.split(",", 1)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=1.05)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    anchors = json.loads(args.anchors.read_text())["records"]
    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    wanted = set(args.video_id) if args.video_id else set(manifest)
    records = []
    for video_id in sorted(wanted & set(manifest)):
        metadata = anchors.get(video_id, {})
        title = metadata.get("title", "")
        city = re.sub(r"[^\w ,.-]", "", city_context(title), flags=re.UNICODE).strip()
        if not city:
            continue
        for question_id, question in manifest[video_id].items():
            if not any(re.search(r"\b(?:head|turn|continue|slight)\b", str(option), re.I) for option in question.get("options", [])):
                continue
            result = rank_question(question, city, cache, args.delay)
            records.append({"video_id": video_id, "question_id": str(question_id), **result})
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            args.cache.write_text(json.dumps(cache, indent=2) + "\n")
        print(video_id, len([row for row in records if row["video_id"] == video_id]))
    payload = {"method": "nominatim ordered street adjacency", "ground_truth_used": False, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"questions": len(records), "videos": len({row['video_id'] for row in records})}))


if __name__ == "__main__":
    main()
