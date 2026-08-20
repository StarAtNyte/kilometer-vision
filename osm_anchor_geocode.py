"""Geocode public route-anchor text and measure geographic cluster coherence."""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path


REJECT = {
    "after", "around", "asian side", "before", "camera", "copyright", "european",
    "filmed", "full", "mic", "music", "osmo pocket", "please", "statue", "the",
    "there", "tour", "turkey", "uhd", "video", "walking", "zoom h1",
}


def distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 12742.0 * math.asin(math.sqrt(value))


def search(query: str, cache: dict, delay: float) -> list[dict]:
    if query in cache:
        return cache[query]
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({
        "q": query, "format": "jsonv2", "limit": 5, "addressdetails": 1,
    })
    request = urllib.request.Request(url, headers={
        "User-Agent": "KilometerVision academic research (private benchmark experiment)",
    })
    with urllib.request.urlopen(request, timeout=30) as response:
        rows = json.loads(response.read().decode("utf-8"))
    cache[query] = [{
        "lat": float(row["lat"]), "lon": float(row["lon"]),
        "display_name": row.get("display_name", ""), "type": row.get("type", ""),
        "importance": float(row.get("importance", 0.0)),
    } for row in rows]
    time.sleep(delay)
    return cache[query]


def useful(values: list[str], limit: int) -> list[str]:
    result = []
    for value in values:
        value = value.strip()
        folded = value.casefold()
        if folded in REJECT or len(value) < 4 or len(value) > 60:
            continue
        if value not in result:
            result.append(value)
        if len(result) == limit:
            break
    return result


def cluster(query_results: dict[str, list[dict]], radius_km: float) -> dict:
    points = [(query, row) for query, rows in query_results.items() for row in rows]
    best: dict | None = None
    for _, center in points:
        members = []
        for query, rows in query_results.items():
            nearby = [row for row in rows if distance_km(
                (center["lat"], center["lon"]), (row["lat"], row["lon"])
            ) <= radius_km]
            if nearby:
                chosen = max(nearby, key=lambda row: row["importance"])
                members.append({"query": query, **chosen})
        score = (len(members), sum(row["importance"] for row in members))
        if best is None or score > tuple(best["score"]):
            best = {"score": list(score), "center": [center["lat"], center["lon"]], "members": members}
    return best or {"score": [0, 0.0], "center": None, "members": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("anchors", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--max-anchors", type=int, default=12)
    parser.add_argument("--radius-km", type=float, default=30.0)
    parser.add_argument("--delay", type=float, default=1.05)
    args = parser.parse_args()

    source = json.loads(args.anchors.read_text())["records"]
    wanted = set(args.video_id) if args.video_id else set(source)
    cache_path = args.output.with_suffix(".cache.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    records = {}
    for video_id, metadata in source.items():
        if video_id not in wanted:
            continue
        values = metadata.get("route_anchor_candidates", [])
        queries = useful(values, args.max_anchors)
        query_results = {}
        for query in queries:
            try:
                query_results[query] = search(query, cache, args.delay)
                cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n")
            except Exception as error:
                query_results[query] = []
                print(f"{video_id}: {query}: {type(error).__name__}: {error}")
        best = cluster(query_results, args.radius_km)
        records[video_id] = {
            "title": metadata.get("title", ""), "queries": queries,
            "queries_with_results": sum(bool(rows) for rows in query_results.values()),
            "cluster": best, "query_results": query_results,
        }
        print(video_id, best["score"][0], "/", len(queries), best["center"])

    result = {
        "ground_truth_used": False, "radius_km": args.radius_km, "records": records,
        "summary": {
            "videos": len(records),
            "coherent_2plus": sum(row["cluster"]["score"][0] >= 2 for row in records.values()),
            "coherent_3plus": sum(row["cluster"]["score"][0] >= 3 for row in records.values()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
