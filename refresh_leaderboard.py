"""Print the public KilometerVision test leaderboard from EvalAI."""
from __future__ import annotations

import json
import urllib.request

URL = "https://eval.ai/api/jobs/challenge_phase_split/7010/leaderboard/?page_size=1000&order_by=Top-1"


def fetch() -> dict:
    with urllib.request.urlopen(URL, timeout=20) as response:
        return json.load(response)


def main() -> None:
    payload = fetch()
    print("| Position | Team | Method | Top-1 | Submitted |")
    print("| ---: | --- | --- | ---: | --- |")
    for position, row in enumerate(payload.get("results", []), 1):
        score = row.get("result", [None])[0]
        print(f"| {position} | {row.get('submission__participant_team__team_name', '')} | "
              f"{row.get('submission__method_name', '')} | {score:.6f} | "
              f"{row.get('submission__submitted_at', '')} |")


if __name__ == "__main__":
    main()
