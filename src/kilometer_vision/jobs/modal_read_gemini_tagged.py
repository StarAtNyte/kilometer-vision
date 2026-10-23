"""Read and score an isolated Gemini validation tag after inference completes."""
from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

import modal

ROOT = Path("/data/2707-kilometer-vision")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = modal.Image.debian_slim(python_version="3.11")
APP = modal.App("kilometervision-read-gemini-tagged")


def _family(question: dict) -> str:
    text = str(question.get("question", "")).lower()
    if "maps correctly" in text:
        return "map"
    if "direction" in text:
        return "compass"
    if "image show" in text:
        return "landmark"
    return "distance"


@APP.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=10 * 60)
def read_tagged(split: str = "validation", tag: str = "fps2-trace-map") -> dict:
    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    rows = []
    for video_id, questions in manifest.items():
        for question_id, question in questions.items():
            path = ROOT / "experiments" / "gemini-direct-video" / f"{split}-{video_id}-{question_id}-{tag}.json"
            if not path.exists():
                rows.append({"video_id": video_id, "question_id": str(question_id), "status": "missing"})
                continue
            record = json.loads(path.read_text())
            raw = str(record.get("raw_response", "")).strip().upper()
            prediction = "ABCDE".find(raw[:1]) if raw else -1
            answer_id = question.get("answer_id")
            rows.append({"video_id": video_id, "question_id": str(question_id), "family": _family(question),
                         "raw_response": raw, "prediction": prediction, "answer_id": answer_id,
                         "correct": prediction == answer_id, "status": "ok" if prediction >= 0 else "unparsed",
                         "settings": record.get("settings", {})})
    valid = [row for row in rows if row["status"] == "ok"]
    by_family = {}
    for family in sorted({row.get("family") for row in valid}):
        subset = [row for row in valid if row["family"] == family]
        by_family[family] = {"correct": sum(row["correct"] for row in subset), "total": len(subset)}
    return {"split": split, "tag": tag, "records": len(rows), "valid": len(valid),
            "correct": sum(row["correct"] for row in valid), "by_family": by_family,
            "status_counts": dict(Counter(row["status"] for row in rows)), "rows": rows,
            "ground_truth_used_for_inference": False}


@APP.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=10 * 60)
def audit_base(split: str = "test") -> dict:
    """Audit the production direct-video records and their media associations."""
    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    archive_path = ROOT / "manifests" / f"{split}_images.zip"
    archive_names = []
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive_names = archive.namelist()
    counters = Counter()
    answers = Counter()
    models = Counter()
    fps_values = Counter()
    clip_bytes = []
    bad_image_refs = []
    durations = Counter()
    for video_id, questions in manifest.items():
        for question_id, question in questions.items():
            counters["questions"] += 1
            family = _family(question)
            counters[f"family:{family}"] += 1
            duration = float(question.get("end_seconds", 0) or 0) - float(question.get("start_seconds", 0) or 0)
            durations[family] += max(0.0, duration)
            refs = question.get("image_paths", [])
            if refs:
                counters["image_questions"] += 1
                if family == "landmark":
                    counters["landmarks_affected_by_old_map_prompt"] += 1
                for reference in refs:
                    matches = [name for name in archive_names if Path(name).name == Path(str(reference)).name]
                    if len(matches) != 1:
                        bad_image_refs.append({"video_id": video_id, "question_id": str(question_id),
                                               "reference": str(reference), "matches": len(matches)})
            record_path = ROOT / "experiments" / "gemini-direct-video" / f"{split}-{video_id}-{question_id}.json"
            if not record_path.exists():
                counters["missing_records"] += 1
                continue
            try:
                record = json.loads(record_path.read_text())
            except (OSError, ValueError):
                counters["invalid_json_records"] += 1
                continue
            counters["records"] += 1
            raw = str(record.get("raw_response", "")).strip().upper()
            if raw in "ABCDE" and len(raw) == 1:
                counters["valid_enum_records"] += 1
                answers[raw] += 1
            else:
                counters["invalid_enum_records"] += 1
            models[str(record.get("model", "missing"))] += 1
            settings = record.get("settings")
            if isinstance(settings, dict):
                counters["records_with_settings"] += 1
                fps_values[str(settings.get("fps", "missing"))] += 1
            else:
                counters["records_without_settings"] += 1
            clip_path = ROOT / "experiments" / "gemini-direct-video" / "clips" / f"{split}-{video_id}-{question_id}-window.mp4"
            if clip_path.exists():
                counters["clips"] += 1
                clip_bytes.append(clip_path.stat().st_size)
            else:
                counters["missing_clips"] += 1
    return {
        "split": split,
        "counts": dict(counters),
        "answers": dict(answers),
        "models": dict(models),
        "fps_values_when_recorded": dict(fps_values),
        "family_total_duration_seconds": dict(durations),
        "clip_bytes": {"min": min(clip_bytes) if clip_bytes else None,
                       "max": max(clip_bytes) if clip_bytes else None,
                       "mean": sum(clip_bytes) / len(clip_bytes) if clip_bytes else None},
        "bad_image_refs": bad_image_refs,
        "ground_truth_used_for_audit": False,
    }


@APP.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=5 * 60)
def preview_adjudication(split: str = "test",
                          target_file: str = "experiments/gemini-direct-video/adjudication-targets-2026-08-26.json",
                          base_tag: str = "", new_tag: str = "leader-repro-clean",
                          vote_tags: str = "adjudication-vote-1,adjudication-vote-2,adjudication-vote-3") -> dict:
    """Partial/in-progress preview of the majority-vote merge over the disagreement set.

    Uses whatever vote records exist right now; missing votes just don't
    contribute. Tie or missing base answer keeps the base (580307) answer,
    matching the final merge's fallback rule.
    """
    VOLUME.reload()
    targets = json.loads((ROOT / target_file.lstrip("/")).read_text())
    tags = [base_tag, new_tag] + [t.strip() for t in vote_tags.split(",") if t.strip()]
    changed = 0
    missing_base = 0
    vote_coverage: Counter = Counter()
    flips = []
    for target in targets:
        video_id, question_id = target["video_id"], str(target["question_id"])
        votes = []
        for tag in tags:
            suffix = f"-{tag}" if tag else ""
            path = ROOT / "experiments" / "gemini-direct-video" / f"{split}-{video_id}-{question_id}{suffix}.json"
            if not path.exists():
                continue
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            raw = str(record.get("raw_response", "")).strip().upper()
            if raw in set("ABCDE"):
                votes.append((tag, raw))
        vote_coverage[len(votes)] += 1
        old_answer = next((value for tag, value in votes if tag == base_tag), None)
        if old_answer is None:
            missing_base += 1
            continue
        counts = Counter(value for _, value in votes)
        top_letter, top_n = counts.most_common(1)[0]
        tied = [letter for letter, n in counts.items() if n == top_n]
        winner = top_letter if len(tied) == 1 else old_answer
        if winner != old_answer:
            changed += 1
            flips.append({"video_id": video_id, "question_id": question_id, "old": old_answer,
                         "winner": winner, "votes": dict(counts), "n_votes": len(votes)})
    return {"targets": len(targets), "changed_so_far": changed, "missing_base": missing_base,
            "vote_coverage_histogram": dict(sorted(vote_coverage.items())),
            "sample_flips": flips[:25], "ground_truth_used_for_inference": False}


@APP.local_entrypoint()
def main(split: str = "validation", tag: str = "fps2-trace-map", action: str = "tagged"):
    if action == "audit-base":
        result = audit_base.remote(split)
    elif action == "preview-adjudication":
        result = preview_adjudication.remote(split)
    else:
        result = read_tagged.remote(split, tag)
    print(json.dumps(result, indent=2))
