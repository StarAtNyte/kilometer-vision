"""Build answer-blind test hybrids over the frozen direct-Gemini submission lane."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import modal

ROOT = Path("/data/2707-kilometer-vision")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (modal.Image.debian_slim(python_version="3.11")
         .add_local_python_source("motion_solver", "question_planner"))
APP = modal.App("kilometervision-build-gemini-hybrid")


def _gemini_index(split: str, video_id: str, question_id: str, tag: str = "") -> int:
    suffix = f"-{tag}" if tag else ""
    path = ROOT / "experiments" / "gemini-direct-video" / f"{split}-{video_id}-{question_id}{suffix}.json"
    record = json.loads(path.read_text())
    raw = str(record.get("raw_response", "")).strip().upper()
    if raw not in set("ABCDE"):
        raise ValueError(f"missing Gemini enum for {video_id}/{question_id}")
    return "ABCDE".index(raw)


@APP.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=10 * 60)
def build_motion_compass(split: str = "test", landmark_tag: str = "",
                         include_motion: bool = True, base_tag: str = "") -> dict:
    from motion_solver import compass_option_index
    from question_planner import classify_question_facets

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    output = {}
    changed = []
    compass_questions = motion_coverage = 0
    landmark_questions = landmark_coverage = 0
    for video_id, questions in manifest.items():
        rows = []
        motion_path = ROOT / "motion" / split / "compass" / f"{video_id}.json"
        try:
            motion_records = json.loads(motion_path.read_text())
        except (OSError, ValueError):
            motion_records = {}
        for question_id, question in questions.items():
            base = _gemini_index(split, video_id, str(question_id), base_tag)
            selected = base
            text = str(question.get("question", "")).lower()
            is_map = "maps correctly" in text
            is_landmark = bool(question.get("image_paths")) and not is_map
            if is_landmark:
                landmark_questions += 1
                if landmark_tag:
                    try:
                        selected = _gemini_index(split, video_id, str(question_id), landmark_tag)
                        landmark_coverage += 1
                    except (OSError, ValueError):
                        pass
            if include_motion and "compass" in classify_question_facets(question):
                compass_questions += 1
                record = motion_records.get(str(question_id), {})
                yaw = record.get("summary", {}).get("net_yaw_degrees") if record.get("status") == "ok" else None
                if yaw is not None:
                    candidate = compass_option_index(question, float(yaw))
                    if candidate is not None:
                        motion_coverage += 1
                        selected = int(candidate)
            if selected != base:
                changed.append({"video_id": video_id, "question_id": str(question_id),
                                "gemini": base, "motion": selected})
            options = question["options"]
            rows.append({"id": question.get("id", int(question_id)),
                         "answer": str(options[selected]), "answer_id": selected})
        output[video_id] = rows
    name = f"{split}_submission_gemini_motion_compass" if include_motion else f"{split}_submission_gemini"
    if base_tag:
        name += f"_{base_tag}"
    if landmark_tag:
        name += f"_{landmark_tag}"
    destination = ROOT / "predictions" / f"{name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    VOLUME.commit()
    return {"split": split, "path": str(destination), "questions": sum(map(len, output.values())),
            "include_motion": include_motion, "base_tag": base_tag,
            "compass_questions": compass_questions, "motion_coverage": motion_coverage,
            "landmark_questions": landmark_questions, "landmark_coverage": landmark_coverage,
            "changes": len(changed), "change_sample": changed[:20],
            "ground_truth_used_for_inference": False}


def _read_letter(split: str, video_id: str, question_id: str, tag: str) -> str | None:
    suffix = f"-{tag}" if tag else ""
    path = ROOT / "experiments" / "gemini-direct-video" / f"{split}-{video_id}-{question_id}{suffix}.json"
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    raw = str(record.get("raw_response", "")).strip().upper()
    return raw if raw in set("ABCDE") else None


@APP.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=10 * 60)
def build_adjudicated_submission(split: str = "test",
                                  target_file: str = "experiments/gemini-direct-video/adjudication-targets-2026-08-26.json",
                                  base_tag: str = "", new_tag: str = "leader-repro-clean",
                                  vote_tags: str = "adjudication-vote-1,adjudication-vote-2,adjudication-vote-3",
                                  out_name: str = "test_submission_adjudicated_2026-08-26") -> dict:
    """Merge base (580307) answers with a majority vote over disputed questions only.

    For each of the 263 disagreement questions, take a plurality among
    whichever of {base, new, vote-1, vote-2, vote-3} answers currently exist
    (some may be missing -- fail-fast retries leave gaps). A tie, or no votes
    at all beyond the base, keeps the base answer. All other (agreed)
    questions pass through the base answer unchanged.
    """
    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    targets = json.loads((ROOT / target_file.lstrip("/")).read_text())
    dispute_keys = {(target["video_id"], str(target["question_id"])) for target in targets}
    tags = [base_tag, new_tag] + [tag.strip() for tag in vote_tags.split(",") if tag.strip()]

    output: dict[str, list[dict]] = {}
    changed = 0
    missing_base_count = 0
    flips = []
    vote_coverage: Counter = Counter()
    for video_id, questions in manifest.items():
        rows = []
        for question_id, question in questions.items():
            qid = str(question_id)
            options = question["options"]
            base_letter = _read_letter(split, video_id, qid, base_tag)
            if base_letter is None:
                missing_base_count += 1
                continue
            selected_letter = base_letter
            if (video_id, qid) in dispute_keys:
                votes = [letter for tag in tags if (letter := _read_letter(split, video_id, qid, tag))]
                vote_coverage[len(votes)] += 1
                if votes:
                    counts = Counter(votes)
                    top_letter, top_n = counts.most_common(1)[0]
                    tied = [letter for letter, n in counts.items() if n == top_n]
                    winner = top_letter if len(tied) == 1 else base_letter
                    if winner != base_letter:
                        changed += 1
                        flips.append({"video_id": video_id, "question_id": qid,
                                     "old": base_letter, "new": winner, "votes": dict(counts)})
                    selected_letter = winner
            answer_id = "ABCDE".index(selected_letter)
            rows.append({"id": question.get("id", int(qid)), "answer": str(options[answer_id]),
                        "answer_id": answer_id})
        output[video_id] = rows

    total_rows = sum(len(rows) for rows in output.values())
    expected_rows = sum(len(questions) for questions in manifest.values())
    destination = ROOT / "predictions" / f"{out_name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    VOLUME.commit()
    return {"path": str(destination), "out_name": out_name, "total_rows": total_rows,
            "expected_rows": expected_rows, "missing_base_count": missing_base_count,
            "dispute_questions": len(dispute_keys), "changed": changed,
            "vote_coverage_histogram": dict(sorted(vote_coverage.items())),
            "sample_flips": flips[:20], "ground_truth_used_for_inference": False}


@APP.local_entrypoint()
def main(split: str = "test", landmark_tag: str = "", include_motion: bool = True,
         base_tag: str = "", action: str = "motion-compass",
         target_file: str = "experiments/gemini-direct-video/adjudication-targets-2026-08-26.json",
         out_name: str = "test_submission_adjudicated_2026-08-26"):
    if action == "adjudicated":
        result = build_adjudicated_submission.remote(split, target_file, base_tag or "",
                                                       "leader-repro-clean",
                                                       "adjudication-vote-1,adjudication-vote-2,adjudication-vote-3",
                                                       out_name)
    else:
        result = build_motion_compass.remote(split, landmark_tag, include_motion, base_tag)
    print(json.dumps(result, indent=2))
