"""OCR audit for candidate-map labels and question-window video text."""
from __future__ import annotations

import io
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import modal

from kilometer_vision.question_planner import classify_question_facets


ROOT = Path("/data/2707-kilometer-vision")
SOURCE_ROOT = Path("/data/shared/walking-tours")
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "tesseract-ocr")
    .pip_install("pillow", "pytesseract")
    .add_local_python_source('kilometer_vision')
)
app = modal.App("kilometervision-map-ocr")


def _source(video_id: str) -> Path | None:
    directory = SOURCE_ROOT / video_id
    if not directory.is_dir():
        return None
    return next(
        (path for path in sorted(directory.iterdir())
         if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and ".partial" not in path.stem),
        None,
    )


def _clean(text: str) -> list[str]:
    rows = []
    for line in text.splitlines():
        cleaned = re.sub(r"[^\w .&'/-]+", " ", line).strip()
        if len(cleaned) >= 3 and any(character.isalpha() for character in cleaned):
            rows.append(cleaned)
    return rows


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=20 * 60)
def candidate_ocr(split: str = "validation", video_id: str = "__all__") -> dict:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    archive_path = ROOT / "manifests" / f"{split}_images.zip"
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        targets = manifest.items() if video_id == "__all__" else [(video_id, manifest.get(video_id, {}))]
        for target_id, questions in targets:
            for question_id, question in questions.items():
                if "map" not in classify_question_facets(question):
                    continue
                candidates = []
                for reference in question.get("image_paths", []):
                    matches = [name for name in members if Path(name).name == Path(str(reference)).name]
                    if len(matches) != 1:
                        candidates.append([]); continue
                    image = Image.open(io.BytesIO(archive.read(matches[0]))).convert("RGB")
                    enlarged = image.resize((image.width * 3, image.height * 3))
                    enlarged = ImageEnhance.Contrast(enlarged).enhance(1.5).filter(ImageFilter.SHARPEN)
                    variants = []
                    for psm in (6, 11, 12):
                        variants.extend(_clean(pytesseract.image_to_string(enlarged, config=f"--psm {psm}")))
                    candidates.append(sorted(set(variants), key=str.lower))
                records.append({"video_id": target_id, "question_id": str(question_id), "answer_id": question.get("answer_id"), "candidates": candidates})
    result = {"split": split, "records": records}
    suffix = "all" if video_id == "__all__" else video_id
    output = ROOT / "experiments" / "map-ocr" / "candidates" / split / f"{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@app.function(image=IMAGE, volumes={"/data": VOLUME}, cpu=4.0, timeout=2 * 60 * 60)
def window_ocr(split: str = "validation", video_id: str = "__all__", sample_seconds: float = 5.0) -> dict:
    """OCR sparse frames from exact map-question windows.

    This is a frozen retrieval diagnostic: no answer labels influence sampling,
    OCR, or scoring. Both full frames and an upper scene crop are read because
    street/shop signs are commonly above the walking-tour vanishing point.
    """
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    records = []
    targets = manifest.items() if video_id == "__all__" else [(video_id, manifest.get(video_id, {}))]
    for target_id, questions in targets:
        source = _source(target_id)
        for question_id, question in questions.items():
            if "map" not in classify_question_facets(question):
                continue
            start = float(question.get("start_seconds", 0.0))
            end = float(question.get("end_seconds", start))
            rows = []
            if source is not None and end > start:
                with tempfile.TemporaryDirectory() as temporary:
                    pattern = Path(temporary) / "%06d.jpg"
                    command = [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-ss", str(start), "-t", str(end - start), "-i", str(source),
                        "-vf", f"fps=1/{sample_seconds},scale=960:-2", "-q:v", "3", str(pattern),
                    ]
                    subprocess.run(command, check=True, timeout=max(120, int(end - start) * 2))
                    for index, path in enumerate(sorted(Path(temporary).glob("*.jpg"))):
                        image = Image.open(path).convert("RGB")
                        crops = (image, image.crop((0, 0, image.width, int(image.height * 0.72))))
                        found = []
                        for crop in crops:
                            enhanced = ImageEnhance.Contrast(crop).enhance(1.5).filter(ImageFilter.SHARPEN)
                            for psm in (6, 11):
                                found.extend(_clean(pytesseract.image_to_string(enhanced, config=f"--psm {psm}")))
                        if found:
                            rows.append({
                                "timestamp_seconds": start + index * sample_seconds,
                                "text": sorted(set(found), key=str.lower),
                            })
            records.append({
                "video_id": target_id, "question_id": str(question_id),
                "answer_id": question.get("answer_id"), "start_seconds": start,
                "end_seconds": end, "sample_seconds": sample_seconds, "frames_with_text": rows,
            })
    result = {"split": split, "records": records}
    suffix = "all" if video_id == "__all__" else video_id
    output = ROOT / "experiments" / "map-ocr" / f"{split}-windows-{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=300)
def map_video_ids(split: str) -> list[str]:
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    return [video_id for video_id, questions in manifest.items()
            if any("map" in classify_question_facets(question) for question in questions.values())]


@app.local_entrypoint()
def main(split: str = "validation", mode: str = "candidates", sample_seconds: float = 5.0) -> None:
    if mode == "windows":
        map_ids = map_video_ids.remote(split)
        calls = [window_ocr.spawn(split, video_id, sample_seconds) for video_id in map_ids]
        partial = [call.get() for call in calls]
        result = {"split": split, "records": [record for item in partial for record in item["records"]]}
    else:
        map_ids = map_video_ids.remote(split)
        calls = [candidate_ocr.spawn(split, video_id) for video_id in map_ids]
        partial = [call.get() for call in calls]
        result = {"split": split, "records": [record for item in partial for record in item["records"]]}
    print(json.dumps(result, indent=2))
