"""Answer-blind Gemini native-video probe using silent exact-window clips."""
from __future__ import annotations

import io
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

import modal

ROOT = Path("/data/2707-kilometer-vision")
SOURCE_ROOT = Path("/data/shared/walking-tours")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
SECRET = modal.Secret.from_name("gemini-api-keys")
MODEL = "gemini-3.5-flash"
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
IMAGE = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("google-genai", "pillow") )
APP = modal.App("kilometervision-gemini-direct-video")


def _key(index: int = 0) -> str:
    pooled = [x.strip() for x in os.environ.get("GEMINI_API_KEYS", "").split(",") if x.strip()]
    if pooled:
        return pooled[index % len(pooled)]
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "API_KEY"):
        if os.environ.get(name): return os.environ[name]
    raise RuntimeError("Gemini key secret has no supported variable")


def _source(video_id: str) -> Path:
    directory = SOURCE_ROOT / video_id
    return next(p for p in sorted(directory.iterdir())
                if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def _clip(source: Path, start: float, end: float, target: Path, fps: float = 0.5, scale: int = 320) -> None:
    duration = max(1.0, end - start)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                    "-i", str(source), "-t", str(duration), "-an", "-vf", f"fps={fps},scale={scale}:-2",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-y", str(target)],
                   check=True, timeout=max(120, int(duration * 3)))


@APP.function(image=IMAGE, secrets=[SECRET], volumes={"/data": VOLUME}, timeout=20 * 60)
def run_one(video_id: str, question_id: str, split: str = "validation", model: str = MODEL, key_index: int = 0, full_video: bool = False, thinking_level: str = "high", fps: float = 0.5, scale: int = 320) -> dict:
    from google import genai
    from google.genai import types
    from PIL import Image, ImageDraw

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    question = manifest[video_id][str(question_id)]
    source = _source(video_id)
    work = ROOT / "experiments" / "gemini-direct-video" / "clips"
    work.mkdir(parents=True, exist_ok=True)
    clip = work / f"{split}-{video_id}-{question_id}.mp4"
    if full_video:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
                               check=True, capture_output=True, text=True)
        clip_start, clip_end = 0.0, float(probe.stdout.strip())
    else:
        clip_start = float(question.get("start_seconds", 0.0)); clip_end = float(question.get("end_seconds", 60.0))
    _clip(source, clip_start, clip_end, clip, fps=fps, scale=scale)
    client = genai.Client(api_key=_key(key_index), http_options=types.HttpOptions(timeout=180_000))
    uploaded = client.files.upload(file=str(clip), config={"mime_type": "video/mp4"})
    for _ in range(60):
        state = getattr(uploaded, "state", None)
        name = getattr(state, "name", str(state))
        if "ACTIVE" in name:
            break
        if "FAILED" in name:
            raise RuntimeError(f"Gemini file processing failed: {name}")
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    parts = [types.Part.from_uri(file_uri=uploaded.uri, mime_type="video/mp4")]
    prompt = ("Answer the question from the walking-tour video. Reason silently, then return only the "
              "single option letter A, B, C, D, or E as the final response. Do not infer from option position alone.\n"
              f"Question: {question['question']}\nOptions (these letters match the printed candidate labels):\n" +
              "\n".join(f"{'ABCDE'[i]}. {option}" for i, option in enumerate(question.get("options", []))))
    archive_path = ROOT / "manifests" / f"{split}_images.zip"
    candidate_images = []
    with zipfile.ZipFile(archive_path) as archive:
        for index, reference in enumerate(question.get("image_paths", [])):
            matches = [name for name in archive.namelist() if Path(name).name == Path(str(reference)).name]
            if len(matches) != 1:
                continue
            raw = archive.read(matches[0])
            with Image.open(io.BytesIO(raw)) as image:
                image = image.convert("RGB"); buf = io.BytesIO(); image.save(buf, format="JPEG", quality=88)
            candidate_images.append(image.copy())
    if candidate_images:
        # Separate, explicitly labelled images reduce tiny-map contact-sheet loss.
        parts.append(types.Part.from_text(text="Candidate maps follow. Each image is labelled by its option letter; compare all five before answering."))
        for index, image in enumerate(candidate_images):
            buf = io.BytesIO(); image.save(buf, format="JPEG", quality=90)
            parts.append(types.Part.from_text(text=f"OPTION {'ABCDE'[index]} CANDIDATE MAP"))
            parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=prompt))
    response = client.models.generate_content(
        model=model, contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
            media_resolution=(types.MediaResolution.MEDIA_RESOLUTION_MEDIUM if scale >= 512 else types.MediaResolution.MEDIA_RESOLUTION_LOW),
            max_output_tokens=128))
    result = {"video_id": video_id, "question_id": str(question_id), "model": model,
              "raw_response": response.text, "ground_truth_used_for_inference": False}
    output = ROOT / "experiments" / "gemini-direct-video" / f"{split}-{video_id}-{question_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@APP.local_entrypoint()
def main(video_id: str = "4MYTCny_FZs", question_id: str = "0", split: str = "validation", key_index: int = 0, full_video: bool = False, thinking_level: str = "high", fps: float = 0.5, scale: int = 320):
    print(json.dumps(run_one.remote(video_id, question_id, split, MODEL, key_index, full_video, thinking_level, fps, scale), indent=2))
