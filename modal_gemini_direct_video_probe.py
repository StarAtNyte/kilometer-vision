"""Answer-blind Gemini native-video probe using silent exact-window clips."""
from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import modal

ROOT = Path("/data/2707-kilometer-vision")
SOURCE_ROOT = Path("/data/shared/walking-tours")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
# The pooled secret exposes GEMINI_API_KEYS and lets us rotate across the
# independently provisioned keys when one free-tier project is exhausted.
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


def _silent_full_source(source: Path, target: Path) -> None:
    """Remove audio while preserving the original video timestamps/frames.

    The leading public method is explicitly described as using silent GCS clips;
    this is distinct from our sparse storyboard approximation. Stream-copying
    avoids a multi-minute full-tour re-encode and preserves the model's native
    temporal sampling opportunity.
    """
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
                    "-map", "0:v:0", "-an", "-c:v", "copy", "-y", str(target)],
                   check=True, timeout=300)


def _sampled_full_source(source: Path, target: Path, fps: float = 0.02, scale: int = 320) -> None:
    """Encode the whole tour at a bounded temporal/spatial rate, silently."""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
                    "-an", "-vf", f"fps={fps},scale={scale}:-2", "-c:v", "libx264",
                    "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p", "-y", str(target)],
                   check=True, timeout=900)


def _upload_cached(client, clip: Path, cache_path: Path):
    """Reuse an ACTIVE Gemini File after transient generation failures."""
    from google.genai import types
    if cache_path.exists():
        try:
            saved = json.loads(cache_path.read_text())
            uploaded = client.files.get(name=saved["name"])
            state = getattr(uploaded, "state", None)
            if "ACTIVE" in getattr(state, "name", str(state)):
                return uploaded
        except Exception:
            pass
    uploaded = client.files.upload(file=str(clip), config={"mime_type": "video/mp4"})
    # Full-source tours can take several minutes for Gemini's video indexer;
    # do not confuse that with a failed model request.
    for _ in range(360):
        state = getattr(uploaded, "state", None)
        name = getattr(state, "name", str(state))
        if "ACTIVE" in name:
            cache_path.write_text(json.dumps({"name": uploaded.name, "uri": uploaded.uri}) + "\n")
            return uploaded
        if "FAILED" in name:
            raise RuntimeError(f"Gemini file processing failed: {name}")
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    raise TimeoutError("Gemini file did not become ACTIVE")


@APP.function(image=IMAGE, secrets=[SECRET], volumes={"/data": VOLUME}, timeout=20 * 60)
def run_one(video_id: str, question_id: str, split: str = "validation", model: str = MODEL, key_index: int = 0, full_video: bool = False, thinking_level: str = "high", fps: float = 0.5, scale: int = 320, context_frames: int = 24, two_stage: bool = False, full_source: bool = False, sampled_full: bool = False, prepare_only: bool = False) -> dict:
    from google import genai
    from google.genai import types
    from PIL import Image, ImageDraw

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    question = manifest[video_id][str(question_id)]
    source = _source(video_id)
    work = ROOT / "experiments" / "gemini-direct-video" / "clips"
    work.mkdir(parents=True, exist_ok=True)
    media_tag = "sampled-full-0p02" if sampled_full else "full-source" if full_source else "window"
    clip = work / f"{split}-{video_id}-{question_id}-{media_tag}.mp4"
    storyboard = []
    storyboard_clip = None
    if sampled_full:
        clip_start, clip_end = 0.0, 0.0
        if not clip.exists():
            _sampled_full_source(source, clip, fps=0.02, scale=min(scale, 320))
    elif full_source:
        clip_start, clip_end = 0.0, 0.0
        if not clip.exists():
            _silent_full_source(source, clip)
    elif full_video:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
                               check=True, capture_output=True, text=True)
        clip_start, clip_end = 0.0, float(probe.stdout.strip())
        # Seek directly to sparse timestamps. This preserves full-route context
        # without decoding/materializing a multi-gigabyte source clip.
        count = max(4, min(int(context_frames), 48))
        with tempfile.TemporaryDirectory() as frame_dir:
            for index in range(count):
                timestamp = clip_start + (clip_end - clip_start) * index / max(1, count - 1)
                frame = Path(frame_dir) / f"{index:03d}.jpg"
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(timestamp),
                                "-i", str(source), "-frames:v", "1", "-vf", f"scale={scale}:-2",
                                "-q:v", "3", "-y", str(frame)], check=True, timeout=120)
                if frame.exists():
                    storyboard.append((timestamp, frame.read_bytes()))
            if storyboard:
                # Re-encode the sparse chronological frames as a tiny silent
                # video. Gemini's temporal encoder can then see order/motion,
                # while avoiding a full-source transcode/upload.
                storyboard_clip = Path(tempfile.mktemp(prefix="kv-storyboard-", suffix=".mp4"))
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                                "-framerate", "2", "-i", str(Path(frame_dir) / "%03d.jpg"),
                                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                "-preset", "ultrafast", "-y", str(storyboard_clip)],
                               check=True, timeout=180)
    else:
        clip_start = float(question.get("start_seconds", 0.0)); clip_end = float(question.get("end_seconds", 60.0))
        _clip(source, clip_start, clip_end, clip, fps=fps, scale=scale)
    if prepare_only:
        result = {"video_id": video_id, "question_id": str(question_id), "clip": str(clip),
                  "bytes": clip.stat().st_size, "full_source": full_source,
                  "sampled_full": sampled_full,
                  "full_video": full_video, "ground_truth_used_for_inference": False}
        print(json.dumps(result, indent=2))
        return result
    client = genai.Client(api_key=_key(key_index), http_options=types.HttpOptions(timeout=180_000))
    if storyboard_clip is not None:
        uploaded = client.files.upload(file=str(storyboard_clip), config={"mime_type": "video/mp4"})
        for _ in range(60):
            state = getattr(uploaded, "state", None)
            name = getattr(state, "name", str(state))
            if "ACTIVE" in name:
                break
            if "FAILED" in name:
                raise RuntimeError(f"Gemini storyboard processing failed: {name}")
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        # Tell Gemini's File API how sparsely to tokenize the reassembled
        # storyboard; otherwise it silently defaults to 1 FPS.
        parts = [types.Part(
            file_data=types.FileData(file_uri=uploaded.uri, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=2.0),
        )]
    else:
        upload_cache = work / f"{split}-{video_id}-{question_id}.upload.json"
        uploaded = _upload_cached(client, clip, upload_cache)
        requested_fps = 0.02 if (sampled_full or full_source) else float(fps)
        parts = [types.Part(
            file_data=types.FileData(file_uri=uploaded.uri, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=requested_fps),
        )]
    route_summary = ""
    if two_stage:
        route_request = ("Inspect this walking-tour clip and produce a compact, answer-blind route certificate. "
                         "List only observable temporal facts: start and end scene, ordered sustained turns "
                         "(left/right), loops or crossings, and relative endpoint displacement. Do not choose "
                         "an answer option or infer candidate-map identities.")
        route_response = client.models.generate_content(
            model=model, contents=[types.Content(role="user", parts=[parts[0], types.Part.from_text(text=route_request)])],
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level=thinking_level), max_output_tokens=512))
        route_summary = getattr(route_response, "text", "") or ""
    map_hint = ("For a map question, first infer the ordered route from the video: starting position, "
                "ending position, major turns, and relative route shape. Then compare those four facts "
                "against every candidate map; do not use the candidate's screen position as evidence.\n"
                if question.get('image_paths') else "")
    if route_summary:
        map_hint += "A separate perception pass produced this route certificate; treat it as evidence, not as an answer label:\n" + route_summary + "\n"
    prompt = ("Answer the question from the walking-tour video. Reason silently, then return only the "
              "single option letter A, B, C, D, or E as the final response. Do not infer from option position alone.\n"
              + map_hint + f"Question: {question['question']}\nOptions (these letters match the printed candidate labels):\n" +
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
                image = image.convert("RGB")
                # Candidate maps can be multi-megapixel scans; keep their
                # geometry legible without letting five images exhaust the
                # model context before the video is read.
                image.thumbnail((max(256, scale), max(256, scale)), Image.Resampling.LANCZOS)
                buf = io.BytesIO(); image.save(buf, format="JPEG", quality=88)
            candidate_images.append(image.copy())
    if candidate_images:
        # Separate, explicitly labelled images reduce tiny-map contact-sheet loss.
        parts.append(types.Part.from_text(text="Candidate maps follow. Each image is labelled by its option letter; compare all five before answering."))
        for index, image in enumerate(candidate_images):
            buf = io.BytesIO(); image.save(buf, format="JPEG", quality=90)
            parts.append(types.Part.from_text(text=f"OPTION {'ABCDE'[index]} CANDIDATE MAP"))
            parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=prompt))
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        temperature=0.0,
        media_resolution=(types.MediaResolution.MEDIA_RESOLUTION_HIGH if scale >= 512 else types.MediaResolution.MEDIA_RESOLUTION_LOW),
        response_mime_type="text/x.enum",
        response_schema={"type": "STRING", "enum": ["A", "B", "C", "D", "E"]},
        max_output_tokens=1024)
    response = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model, contents=[types.Content(role="user", parts=parts)], config=config)
            break
        except Exception as error:
            # Provider 503s are transient; cap retries so quota/cost remains
            # predictable and never retry malformed requests (400) or quota
            # exhaustion (429).
            message = str(error)
            if "503" not in message or attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))
    result = {"video_id": video_id, "question_id": str(question_id), "model": model,
              "raw_response": response.text, "ground_truth_used_for_inference": False}
    output = ROOT / "experiments" / "gemini-direct-video" / f"{split}-{video_id}-{question_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@APP.local_entrypoint()
def main(video_id: str = "4MYTCny_FZs", question_id: str = "0", split: str = "validation", model: str = MODEL, key_index: int = 0, full_video: bool = False, thinking_level: str = "high", fps: float = 0.5, scale: int = 320, context_frames: int = 24, two_stage: bool = False, full_source: bool = False, sampled_full: bool = False, prepare_only: bool = False):
    print(json.dumps(run_one.remote(video_id, question_id, split, model, key_index, full_video, thinking_level, fps, scale, context_frames, two_stage, full_source, sampled_full, prepare_only), indent=2))
