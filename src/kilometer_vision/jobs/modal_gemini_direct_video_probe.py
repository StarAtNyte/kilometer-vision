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
# Select the Modal secret at launch via GEMINI_MODAL_SECRET; never store key
# material in source. The default preserves the pooled production secret.
SECRET = modal.Secret.from_name(os.environ.get("GEMINI_MODAL_SECRET", "gemini-api-keys"))
MODEL = "gemini-3.7-flash"
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
IMAGE = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("google-genai", "pillow", "google-auth") )
APP = modal.App("kilometervision-gemini-direct-video")

# Vertex AI path: bypasses the AI-Studio free-tier per-key request cap by
# billing through a real GCP project instead. The GCP secret must always be
# part of the function's declared deps (not conditional on an env var) --
# Modal re-hydrates the function inside the remote container, where a plain
# os.environ flag set only on the local `modal run` invocation is not visible,
# so a conditional deps list disagrees between local registration and remote
# hydration. Whether to actually use it is instead a per-call `use_vertex`
# argument, threaded through normally like any other parameter.
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "veotest-460812")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
GCP_SECRET_NAME = os.environ.get("GCP_MODAL_SECRET", "gcp-vertex-service-account")
SECRETS = [SECRET, modal.Secret.from_name(GCP_SECRET_NAME)]


def _is_map_question(question: dict) -> bool:
    """Map candidates have five route images; landmark questions may also have an image."""
    return "maps correctly" in str(question.get("question", "")).lower()


def _dino_landmark_timestamps(video_id: str, question_id: str, split: str, limit: int = 6) -> list[float]:
    """Read frozen answer-blind landmark retrieval evidence, if available."""
    path = ROOT / "experiments" / "landmark-dinov2" / split / f"{video_id}.json"
    if not path.exists():
        return []
    try:
        record = json.loads(path.read_text()).get(str(question_id), {})
    except (OSError, ValueError):
        return []
    ranked = sorted(
        record.get("top_matches", []),
        key=lambda item: (
            float(item.get("patch_local_similarity", 0.0) or 0.0),
            float(item.get("mutual_patch_ratio", 0.0) or 0.0),
        ),
        reverse=True,
    )
    selected: list[float] = []
    for item in ranked:
        try:
            timestamp = float(item["timestamp_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        # Avoid spending the evidence budget on adjacent copies of one frame.
        if all(abs(timestamp - existing) >= 4.0 for existing in selected):
            selected.append(timestamp)
        if len(selected) >= limit:
            break
    return sorted(selected)


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


def _clip(source: Path, start: float, end: float, target: Path, fps: float = 2.0, scale: int = 320) -> None:
    duration = max(1.0, end - start)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                    "-i", str(source), "-t", str(duration), "-an", "-vf", f"fps={fps},scale={scale}:-2",
                   "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-y", str(target)],
                   check=True, timeout=max(120, int(duration * 3)))


def _valid_video_file(path: Path) -> bool:
    """Reject failed ffmpeg placeholders before treating a cache hit as valid."""
    if not path.exists() or path.stat().st_size < 4096:
        return False
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], capture_output=True, text=True)
    if probe.returncode != 0:
        return False
    try:
        return float(probe.stdout.strip()) > 0.5
    except ValueError:
        return False


def _cap_inline_video(path: Path, max_bytes: int = 10 * 1024 * 1024) -> None:
    """Bound Vertex inline payload bytes without changing frame rate or dimensions."""
    if path.stat().st_size <= max_bytes:
        return
    for crf in (28, 32, 36):
        temporary = path.with_name(f"{path.stem}.crf{crf}.partial.mp4")
        try:
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-y", str(temporary),
            ], check=True, timeout=900)
            if temporary.stat().st_size < path.stat().st_size:
                os.replace(temporary, path)
            if path.stat().st_size <= max_bytes:
                return
        finally:
            temporary.unlink(missing_ok=True)
    raise ValueError(f"clip remains too large for inline Vertex request: {path.stat().st_size} bytes")


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


@APP.function(image=IMAGE, secrets=SECRETS, volumes={"/data": VOLUME}, timeout=20 * 60)
def run_one(video_id: str, question_id: str, split: str = "validation", model: str = MODEL, key_index: int = 0, full_video: bool = False, thinking_level: str = "high", fps: float = 2.0, scale: int = 320, context_frames: int = 24, two_stage: bool = False, full_source: bool = False, sampled_full: bool = False, prepare_only: bool = False, use_vertex: bool = False, force: bool = False, record_tag: str = "", adaptive_landmark: bool = False) -> dict:
    from google import genai
    from google.genai import types
    from PIL import Image, ImageDraw

    VOLUME.reload()
    suffix = f"-{record_tag}" if record_tag else ""
    output_path = ROOT / "experiments" / "gemini-direct-video" / f"{split}-{video_id}-{question_id}{suffix}.json"
    if output_path.exists() and not force:
        try:
            cached = json.loads(output_path.read_text())
            raw = str(cached.get("raw_response", "")).strip().upper()
            if raw in set("ABCDE"):
                return cached
        except (OSError, ValueError):
            pass
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    question = manifest[video_id][str(question_id)]
    is_map = _is_map_question(question)
    is_landmark = bool(question.get("image_paths")) and not is_map
    source = _source(video_id)
    work = ROOT / "experiments" / "gemini-direct-video" / "clips"
    work.mkdir(parents=True, exist_ok=True)
    fps_tag = str(float(fps)).replace(".", "p")
    media_tag = ("sampled-full-0p02" if sampled_full else "full-source" if full_source
                 else f"window-fps{fps_tag}-scale{int(scale)}")
    clip = work / f"{split}-{video_id}-{question_id}-{media_tag}.mp4"
    storyboard = []
    storyboard_clip = None
    if sampled_full:
        clip_start, clip_end = 0.0, 0.0
        if not _valid_video_file(clip):
            _sampled_full_source(source, clip, fps=0.02, scale=min(scale, 320))
    elif full_source:
        clip_start, clip_end = 0.0, 0.0
        if not _valid_video_file(clip):
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
        if not _valid_video_file(clip):
            _clip(source, clip_start, clip_end, clip, fps=fps, scale=scale)
    if prepare_only:
        result = {"video_id": video_id, "question_id": str(question_id), "clip": str(clip),
                  "bytes": clip.stat().st_size, "full_source": full_source,
                  "sampled_full": sampled_full,
                  "full_video": full_video, "ground_truth_used_for_inference": False}
        print(json.dumps(result, indent=2))
        return result
    if use_vertex:
        creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not creds_json:
            raise RuntimeError(f"use_vertex=True but secret {GCP_SECRET_NAME!r} has no GOOGLE_APPLICATION_CREDENTIALS_JSON")
        creds_path = Path(tempfile.gettempdir()) / "gcp-vertex-sa.json"
        creds_path.write_text(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
        client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION,
                               http_options=types.HttpOptions(timeout=180_000))
    else:
        client = genai.Client(api_key=_key(key_index), http_options=types.HttpOptions(timeout=180_000))
    if use_vertex:
        # Vertex AI has no Files API (client.files.upload only works on the
        # Gemini Developer/AI-Studio client) -- send the clip inline instead.
        # These are short, low-res, silent clips, so they stay well under
        # Vertex's inline request-size limit.
        target_clip = storyboard_clip if storyboard_clip is not None else clip
        _cap_inline_video(target_clip)
        requested_fps = 2.0 if storyboard_clip is not None else (0.02 if (sampled_full or full_source) else float(fps))
        parts = [types.Part(
            inline_data=types.Blob(data=target_clip.read_bytes(), mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=requested_fps),
        )]
    elif storyboard_clip is not None:
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
        upload_cache = work / f"{split}-{video_id}-{question_id}-{media_tag}.upload.json"
        uploaded = _upload_cached(client, clip, upload_cache)
        requested_fps = 0.02 if (sampled_full or full_source) else float(fps)
        parts = [types.Part(
            file_data=types.FileData(file_uri=uploaded.uri, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=requested_fps),
        )]
    route_summary = ""
    if two_stage:
        route_request = ("Inspect this walking-tour clip and produce a compact, answer-blind TRACE-style route state. "
                         "Return strict JSON with these fields: meta_context, ordered_turns, crossings_or_loops, "
                         "camera_trajectory, endpoint_displacement, visible_landmarks, uncertainty, and evidence_timestamps. "
                         "For ordered_turns, distinguish sustained path turns from brief head/camera pans. "
                         "Use only observable temporal facts; do not choose an answer option or infer candidate-map identities.")
        route_response = client.models.generate_content(
            model=model, contents=[types.Content(role="user", parts=[parts[0], types.Part.from_text(text=route_request)])],
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level=thinking_level), max_output_tokens=512))
        route_summary = getattr(route_response, "text", "") or ""
    map_hint = ("For a map question, first infer the ordered route from the video: starting position, "
                "ending position, major turns, and relative route shape. Then compare those four facts "
                "against every candidate map; do not use the candidate's screen position as evidence.\n"
                if is_map else "")
    if route_summary:
        map_hint += "A separate perception pass produced this route certificate; treat it as evidence, not as an answer label:\n" + route_summary + "\n"
    prompt = ("Answer the question from the walking-tour video. Reason silently, then return only the "
              "single option letter A, B, C, D, or E as the final response. Do not infer from option position alone.\n"
              + map_hint + f"Question: {question['question']}\nOptions (these letters match the printed candidate labels):\n" +
              "\n".join(f"{'ABCDE'[i]}. {option}" for i, option in enumerate(question.get("options", []))))
    archive_path = ROOT / "manifests" / f"{split}_images.zip"
    supplied_images = []
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
                image_limit = max(640, scale) if is_landmark and adaptive_landmark else max(256, scale)
                image.thumbnail((image_limit, image_limit), Image.Resampling.LANCZOS)
                buf = io.BytesIO(); image.save(buf, format="JPEG", quality=88)
            supplied_images.append(image.copy())
    if is_map and supplied_images:
        # Separate, explicitly labelled images reduce tiny-map contact-sheet loss.
        parts.append(types.Part.from_text(text="Candidate maps follow. Each image is labelled by its option letter; compare all five before answering."))
        for index, image in enumerate(supplied_images):
            buf = io.BytesIO(); image.save(buf, format="JPEG", quality=90)
            parts.append(types.Part.from_text(text=f"OPTION {'ABCDE'[index]} CANDIDATE MAP"))
            parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    elif is_landmark and supplied_images:
        parts.append(types.Part.from_text(text="REFERENCE LANDMARK IMAGE supplied by the question. Search for this place in the walking-tour video; it is not an answer option or a candidate map."))
        buf = io.BytesIO(); supplied_images[0].save(buf, format="JPEG", quality=94)
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
        if adaptive_landmark:
            evidence_timestamps = _dino_landmark_timestamps(video_id, str(question_id), split)
            if evidence_timestamps:
                parts.append(types.Part.from_text(text="Retrieved chronological candidate frames follow. Timestamps are absolute video times. Retrieval is approximate evidence: verify each frame visually and allow that the reference may be absent."))
                for timestamp in evidence_timestamps:
                    frame_path = Path(tempfile.mktemp(prefix="kv-landmark-", suffix=".jpg"))
                    try:
                        subprocess.run([
                            "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(timestamp),
                            "-i", str(source), "-frames:v", "1", "-vf", "scale=640:-2",
                            "-q:v", "2", "-y", str(frame_path),
                        ], check=True, timeout=120)
                        evidence = Image.open(frame_path).convert("RGB")
                        draw = ImageDraw.Draw(evidence)
                        draw.rectangle((0, 0, 260, 30), fill=(0, 0, 0))
                        draw.text((8, 7), f"VIDEO TIMESTAMP {timestamp:.1f}s", fill=(255, 255, 255))
                        buf = io.BytesIO(); evidence.save(buf, format="JPEG", quality=94)
                        parts.append(types.Part.from_text(text=f"RETRIEVED VIDEO FRAME AT {timestamp:.1f} SECONDS"))
                        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
                    finally:
                        frame_path.unlink(missing_ok=True)
    parts.append(types.Part.from_text(text=prompt))
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        temperature=0.0,
        media_resolution=(types.MediaResolution.MEDIA_RESOLUTION_HIGH
                          if scale >= 512 or (is_landmark and adaptive_landmark)
                          else types.MediaResolution.MEDIA_RESOLUTION_LOW),
        response_mime_type="text/x.enum",
        response_schema={"type": "STRING", "enum": ["A", "B", "C", "D", "E"]},
        max_output_tokens=1024)
    response = None
    retry_deadline = time.monotonic() + 900  # stay well under the 20min function timeout
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.models.generate_content(
                model=model, contents=[types.Content(role="user", parts=parts)], config=config)
            break
        except Exception as error:
            # Keep retrying capacity errors until success or the deadline --
            # a full sweep is expensive to re-launch, so prefer to camp on a
            # single item over Vertex congestion rather than leave it empty.
            # Never retry a malformed request (400).
            message = str(error)
            transient = "503" in message or "429" in message or "504" in message
            if not transient or time.monotonic() >= retry_deadline:
                raise
            time.sleep(min(20 * attempt, 60))
    result = {"video_id": video_id, "question_id": str(question_id), "model": model,
              "raw_response": response.text, "ground_truth_used_for_inference": False,
              "settings": {"fps": float(fps), "scale": int(scale), "thinking_level": thinking_level,
                           "two_stage": bool(two_stage), "adaptive_landmark": bool(adaptive_landmark),
                           "record_tag": record_tag}}
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@APP.local_entrypoint()
def main(video_id: str = "4MYTCny_FZs", question_id: str = "0", split: str = "validation", model: str = MODEL, key_index: int = 0, full_video: bool = False, thinking_level: str = "high", fps: float = 2.0, scale: int = 320, context_frames: int = 24, two_stage: bool = False, full_source: bool = False, sampled_full: bool = False, prepare_only: bool = False, use_vertex: bool = False, force: bool = False, record_tag: str = "", adaptive_landmark: bool = False):
    print(json.dumps(run_one.remote(video_id, question_id, split, model, key_index, full_video, thinking_level, fps, scale, context_frames, two_stage, full_source, sampled_full, prepare_only, use_vertex, force, record_tag, adaptive_landmark), indent=2))


@APP.local_entrypoint()
def run_all(split: str = "test", model: str = "gemini-3.7-flash", fps: float = 2.0, scale: int = 320,
            thinking_level: str = "low", use_vertex: bool = True, limit: int = 0, offset: int = 0,
            sleep_s: float = 1.0, two_stage_maps: bool = False, adaptive_landmarks: bool = False,
            record_tag: str = "", only_family: str = ""):
    """Sequential answer-blind pass over every question in the split.

    Each call is idempotent (run_one skips and returns the cached record if
    its output file already exists), so an interrupted run can simply be
    re-launched with the same arguments to resume from where it left off,
    without re-billing already-answered questions.
    """
    cache_name = "test_manifest.json" if split == "test" else "validation_manifest_latest.json"
    manifest = json.loads((Path(".cache/remote_probe") / cache_name).read_text())
    targets = [(video_id, question_id) for video_id, questions in manifest.items()
               for question_id in questions if isinstance(questions, dict)]
    if only_family:
        wanted = only_family.strip().lower()
        targets = [(video_id, question_id) for video_id, question_id in targets
                   if (("map" if _is_map_question(manifest[video_id][question_id]) else
                        "landmark" if manifest[video_id][question_id].get("image_paths") else
                        "compass" if "direction" in str(manifest[video_id][question_id].get("question", "")).lower() else
                        "distance") == wanted)]
    targets = targets[offset:(offset + limit) if limit else None]
    total = len(targets)
    ok = err = 0
    for index, (video_id, question_id) in enumerate(targets):
        try:
            question = manifest[video_id][question_id]
            is_map = _is_map_question(question)
            run_one.remote(video_id, question_id, split=split, model=model, thinking_level=thinking_level,
                            fps=fps, scale=scale, use_vertex=use_vertex,
                            two_stage=bool(two_stage_maps and is_map),
                            adaptive_landmark=bool(adaptive_landmarks and question.get("image_paths") and not is_map),
                            record_tag=record_tag)
            ok += 1
            status = "ok"
        except Exception as error:
            err += 1
            status = f"error:{type(error).__name__}: {error}"
        print(f"[{index + 1}/{total}] {video_id}/{question_id} -> {status}  (ok={ok} err={err})", flush=True)
        if sleep_s:
            time.sleep(sleep_s)
    print(json.dumps({"total": total, "ok": ok, "err": err}, indent=2))


@APP.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=24 * 3600)
def run_cooled_landmark_batch(split: str = "test", model: str = "gemini-3.7-flash",
                              fps: float = 2.0, scale: int = 320,
                              thinking_level: str = "low", cooldown_s: float = 8.0,
                              record_tag: str = "landmark-packaging-fix") -> dict:
    """Persistent single-worker landmark sweep with provider-friendly pacing."""
    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    targets = [
        (video_id, question_id)
        for video_id, questions in manifest.items()
        for question_id, question in questions.items()
        if question.get("image_paths") and not _is_map_question(question)
    ]
    ok = err = 0
    failures = []
    for index, (video_id, question_id) in enumerate(targets):
        try:
            run_one.remote(
                video_id, question_id, split=split, model=model,
                thinking_level=thinking_level, fps=fps, scale=scale,
                use_vertex=True, record_tag=record_tag,
            )
            ok += 1
        except Exception as error:
            err += 1
            failures.append({"video_id": video_id, "question_id": str(question_id),
                             "error": f"{type(error).__name__}: {error}"})
        print(f"[{index + 1}/{len(targets)}] {video_id}/{question_id} ok={ok} err={err}", flush=True)
        if cooldown_s:
            time.sleep(cooldown_s)
    return {"total": len(targets), "ok": ok, "err": err, "failures": failures,
            "record_tag": record_tag, "ground_truth_used_for_inference": False}


@APP.local_entrypoint()
def start_cooled_landmarks(split: str = "test", cooldown_s: float = 8.0,
                           record_tag: str = "landmark-packaging-fix"):
    call = run_cooled_landmark_batch.spawn(split=split, cooldown_s=cooldown_s,
                                           record_tag=record_tag)
    print(json.dumps({"function_call_id": call.object_id, "split": split,
                      "cooldown_s": cooldown_s, "record_tag": record_tag}, indent=2))


@APP.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=24 * 3600)
def run_cooled_direct_batch(split: str = "test", model: str = "gemini-3.7-flash",
                            fps: float = 2.0, scale: int = 320,
                            thinking_level: str = "low", cooldown_s: float = 8.0,
                            record_tag: str = "leader-repro-clean",
                            video_id: str = "__all__") -> dict:
    """Persistent pure-Gemini sweep; failures remain missing for later retry."""
    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    requested_videos = ({item.strip() for item in video_id.split(",") if item.strip()}
                        if video_id != "__all__" else set())
    targets = [(target_video_id, question_id)
               for target_video_id, questions in manifest.items()
               if video_id == "__all__" or target_video_id in requested_videos
               for question_id in questions]
    ok = err = 0
    failures = []
    for index, (video_id, question_id) in enumerate(targets):
        try:
            record = run_one.remote(
                video_id, question_id, split=split, model=model,
                thinking_level=thinking_level, fps=fps, scale=scale,
                use_vertex=True, record_tag=record_tag,
                two_stage=False, adaptive_landmark=False,
            )
            raw = str(record.get("raw_response", "")).strip().upper()
            if raw not in set("ABCDE"):
                raise ValueError(f"unparseable Gemini response: {raw!r}")
            ok += 1
        except Exception as error:
            err += 1
            failures.append({"video_id": video_id, "question_id": str(question_id),
                             "error": f"{type(error).__name__}: {error}"})
        print(f"[{index + 1}/{len(targets)}] {video_id}/{question_id} ok={ok} err={err}", flush=True)
        if cooldown_s:
            time.sleep(cooldown_s)
    return {"total": len(targets), "ok": ok, "err": err, "failures": failures,
            "settings": {"model": model, "fps": fps, "scale": scale,
                         "thinking_level": thinking_level, "cooldown_s": cooldown_s,
                         "record_tag": record_tag, "fallback": False,
                         "video_id": video_id},
            "ground_truth_used_for_inference": False}


@APP.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=24 * 3600)
def run_targeted_vote_batch(target_file: str, split: str = "test",
                            model: str = "gemini-3.7-flash",
                            fps: float = 2.0, scale: int = 320,
                            thinking_level: str = "high", cooldown_s: float = 8.0,
                            record_tag: str = "adjudication-vote",
                            two_stage_maps: bool = True,
                            adaptive_landmarks: bool = True) -> dict:
    """Run one answer-blind vote over an explicit question-ID list."""
    VOLUME.reload()
    path = ROOT / target_file.lstrip("/")
    targets = json.loads(path.read_text())
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    ok = err = 0
    failures = []
    for index, target in enumerate(targets):
        video_id = str(target["video_id"])
        question_id = str(target["question_id"])
        question = manifest[video_id][question_id]
        is_map = _is_map_question(question)
        is_landmark = bool(question.get("image_paths")) and not is_map
        try:
            record = run_one.remote(
                video_id, question_id, split=split, model=model,
                thinking_level=thinking_level, fps=fps, scale=scale,
                use_vertex=True, record_tag=record_tag,
                two_stage=bool(two_stage_maps and is_map),
                adaptive_landmark=bool(adaptive_landmarks and is_landmark),
            )
            raw = str(record.get("raw_response", "")).strip().upper()
            if raw not in set("ABCDE"):
                raise ValueError(f"unparseable Gemini response: {raw!r}")
            ok += 1
        except Exception as error:
            err += 1
            failures.append({"video_id": video_id, "question_id": question_id,
                             "error": f"{type(error).__name__}: {error}"})
        print(f"[{index + 1}/{len(targets)}] {video_id}/{question_id} ok={ok} err={err}", flush=True)
        if cooldown_s:
            time.sleep(cooldown_s)
    return {"total": len(targets), "ok": ok, "err": err, "failures": failures,
            "settings": {"model": model, "fps": fps, "scale": scale,
                         "thinking_level": thinking_level, "cooldown_s": cooldown_s,
                         "record_tag": record_tag, "two_stage_maps": two_stage_maps,
                         "adaptive_landmarks": adaptive_landmarks},
            "target_file": target_file, "ground_truth_used_for_inference": False}


@APP.local_entrypoint()
def start_cooled_direct(split: str = "test", cooldown_s: float = 8.0,
                        record_tag: str = "leader-repro-clean",
                        video_id: str = "__all__"):
    call = run_cooled_direct_batch.spawn(split=split, cooldown_s=cooldown_s,
                                         record_tag=record_tag, video_id=video_id)
    print(json.dumps({"function_call_id": call.object_id, "split": split,
                      "cooldown_s": cooldown_s, "record_tag": record_tag,
                      "video_id": video_id, "fallback": False}, indent=2))


@APP.local_entrypoint()
def start_targeted_vote(target_file: str, split: str = "test",
                        fps: float = 2.0, scale: int = 320,
                        thinking_level: str = "high", cooldown_s: float = 8.0,
                        record_tag: str = "adjudication-vote",
                        two_stage_maps: bool = True,
                        adaptive_landmarks: bool = True):
    call = run_targeted_vote_batch.spawn(
        target_file=target_file, split=split, fps=fps, scale=scale,
        thinking_level=thinking_level, cooldown_s=cooldown_s,
        record_tag=record_tag, two_stage_maps=two_stage_maps,
        adaptive_landmarks=adaptive_landmarks)
    print(json.dumps({"function_call_id": call.object_id, "target_file": target_file,
                      "split": split, "fps": fps, "scale": scale,
                      "thinking_level": thinking_level, "record_tag": record_tag,
                      "two_stage_maps": two_stage_maps,
                      "adaptive_landmarks": adaptive_landmarks}, indent=2))
