"""Label-safe strong Gemini map-reasoning experiments."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import modal

from map_solver import orient_path_from_start, route_marker_positions, route_path
from question_planner import classify_question_facets


SECRET = modal.Secret.from_name("gemini-api-key")
ROOT = Path("/data/2707-kilometer-vision")
SOURCE_ROOT = Path("/data/shared/walking-tours")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
# The configured API project exposes 3.1 Pro but assigns it zero free-tier
# request/input-token quota. Keep the frozen gate on the strongest usable
# native-video Flash endpoint instead.
MODEL = "gemini-3.5-flash"
IMAGE = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("google-genai", "pillow", "numpy==1.26.4", "opencv-python-headless")
         .add_local_python_source("map_solver", "question_planner"))
app = modal.App("kilometervision-gemini-geometry")


def _api_key() -> str:
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    raise RuntimeError("gemini-api-key secret has no supported key name")


@app.function(image=IMAGE, secrets=[SECRET], timeout=300)
def list_models() -> dict:
    from google import genai

    # HttpOptions.timeout is milliseconds. Bound occasional server-side stalls
    # so a failed gate cannot consume the entire Modal function lifetime.
    client = genai.Client(api_key=_api_key(), http_options=types.HttpOptions(timeout=180_000))
    names = []
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if "generateContent" in actions or "generate_content" in actions:
            names.append(model.name)
    return {"credential_variable_names": sorted(name for name in os.environ
            if name in {"GOOGLE_API_KEY", "GEMINI_API_KEY", "API_KEY"}),
            "generation_models": names}


def _source(video_id: str) -> Path:
    directory = SOURCE_ROOT / video_id
    source = next((path for path in sorted(directory.iterdir()) if path.is_file()
                   and path.suffix.lower() in VIDEO_SUFFIXES and ".partial" not in path.stem), None)
    if source is None:
        raise FileNotFoundError(video_id)
    return source


def _normalise(points, box):
    import numpy as np
    values = np.asarray(points, dtype="float64")
    if len(values) < 2:
        return []
    values -= values.min(0, keepdims=True)
    extent = np.maximum(values.max(0), 1e-6)
    scale = min((box[2]-box[0]-24)/extent[0], (box[3]-box[1]-24)/extent[1])
    values *= scale
    values += [(box[0]+box[2]-values[:, 0].max())/2,
               (box[1]+box[3]-values[:, 1].max())/2]
    return [tuple(row) for row in values]


def _panel(trace, candidates):
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (1600, 620), "white"); draw = ImageDraw.Draw(canvas)
    boxes = [(20, 50, 500, 600), (520, 50, 860, 315), (880, 50, 1220, 315),
             (1240, 50, 1580, 315), (690, 335, 1030, 600), (1050, 335, 1390, 600)]
    draw.text((20, 15), "LOW-CONFIDENCE VIDEO MOTION TRACE (blue start -> red end)", fill="black")
    line = _normalise(trace, boxes[0])
    for index in range(1, len(line)):
        ratio = index/max(1, len(line)-1)
        draw.line((line[index-1], line[index]), fill=(round(220*ratio), 40, round(220*(1-ratio))), width=6)
    for (letter, points), box in zip(candidates, boxes[1:]):
        draw.rectangle(box, outline=(160, 160, 160), width=2)
        draw.text((box[0]+8, box[1]+8), f"Candidate {letter}: ordered S -> E", fill="black")
        route = _normalise(points, (box[0]+4, box[1]+28, box[2]-4, box[3]-4))
        if route:
            draw.line(route, fill=(210, 30, 30), width=6)
            for point, color in ((route[0], "green"), (route[-1], "red")):
                draw.ellipse((point[0]-7, point[1]-7, point[0]+7, point[1]+7), fill=color)
    return canvas


def _archive_bytes(archive, reference: str) -> bytes:
    matches = [name for name in archive.namelist()
               if Path(name).name == Path(str(reference)).name]
    if len(matches) != 1:
        raise ValueError(f"ambiguous archive reference: {reference}")
    return archive.read(matches[0])


@app.function(image=IMAGE, secrets=[SECRET], volumes={"/data": VOLUME}, timeout=30*60)
def run_gate(video_id: str = "G0fWvIxy88A", perturbation: str = "base",
             pipeline: str = "joint") -> dict:
    import io
    from google import genai
    from google.genai import types
    from PIL import Image

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / "validation.json").read_text())
    motion_file = ROOT / "experiments" / "dpvo-motion" / "validation" / \
        "fps1-focal0.9-w256-b256-p16" / f"{video_id}.json"
    motions = json.loads(motion_file.read_text())
    question_id, question = next((qid, row) for qid, row in manifest[video_id].items()
                                 if "map" in classify_question_facets(row)
                                 and len(row.get("image_paths", [])) == 5)
    motion = motions[str(question_id)]
    start = float(question["start_seconds"]); end = float(question["end_seconds"])
    with zipfile.ZipFile(ROOT / "manifests" / "validation_images.zip") as archive:
        raw_candidates = [(letter, _archive_bytes(archive, reference))
                          for letter, reference in zip("ABCDE", question["image_paths"])]
    candidates = []
    for letter, raw in raw_candidates:
        markers = route_marker_positions(raw)
        candidates.append((letter, orient_path_from_start(route_path(raw), markers)))
    order = [0, 1, 2, 3, 4] if perturbation == "base" else [2, 3, 4, 0, 1]
    ordered = [candidates[index] for index in order]
    panel = _panel(motion["trajectory_2d"], ordered)
    panel_buffer = io.BytesIO(); panel.save(panel_buffer, format="PNG")

    with tempfile.TemporaryDirectory() as temporary:
        clip = Path(temporary) / "clip.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                        "-t", str(end-start), "-i", str(_source(video_id)),
                        "-an", "-vf", "fps=4,scale=640:-2", "-c:v", "libx264", "-crf", "30",
                        "-preset", "veryfast", "-movflags", "+faststart", str(clip)],
                       check=True, timeout=600)
        clip_bytes = clip.read_bytes()
    joint_parts = [types.Part.from_text(text=(
        "Solve this walking-tour route-map question from the complete ordered video clip. "
        "The first comparison panel contains a LOW-CONFIDENCE camera-motion trace and all five "
        "candidate routes normalized independently. Do not trust its absolute rotation, reflection, "
        "or scale. Infer sustained body movement rather than brief head/camera pans. Compare ordered "
        "turns, straight segments, loops, crossings, endpoint displacement, and visible street/landmark "
        "evidence. The five original candidate maps follow, each explicitly labelled with its stable "
        "answer identity. Return the identity, not its display position.\n\nQuestion: " +
        str(question.get("question", question.get("question_text", "Which map matches the route?"))))),
        types.Part.from_bytes(data=panel_buffer.getvalue(), mime_type="image/png"),
        types.Part.from_text(text="Complete exact-window walking video:"),
        types.Part.from_bytes(data=clip_bytes, mime_type="video/mp4")]
    for index in order:
        letter, raw = raw_candidates[index]
        with Image.open(io.BytesIO(raw)) as image:
            converted = io.BytesIO(); image.convert("RGB").save(converted, format="PNG")
        joint_parts += [types.Part.from_text(text=f"Original candidate identity {letter}:"),
                        types.Part.from_bytes(data=converted.getvalue(), mime_type="image/png")]
    client = genai.Client(api_key=_api_key())
    route_evidence = None
    if pipeline == "two-stage":
        evidence_response = client.models.generate_content(
            model=MODEL, contents=[types.Content(role="user", parts=[
                types.Part.from_text(text=(
                    "Watch this complete ordered walking-tour clip without seeing any answer choices. "
                    "Recover the route independently. Report ordered streets/paths and landmarks, start "
                    "and end locations, sustained body turns (ignore brief head pans), retracing or "
                    "crossings, and uncertainty. Do not guess an answer choice.")),
                types.Part.from_bytes(data=clip_bytes, mime_type="video/mp4")])],
            config=types.GenerateContentConfig(
                temperature=0, response_mime_type="application/json",
                response_schema={"type":"object", "properties":{
                    "ordered_route":{"type":"string"}, "start":{"type":"string"},
                    "end":{"type":"string"}, "turns_topology":{"type":"string"},
                    "uncertainty":{"type":"string"}},
                    "required":["ordered_route","start","end","turns_topology","uncertainty"]}))
        route_evidence = json.loads(evidence_response.text)
        request_parts = [types.Part.from_text(text=(
            "Select the candidate map matching this independently extracted walking route. Focus on exact "
            "S and E placement, ordered named streets/landmarks, intersections, turns, retracing, and route "
            "topology. Candidate identity labels are stable; return identity, not display position.\n\n"
            "Route evidence:\n" + json.dumps(route_evidence, ensure_ascii=False)))]
        for index in order:
            letter, raw = raw_candidates[index]
            request_parts += [types.Part.from_text(text=f"Candidate identity {letter}:"),
                              types.Part.from_bytes(data=raw, mime_type="image/png")]
    elif pipeline == "joint":
        request_parts = joint_parts
    else:
        raise ValueError(pipeline)
    response = client.models.generate_content(
        model=MODEL, contents=[types.Content(role="user", parts=request_parts)],
        config=types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json",
            response_schema={"type":"object", "properties":{
                "answer":{"type":"string", "enum":list("ABCDE")},
                "reason":{"type":"string"}, "confidence":{"type":"number"}},
                "required":["answer", "reason", "confidence"]}))
    parsed = json.loads(response.text)
    result = {"video_id": video_id, "question_id": str(question_id), "model": MODEL,
              "pipeline": pipeline, "route_evidence": route_evidence,
              "perturbation": perturbation, "display_order": ["ABCDE"[i] for i in order],
              "prediction": "ABCDE".index(parsed["answer"]), "response": parsed,
              "clip_bytes": len(clip_bytes), "ground_truth_used_for_inference": False,
              "answer_id": question.get("answer_id")}
    output = ROOT / "experiments" / "gemini-native-video-map" / \
        f"{video_id}-{pipeline}-{perturbation}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n"); VOLUME.commit()
    return result


@app.local_entrypoint()
def main(action: str = "list-models", video_id: str = "G0fWvIxy88A") -> None:
    if action == "list-models":
        result = list_models.remote()
    elif action in {"base", "permute", "base2", "permute2"}:
        result = run_gate.remote(video_id=video_id,
                                 perturbation="permute" if action.startswith("permute") else "base",
                                 pipeline="two-stage" if action.endswith("2") else "joint")
    else:
        raise ValueError(action)
    print(json.dumps(result, indent=2))
