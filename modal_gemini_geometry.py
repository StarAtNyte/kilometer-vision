"""Label-safe strong Gemini map-reasoning experiments."""
from __future__ import annotations

import json
import io
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import modal

from map_solver import orient_path_from_start, route_marker_positions, route_path
from question_planner import classify_question_facets


SECRET = modal.Secret.from_name("gemini-api-key")
SECRET_POOL = modal.Secret.from_name("gemini-api-keys")
ROOT = Path("/data/2707-kilometer-vision")
SOURCE_ROOT = Path("/data/shared/walking-tours")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
# The configured API project exposes 3.1 Pro but assigns it zero free-tier
# request/input-token quota. Keep the frozen gate on the strongest usable
# native-video Flash endpoint instead.
MODEL = "gemini-3.5-flash"
MAP_MODEL = "gemini-3.6-flash"
IMAGE = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "tesseract-ocr")
         .pip_install("google-genai", "pillow", "numpy==1.26.4", "opencv-python-headless",
                      "pytesseract")
         .add_local_python_source("map_solver", "question_planner"))
app = modal.App("kilometervision-gemini-geometry")


def _api_key(key_index: int | None = None) -> str:
    if key_index is not None and os.environ.get("GEMINI_API_KEYS"):
        keys = [value.strip() for value in os.environ["GEMINI_API_KEYS"].split(",")
                if value.strip()]
        if not 0 <= key_index < len(keys):
            raise IndexError(f"Gemini key index {key_index} outside pool of {len(keys)}")
        return keys[key_index]
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    raise RuntimeError("gemini-api-key secret has no supported key name")


def _parse_json_response(text: str) -> dict:
    """Parse JSON, tolerating the small malformed fragments returned by Flash."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r"\\u(?![0-9a-fA-F]{4})", r"\\\\u", text)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Flash occasionally emits a valid answer followed by one malformed
            # quoted field. Recover only the schema fields needed downstream;
            # never infer the answer from candidate geometry or labels.
            answer = re.search(r'"answer"\s*:\s*"([A-E])"', text, re.I)
            if not answer:
                raise
            confidence = re.search(r'"confidence"\s*:\s*([-+]?\d+(?:\.\d+)?)', text)
            result = {"answer": answer.group(1).upper()}
            result["confidence"] = float(confidence.group(1)) if confidence else 0.0
            for field in ("observed_start", "observed_end", "reason", "start_pin_location", "end_pin_location"):
                match = re.search(rf'"{field}"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.S)
                if match:
                    result[field] = match.group(1)
            return result


@app.function(image=IMAGE, secrets=[SECRET], timeout=300)
def list_models() -> dict:
    from google import genai
    from google.genai import types

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


@app.function(image=IMAGE, secrets=[SECRET_POOL], timeout=60)
def inspect_key_pool() -> dict:
    """Return credential variable metadata without returning secret values."""
    names = sorted(name for name in os.environ if "KEY" in name.upper()
                   or "GEMINI" in name.upper() or "GOOGLE" in name.upper())
    pooled = os.environ.get("GEMINI_API_KEYS", "")
    separators = {separator: pooled.count(separator) for separator in (",", "\n", ";")}
    return {"variables": names, "value_lengths": {name: len(os.environ[name]) for name in names},
            "pool_separator_counts": separators}


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=120)
def audit_maps(video_id: str) -> dict:
    """Return answer-blind route-overlay geometry for candidate inspection."""
    import cv2
    import numpy as np

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / "validation.json").read_text())
    question_id, question = next((qid, row) for qid, row in manifest[video_id].items()
                                 if "map" in classify_question_facets(row)
                                 and len(row.get("image_paths", [])) == 5)
    records = []
    with zipfile.ZipFile(ROOT / "manifests" / "validation_images.zip") as archive:
        for identity, reference in zip("ABCDE", question["image_paths"]):
            raw = _archive_bytes(archive, reference)
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            markers = route_marker_positions(raw)
            path = orient_path_from_start(route_path(raw), markers)
            if path:
                indices = np.linspace(0, len(path) - 1, 16).round().astype(int)
                sampled = [[round(path[index][0] / image.shape[1], 4),
                            round(path[index][1] / image.shape[0], 4)] for index in indices]
            else:
                sampled = []
            records.append({"identity": identity, "width": int(image.shape[1]),
                            "height": int(image.shape[0]),
                            "markers": {key: [round(value[0] / image.shape[1], 4),
                                              round(value[1] / image.shape[0], 4)]
                                        for key, value in markers.items()},
                            "sampled_path": sampled})
    return {"video_id": video_id, "question_id": str(question_id), "candidates": records}


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=5*60)
def audit_map_ocr(video_id: str) -> dict:
    """Pool positioned OCR from the five shared basemap renderings."""
    import cv2
    import numpy as np
    import pytesseract
    from pytesseract import Output

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / "validation.json").read_text())
    question_id, question = next((qid, row) for qid, row in manifest[video_id].items()
                                 if "map" in classify_question_facets(row)
                                 and len(row.get("image_paths", [])) == 5)
    records = []
    with zipfile.ZipFile(ROOT / "manifests" / "validation_images.zip") as archive:
        for identity, reference in zip("ABCDE", question["image_paths"]):
            raw = _archive_bytes(archive, reference)
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            enlarged = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            data = pytesseract.image_to_data(enlarged, config="--psm 11", output_type=Output.DICT)
            tokens = []
            for index, text_value in enumerate(data["text"]):
                text_value = text_value.strip()
                confidence = float(data["conf"][index])
                if not text_value or confidence < 20:
                    continue
                x = (data["left"][index] + data["width"][index] / 2) / enlarged.shape[1]
                y = (data["top"][index] + data["height"][index] / 2) / enlarged.shape[0]
                tokens.append({"text": text_value, "confidence": round(confidence, 1),
                               "x": round(x, 4), "y": round(y, 4)})
            records.append({"identity": identity, "markers": route_marker_positions(raw),
                            "tokens": tokens})
    result = {"video_id": video_id, "question_id": str(question_id), "records": records}
    output = ROOT / "experiments" / "gemini-native-video-map" / f"{video_id}-map-ocr.json"
    output.write_text(json.dumps(result, indent=2) + "\n"); VOLUME.commit()
    return {"video_id": video_id, "question_id": str(question_id),
            "token_counts": {record["identity"]: len(record["tokens"]) for record in records},
            "artifact": str(output)}


@app.function(image=IMAGE, secrets=[SECRET, SECRET_POOL], volumes={"/data": VOLUME}, timeout=10*60)
def run_map_audit(video_id: str, perturbation: str = "base",
                  key_index: int | None = None) -> dict:
    """Map-only second stage using sealed candidate-hidden route evidence."""
    from google import genai
    from google.genai import types
    from PIL import Image

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / "validation.json").read_text())
    question_id, question = next((qid, row) for qid, row in manifest[video_id].items()
                                 if "map" in classify_question_facets(row)
                                 and len(row.get("image_paths", [])) == 5)
    evidence_path = ROOT / "experiments" / "gemini-native-video-map" / \
        f"{video_id}-evidence-only.json"
    evidence = json.loads(evidence_path.read_text())["route_evidence"]
    with zipfile.ZipFile(ROOT / "manifests" / "validation_images.zip") as archive:
        candidates = [(letter, _archive_bytes(archive, reference))
                      for letter, reference in zip("ABCDE", question["image_paths"])]
    order = [0, 1, 2, 3, 4] if perturbation == "base" else [2, 3, 4, 0, 1]
    parts = [types.Part.from_text(text=(
        "Audit which north-up candidate map matches the independently extracted route below. Resolve "
        "the exact direction of travel: identify the real location under each S and E pin from visible "
        "street/POI labels, then compare ordered streets, turns, topology, and distance. Some candidates "
        "are the identical polyline with S/E reversed. The supplied normalized marker coordinates use "
        "image x right and y down. Candidate identities are stable; return identity, not display slot.\n\n"
        "Candidate-hidden route evidence:\n" + json.dumps(evidence, ensure_ascii=False)))]
    geometry = {}
    for index in order:
        letter, raw = candidates[index]
        markers = route_marker_positions(raw)
        geometry[letter] = {key: [round(value[0] / 400, 4), round(value[1] / 400, 4)]
                            for key, value in markers.items()}
        with Image.open(io.BytesIO(raw)) as candidate_image:
            enlarged = candidate_image.convert("RGB").resize(
                (candidate_image.width * 4, candidate_image.height * 4), Image.Resampling.LANCZOS)
            buffer = io.BytesIO(); enlarged.save(buffer, format="PNG")
        parts += [types.Part.from_text(text=(
                      f"Candidate identity {letter}; normalized markers {json.dumps(geometry[letter])}:")),
                  types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")]
    client = genai.Client(api_key=_api_key(key_index),
                          http_options=types.HttpOptions(timeout=180_000))
    response = client.models.generate_content(
        model=MAP_MODEL, contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            max_output_tokens=1200, response_mime_type="application/json",
            response_schema={"type":"object", "properties":{
                "answer":{"type":"string", "enum":list("ABCDE")},
                "start_pin_location":{"type":"string"},
                "end_pin_location":{"type":"string"},
                "reason":{"type":"string"}, "confidence":{"type":"number"}},
                "required":["answer","start_pin_location","end_pin_location",
                            "reason","confidence"]}))
    parsed = json.loads(response.text)
    result = {"video_id": video_id, "question_id": str(question_id), "model": MAP_MODEL,
              "pipeline": "candidate-hidden-evidence-highres-map-audit",
              "perturbation": perturbation, "display_order": ["ABCDE"[i] for i in order],
              "geometry": geometry, "prediction": "ABCDE".index(parsed["answer"]),
              "response": parsed, "ground_truth_used_for_inference": False,
              "answer_id": question.get("answer_id")}
    output = ROOT / "experiments" / "gemini-native-video-map" / \
        f"{video_id}-map-audit-{perturbation}.json"
    output.write_text(json.dumps(result, indent=2) + "\n"); VOLUME.commit()
    return result


@app.function(image=IMAGE, secrets=[SECRET, SECRET_POOL], volumes={"/data": VOLUME}, timeout=5*60)
def ground_route_direction(video_id: str, key_index: int | None = None) -> dict:
    """Fact-check endpoint identities/direction with Google Maps grounding."""
    from google import genai
    from google.genai import types

    VOLUME.reload()
    evidence_path = ROOT / "experiments" / "gemini-native-video-map" / \
        f"{video_id}-evidence-only.json"
    evidence = json.loads(evidence_path.read_text())["route_evidence"]
    client = genai.Client(api_key=_api_key(key_index),
                          http_options=types.HttpOptions(timeout=180_000))
    response = client.models.generate_content(
        model=MAP_MODEL,
        contents=(
            "Use Google Maps to fact-check the start and end places in this independently extracted "
            "walking route. Resolve hallucinated or conflicting addresses from the place/street context. "
            "Return concise JSON with keys start_place, end_place, start_address, end_address, "
            "start_to_end_compass (one of N,NE,E,SE,S,SW,W,NW), and explanation. Do not discuss answer "
            "choices. Route evidence:\n" + json.dumps(evidence, ensure_ascii=False)),
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_maps=types.GoogleMaps())],
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            max_output_tokens=1200))
    chunks = []
    grounding = response.candidates[0].grounding_metadata if response.candidates else None
    for chunk in (getattr(grounding, "grounding_chunks", None) or []):
        maps = getattr(chunk, "maps", None)
        if maps:
            chunks.append({"title": maps.title, "uri": maps.uri})
    result = {"video_id": video_id, "model": MAP_MODEL, "raw_text": response.text,
              "grounding_chunks": chunks, "ground_truth_used_for_inference": False}
    output = ROOT / "experiments" / "gemini-native-video-map" / \
        f"{video_id}-maps-grounding.json"
    output.write_text(json.dumps(result, indent=2) + "\n"); VOLUME.commit()
    return result


@app.function(image=IMAGE, secrets=[SECRET, SECRET_POOL], volumes={"/data": VOLUME}, timeout=10*60)
def run_independent_map_audit(video_id: str, key_index: int | None = None,
                              evidence_variant: str = "evidence-only") -> dict:
    """Score each candidate in an isolated request to eliminate layout competition."""
    from google import genai
    from google.genai import types
    from PIL import Image
    import time

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / "validation.json").read_text())
    question_id, question = next((qid, row) for qid, row in manifest[video_id].items()
                                 if "map" in classify_question_facets(row)
                                 and len(row.get("image_paths", [])) == 5)
    evidence = json.loads((ROOT / "experiments" / "gemini-native-video-map" /
                           f"{video_id}-{evidence_variant}.json").read_text())["route_evidence"]
    client = genai.Client(api_key=_api_key(key_index),
                          http_options=types.HttpOptions(timeout=180_000))
    output = ROOT / "experiments" / "gemini-native-video-map" / \
        f"{video_id}-independent-map-audit{'-dense' if evidence_variant == 'evidence-dense' else ''}.json"
    records = []
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get("pipeline") == "candidate-hidden-evidence-independent-highres-map-audits":
            records = previous.get("records", [])
    completed = {record["identity"] for record in records}
    with zipfile.ZipFile(ROOT / "manifests" / "validation_images.zip") as archive:
        for identity, reference in zip("ABCDE", question["image_paths"]):
            if identity in completed:
                continue
            raw = _archive_bytes(archive, reference)
            markers = route_marker_positions(raw)
            with Image.open(io.BytesIO(raw)) as candidate_image:
                enlarged = candidate_image.convert("RGB").resize(
                    (candidate_image.width * 4, candidate_image.height * 4), Image.Resampling.LANCZOS)
                buffer = io.BytesIO(); enlarged.save(buffer, format="PNG")
            prompt = (
                "Evaluate this single north-up candidate map against independently extracted walking "
                "route evidence. Identify the actual street/POI under S and E, watch for reversed "
                "endpoints, then assess ordered streets, turns, topology, and distance. Give an absolute "
                "match score from 0 to 100; do not compare against or speculate about unseen candidates.\n\n"
                "Route evidence:\n" + json.dumps(evidence, ensure_ascii=False) +
                "\n\nNormalized S/E marker coordinates (x right, y down): " +
                json.dumps(markers))
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=MAP_MODEL, contents=[prompt,
                            types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")],
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_level="low"),
                            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                            max_output_tokens=800, response_mime_type="application/json",
                            response_schema={"type":"object", "properties":{
                                "match_score":{"type":"number"},
                                "start_pin_location":{"type":"string"},
                                "end_pin_location":{"type":"string"},
                                "reason":{"type":"string"}},
                                "required":["match_score","start_pin_location",
                                            "end_pin_location","reason"]}))
                    break
                except Exception as error:
                    if getattr(error, "code", None) != 503 or attempt == 2:
                        raise
                    time.sleep(2 ** (attempt + 1))
            records.append({"identity": identity, "response": _parse_json_response(response.text)})
            partial = {"video_id": video_id, "question_id": str(question_id),
                       "model": MAP_MODEL,
                       "pipeline": "candidate-hidden-evidence-independent-highres-map-audits",
                       "records": records, "prediction": None,
                       "ground_truth_used_for_inference": False,
                       "answer_id": question.get("answer_id")}
            output.write_text(json.dumps(partial, indent=2) + "\n"); VOLUME.commit()
    prediction_identity = max(records, key=lambda row: row["response"]["match_score"])["identity"]
    result = {"video_id": video_id, "question_id": str(question_id), "model": MAP_MODEL,
              "pipeline": "candidate-hidden-evidence-independent-highres-map-audits",
              "records": records, "prediction": "ABCDE".index(prediction_identity),
              "ground_truth_used_for_inference": False, "answer_id": question.get("answer_id")}
    output.write_text(json.dumps(result, indent=2) + "\n"); VOLUME.commit()
    return result


@app.function(image=IMAGE, secrets=[SECRET, SECRET_POOL], volumes={"/data": VOLUME}, timeout=20*60)
def run_independent_video_map_audit(video_id: str, key_index: int | None = None) -> dict:
    """Judge each map directly against video in isolated, resumable requests."""
    from google import genai
    from google.genai import types
    from PIL import Image
    import time

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / "validation.json").read_text())
    question_id, question = next((qid, row) for qid, row in manifest[video_id].items()
                                 if "map" in classify_question_facets(row)
                                 and len(row.get("image_paths", [])) == 5)
    start = float(question["start_seconds"]); end = float(question["end_seconds"])
    with tempfile.TemporaryDirectory() as temporary:
        clip = Path(temporary) / "clip.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                        "-t", str(end-start), "-i", str(_source(video_id)), "-an",
                        "-vf", "fps=2,scale=640:-2", "-c:v", "libx264", "-crf", "30",
                        "-preset", "veryfast", "-movflags", "+faststart", str(clip)],
                       check=True, timeout=600)
        clip_bytes = clip.read_bytes()
    output = ROOT / "experiments" / "gemini-native-video-map" / \
        f"{video_id}-independent-video-map-audit.json"
    records = []
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get("pipeline") == "independent-native-video-single-map-audits":
            records = previous.get("records", [])
    completed = {record["identity"] for record in records}
    client = genai.Client(api_key=_api_key(key_index),
                          http_options=types.HttpOptions(timeout=180_000))
    with zipfile.ZipFile(ROOT / "manifests" / "validation_images.zip") as archive:
        for identity, reference in zip("ABCDE", question["image_paths"]):
            if identity in completed:
                continue
            raw = _archive_bytes(archive, reference)
            markers = route_marker_positions(raw)
            with Image.open(io.BytesIO(raw)) as candidate_image:
                enlarged = candidate_image.convert("RGB").resize(
                    (candidate_image.width * 4, candidate_image.height * 4), Image.Resampling.LANCZOS)
                buffer = io.BytesIO(); enlarged.save(buffer, format="PNG")
            prompt = (
                "Evaluate whether this single north-up map is the walking route in the complete ordered "
                "video. Infer sustained body movement rather than head pans. Verify visible streets/POIs, "
                "exact S and E locations, direction of travel, ordered turns, topology, and distance. Give "
                "an absolute match score from 0 to 100; do not compare against unseen candidates. "
                "Normalized markers (x right, y down): " + json.dumps(markers))
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=MAP_MODEL, contents=[prompt,
                            types.Part.from_bytes(data=clip_bytes, mime_type="video/mp4"),
                            types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")],
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_level="low"),
                            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                            max_output_tokens=1800, response_mime_type="application/json",
                            response_schema={"type":"object", "properties":{
                                "match_score":{"type":"number"},
                                "observed_route":{"type":"string"},
                                "reason":{"type":"string"}},
                                "required":["match_score","observed_route","reason"]}))
                    break
                except Exception as error:
                    if getattr(error, "code", None) != 503 or attempt == 2:
                        raise
                    time.sleep(2 ** (attempt + 1))
            records.append({"identity": identity, "response": _parse_json_response(response.text)})
            partial = {"video_id": video_id, "question_id": str(question_id),
                       "model": MAP_MODEL, "pipeline": "independent-native-video-single-map-audits",
                       "records": records, "prediction": None, "clip_bytes": len(clip_bytes),
                       "ground_truth_used_for_inference": False,
                       "answer_id": question.get("answer_id")}
            output.write_text(json.dumps(partial, indent=2) + "\n"); VOLUME.commit()
    prediction_identity = max(records, key=lambda row: row["response"]["match_score"])["identity"]
    result = {"video_id": video_id, "question_id": str(question_id), "model": MAP_MODEL,
              "pipeline": "independent-native-video-single-map-audits", "records": records,
              "prediction": "ABCDE".index(prediction_identity), "clip_bytes": len(clip_bytes),
              "ground_truth_used_for_inference": False, "answer_id": question.get("answer_id")}
    output.write_text(json.dumps(result, indent=2) + "\n"); VOLUME.commit()
    return result


@app.function(image=IMAGE, secrets=[SECRET, SECRET_POOL], volumes={"/data": VOLUME}, timeout=10*60)
def run_video_map_duel(video_id: str, identities: str, perturbation: str = "base",
                       key_index: int | None = None, model_name: str = MAP_MODEL,
                       split: str = "validation", target_question: str = "__all__") -> dict:
    """Resolve a close independent-score shortlist with endpoint-focused video evidence."""
    import time
    from google import genai
    from google.genai import types
    from PIL import Image

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    question_id, question = next((qid, row) for qid, row in manifest[video_id].items()
                                 if "map" in classify_question_facets(row)
                                 and len(row.get("image_paths", [])) == 5
                                 and (target_question == "__all__" or str(qid) == str(target_question)))
    chosen = list(identities)
    if perturbation == "reverse":
        chosen.reverse()
    with tempfile.TemporaryDirectory() as temporary:
        clip = Path(temporary) / "clip.mp4"
        start = float(question["start_seconds"]); end = float(question["end_seconds"])
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                        "-t", str(end-start), "-i", str(_source(video_id)), "-an",
                        "-vf", "fps=2,scale=640:-2", "-c:v", "libx264", "-crf", "30",
                        "-preset", "veryfast", "-movflags", "+faststart", str(clip)],
                       check=True, timeout=600)
        clip_bytes = clip.read_bytes()
    with zipfile.ZipFile(ROOT / "manifests" / f"{split}_images.zip") as archive:
        by_identity = {letter: _archive_bytes(archive, reference)
                       for letter, reference in zip("ABCDE", question["image_paths"])}
    parts = [types.Part.from_text(text=(
        "Resolve this close shortlist using the complete ordered walking video. Concentrate on the first "
        "and last locations, exact S/E placement and travel direction, then ordered turns and route "
        "extent. Brief head pans are not body turns. Candidate identity labels are stable; return identity, "
        "not display slot.")), types.Part.from_bytes(data=clip_bytes, mime_type="video/mp4")]
    for identity in chosen:
        raw = by_identity[identity]
        with Image.open(io.BytesIO(raw)) as candidate_image:
            enlarged = candidate_image.convert("RGB").resize(
                (candidate_image.width * 4, candidate_image.height * 4), Image.Resampling.LANCZOS)
            buffer = io.BytesIO(); enlarged.save(buffer, format="PNG")
        parts += [types.Part.from_text(text=f"Candidate identity {identity}:"),
                  types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")]
    client = genai.Client(api_key=_api_key(key_index),
                          http_options=types.HttpOptions(timeout=180_000))
    generation_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="low"),
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        max_output_tokens=1000, response_mime_type="application/json",
        response_schema={"type":"object", "properties":{
            "answer":{"type":"string", "enum":list(identities)},
            "observed_start":{"type":"string"}, "observed_end":{"type":"string"},
            "reason":{"type":"string"}, "confidence":{"type":"number"}},
            "required":["answer","observed_start","observed_end","reason","confidence"]})
    response = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name, contents=[types.Content(role="user", parts=parts)],
                config=generation_config)
            break
        except Exception as error:
            if getattr(error, "code", None) != 503 or attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
    parsed = _parse_json_response(response.text)
    result = {"video_id": video_id, "question_id": str(question_id), "model": model_name,
              "pipeline": "endpoint-focused-video-map-duel", "identities": identities,
              "perturbation": perturbation, "display_order": chosen,
              "prediction": "ABCDE".index(parsed["answer"]), "response": parsed,
              "ground_truth_used_for_inference": False, "answer_id": question.get("answer_id")}
    model_tag = re.sub(r"[^A-Za-z0-9]+", "", model_name)
    output_name = (f"{video_id}-duel-{identities}-{perturbation}-{model_tag}.json"
                   if split == "validation" else
                   f"{split}-{video_id}-duel-{identities}-{perturbation}-{model_tag}.json")
    output = ROOT / "experiments" / "gemini-native-video-map" / output_name
    output.write_text(json.dumps(result, indent=2) + "\n"); VOLUME.commit()
    return result


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


@app.function(image=IMAGE, secrets=[SECRET, SECRET_POOL], volumes={"/data": VOLUME}, timeout=30*60)
def run_gate(video_id: str = "G0fWvIxy88A", perturbation: str = "base",
             pipeline: str = "joint", key_index: int | None = None) -> dict:
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
        # Candidate-hidden semantic extraction does not need dense motion. A
        # 1 FPS clip keeps all street/landmark context while avoiding the API
        # latency path observed with the 4 FPS evidence-only request.
        clip_fps = 1 if pipeline in {"two-stage", "evidence-only"} else 4
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                        "-t", str(end-start), "-i", str(_source(video_id)),
                        "-an", "-vf", f"fps={clip_fps},scale=640:-2", "-c:v", "libx264", "-crf", "30",
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
    client = genai.Client(api_key=_api_key(key_index),
                          http_options=types.HttpOptions(timeout=180_000))
    route_evidence = None
    if pipeline in {"two-stage", "evidence-only", "evidence-dense"}:
        evidence_response = client.models.generate_content(
            model=MODEL, contents=[types.Content(role="user", parts=[
                types.Part.from_text(text=(
                    "Watch this complete ordered walking-tour clip without seeing any answer choices. "
                    "Recover the route independently. Report ordered streets/paths and landmarks, start "
                    "and end locations, sustained body turns (ignore brief head pans), retracing or "
                    "crossings, and uncertainty. Also give concise, independently geocodable start and "
                    "end queries including city/country when identifiable. Do not guess an answer choice.")),
                types.Part.from_bytes(data=clip_bytes, mime_type="video/mp4")])],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                max_output_tokens=1200, response_mime_type="application/json",
                response_schema={"type":"object", "properties":{
                    "ordered_route":{"type":"string"}, "start":{"type":"string"},
                    "end":{"type":"string"}, "turns_topology":{"type":"string"},
                    "uncertainty":{"type":"string"},
                    "start_geocode_query":{"type":"string"},
                    "end_geocode_query":{"type":"string"}},
                    "required":["ordered_route","start","end","turns_topology","uncertainty",
                                "start_geocode_query","end_geocode_query"]}))
        route_evidence = json.loads(evidence_response.text)
        if pipeline in {"evidence-only", "evidence-dense"}:
            result = {"video_id": video_id, "question_id": str(question_id), "model": MODEL,
                      "pipeline": pipeline, "route_evidence": route_evidence,
                      "clip_bytes": len(clip_bytes), "ground_truth_used_for_inference": False,
                      "answer_id": question.get("answer_id")}
            output = ROOT / "experiments" / "gemini-native-video-map" / \
                f"{video_id}-{pipeline}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n"); VOLUME.commit()
            return result
        request_parts = [types.Part.from_text(text=(
            "Select the candidate map matching this independently extracted walking route. Focus on exact "
            "S and E placement, ordered named streets/landmarks, intersections, turns, retracing, and route "
            "topology. Candidate identity labels are stable; return identity, not display position.\n\n"
            "Route evidence:\n" + json.dumps(route_evidence, ensure_ascii=False)))]
        for index in order:
            letter, raw = raw_candidates[index]
            request_parts += [types.Part.from_text(text=f"Candidate identity {letter}:"),
                              types.Part.from_bytes(data=raw, mime_type="image/png")]
    elif pipeline == "joint-debate":
        preliminary = client.models.generate_content(
            model=MODEL, contents=[types.Content(role="user", parts=joint_parts)],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                max_output_tokens=1200, response_mime_type="application/json",
                response_schema={"type":"object", "properties":{
                    "answer":{"type":"string", "enum":list("ABCDE")},
                    "ordered_route":{"type":"string"}, "start":{"type":"string"},
                    "end":{"type":"string"}, "turns_topology":{"type":"string"},
                    "uncertainty":{"type":"string"}},
                    "required":["answer","ordered_route","start","end",
                                "turns_topology","uncertainty"]}))
        preliminary_parsed = json.loads(preliminary.text)
        provisional_answer = preliminary_parsed.pop("answer")
        route_evidence = preliminary_parsed
        request_parts = [types.Part.from_text(text=(
            "Independently audit which candidate map matches these route facts. The provisional answer "
            "has been withheld. Check exact S and E placement, ordered named streets/landmarks, turns, "
            "retracing, and topology. Candidate identity labels are stable; return identity, not display "
            "position.\n\nRoute facts:\n" + json.dumps(route_evidence, ensure_ascii=False)))]
        for index in order:
            letter, raw = raw_candidates[index]
            request_parts += [types.Part.from_text(text=f"Candidate identity {letter}:"),
                              types.Part.from_bytes(data=raw, mime_type="image/png")]
    elif pipeline == "joint":
        provisional_answer = None
        request_parts = joint_parts
    else:
        raise ValueError(pipeline)
    response = client.models.generate_content(
        model=MODEL, contents=[types.Content(role="user", parts=request_parts)],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            max_output_tokens=800, response_mime_type="application/json",
            response_schema={"type":"object", "properties":{
                "answer":{"type":"string", "enum":list("ABCDE")},
                "reason":{"type":"string"}, "confidence":{"type":"number"}},
                "required":["answer", "reason", "confidence"]}))
    parsed = json.loads(response.text)
    result = {"video_id": video_id, "question_id": str(question_id), "model": MODEL,
              "pipeline": pipeline, "route_evidence": route_evidence,
              "provisional_answer": provisional_answer if pipeline == "joint-debate" else None,
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
def main(action: str = "list-models", video_id: str = "G0fWvIxy88A",
         key_index: int = -1, identities: str = "BC",
         model: str = MAP_MODEL, split: str = "validation",
         target_question: str = "__all__") -> None:
    if action == "list-models":
        result = list_models.remote()
    elif action == "inspect-pool":
        result = inspect_key_pool.remote()
    elif action == "audit-maps":
        result = audit_maps.remote(video_id)
    elif action == "ocr-maps":
        result = audit_map_ocr.remote(video_id)
    elif action in {"map-base", "map-permute"}:
        result = run_map_audit.remote(video_id,
                                      "permute" if action.endswith("permute") else "base",
                                      None if key_index < 0 else key_index)
    elif action == "ground-route":
        result = ground_route_direction.remote(video_id,
                                               None if key_index < 0 else key_index)
    elif action == "independent-maps":
        result = run_independent_map_audit.remote(video_id,
                                                 None if key_index < 0 else key_index)
    elif action == "independent-maps-dense":
        result = run_independent_map_audit.remote(video_id,
                                                 None if key_index < 0 else key_index,
                                                 "evidence-dense")
    elif action == "independent-video-maps":
        result = run_independent_video_map_audit.remote(
            video_id, None if key_index < 0 else key_index)
    elif action in {"duel", "duel-reverse"}:
        result = run_video_map_duel.remote(
            video_id, identities, "reverse" if action.endswith("reverse") else "base",
            None if key_index < 0 else key_index, model, split, target_question)
    elif action == "evidence":
        result = run_gate.remote(video_id=video_id, pipeline="evidence-only",
                                 key_index=None if key_index < 0 else key_index)
    elif action == "evidence-dense":
        result = run_gate.remote(video_id=video_id, pipeline="evidence-dense",
                                 key_index=None if key_index < 0 else key_index)
    elif action in {"base", "permute", "base2", "permute2", "base3", "permute3"}:
        result = run_gate.remote(video_id=video_id,
                                 perturbation="permute" if action.startswith("permute") else "base",
                                 pipeline=("two-stage" if action.endswith("2") else
                                           "joint-debate" if action.endswith("3") else "joint"),
                                 key_index=None if key_index < 0 else key_index)
    else:
        raise ValueError(action)
    print(json.dumps(result, indent=2))
