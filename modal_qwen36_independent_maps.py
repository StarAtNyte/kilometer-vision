"""Order-free Qwen3.6 verifier: score each candidate map in isolation.

The model never sees answer labels, other candidates, or validation labels. Each
candidate receives identical ordered video evidence and its original map pixels.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import modal

from question_planner import classify_question_facets


ROOT = Path("/data/2707-kilometer-vision")
SOURCE_ROOT = Path("/data/shared/walking-tours")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
CHECKPOINT = "Qwen/Qwen3.8-27B"
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1",
        "accelerate", "qwen-vl-utils==0.0.14", "pillow",
    )
    .pip_install("git+https://github.com/huggingface/transformers.git")
    .add_local_python_source("question_planner")
)
app = modal.App("kilometervision-qwen36-independent-maps")


def _source(video_id: str) -> Path:
    directory = SOURCE_ROOT / video_id
    source = next((p for p in sorted(directory.iterdir()) if p.is_file()
                   and p.suffix.lower() in VIDEO_SUFFIXES and ".partial" not in p.stem), None)
    if source is None:
        raise FileNotFoundError(video_id)
    return source


def _frame(source: Path, timestamp: float):
    from PIL import Image
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "frame.jpg"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(timestamp),
             "-i", str(source), "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "3", str(target)],
            check=True, timeout=90,
        )
        return Image.open(target).convert("RGB").copy()


def _candidate_bytes(question: dict, archive: zipfile.ZipFile) -> list[bytes]:
    members = archive.namelist()
    result = []
    for reference in question.get("image_paths", []):
        matches = [name for name in members if Path(name).name == Path(str(reference)).name]
        if len(matches) != 1:
            raise ValueError(f"candidate {reference!r}: expected one archive member, got {len(matches)}")
        result.append(archive.read(matches[0]))
    return result


def _score(text: str) -> int | None:
    match = re.search(r'"score"\s*:\s*(100|[0-9]{1,2})', text)
    return int(match.group(1)) if match else None


@app.function(image=IMAGE, gpu="H200", memory=65536, volumes={"/data": VOLUME}, timeout=6 * 3600)
def run(split: str = "validation", video_id: str = "__all__", frame_count: int = 16,
        extract_map_ocr: bool = False):
    os.environ["HF_HOME"] = str(ROOT / "checkpoints" / "huggingface")
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    VOLUME.reload()
    model = AutoModelForImageTextToText.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
        cache_dir=os.environ["HF_HOME"], attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(CHECKPOINT, cache_dir=os.environ["HF_HOME"])
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    anchor_path = ROOT / "experiments" / f"video-route-anchors-{split}.json"
    anchors = json.loads(anchor_path.read_text()).get("records", {}) if anchor_path.exists() else {}
    records = []
    wanted_ids = set(video_id.split(",")) if video_id != "__all__" else None
    with zipfile.ZipFile(ROOT / "manifests" / f"{split}_images.zip") as archive:
        for target, questions in manifest.items():
            if wanted_ids is not None and target not in wanted_ids:
                continue
            for question_id, question in questions.items():
                if "map" not in classify_question_facets(question) or len(question.get("image_paths", [])) != 5:
                    continue
                start = float(question.get("start_seconds", 0))
                end = float(question.get("end_seconds", start))
                timestamps = [start + (end - start) * i / max(frame_count - 1, 1)
                              for i in range(frame_count)]
                frames = [_frame(_source(target), timestamp) for timestamp in timestamps]
                candidates = _candidate_bytes(question, archive)
                public_title = str(anchors.get(target, {}).get("title", ""))
                evidence_prompt = (
                    "Extract a candidate-blind route record from these chronologically ordered walking frames. "
                    f"Public source-video title (use only as city-level grounding): {public_title or 'unavailable'}. "
                    "Report only directly supported evidence: city/area, readable street or landmark names, start "
                    "and end surroundings, overall compass displacement if inferable, elevation change, and ordered "
                    "junction/turn sequence. Explicitly mark uncertain fields. Return compact JSON only."
                )
                evidence_content = []
                for order, frame in enumerate(frames):
                    evidence_content.extend([
                        {"type": "text", "text": f"Ordered frame {order + 1}/{len(frames)}:"},
                        {"type": "image", "image": frame},
                    ])
                evidence_content.append({"type": "text", "text": evidence_prompt})
                evidence_messages = [{"role": "user", "content": evidence_content}]
                evidence_rendered = processor.apply_chat_template(
                    evidence_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False, preserve_thinking=False,
                )
                evidence_images, evidence_videos = process_vision_info(evidence_messages)
                evidence_inputs = processor(
                    text=[evidence_rendered], images=evidence_images, videos=evidence_videos,
                    padding=True, return_tensors="pt",
                ).to(model.device)
                with torch.inference_mode():
                    evidence_generated = model.generate(
                        **evidence_inputs, max_new_tokens=320, do_sample=False,
                    )
                route_evidence = processor.batch_decode(
                    evidence_generated[:, evidence_inputs.input_ids.shape[1]:], skip_special_tokens=True,
                )[0].strip()
                del evidence_inputs, evidence_generated
                torch.cuda.empty_cache()
                audits = []
                for candidate_index, raw in enumerate(candidates):
                    candidate = Image.open(io.BytesIO(raw)).convert("RGB")
                    candidate.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                    map_ocr = None
                    if extract_map_ocr:
                        ocr_prompt = (
                            "Read this north-up route map without guessing the represented video. Return compact JSON "
                            "with S and E marker centers and every confidently readable street, place, business, or "
                            "landmark label. Coordinates must be normalized x,y in [0,1] from the image top-left. "
                            "Schema: {\"S\":[x,y],\"E\":[x,y],\"labels\":[{\"text\":str,\"xy\":[x,y]}]}. "
                            "Omit uncertain labels; JSON only."
                        )
                        ocr_messages = [{"role": "user", "content": [
                            {"type": "image", "image": candidate},
                            {"type": "text", "text": ocr_prompt},
                        ]}]
                        ocr_rendered = processor.apply_chat_template(
                            ocr_messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False, preserve_thinking=False,
                        )
                        ocr_images, ocr_videos = process_vision_info(ocr_messages)
                        ocr_inputs = processor(text=[ocr_rendered], images=ocr_images, videos=ocr_videos,
                                               padding=True, return_tensors="pt").to(model.device)
                        with torch.inference_mode():
                            ocr_generated = model.generate(**ocr_inputs, max_new_tokens=512, do_sample=False)
                        map_ocr = processor.batch_decode(
                            ocr_generated[:, ocr_inputs.input_ids.shape[1]:], skip_special_tokens=True,
                        )[0].strip()
                        del ocr_inputs, ocr_generated
                        torch.cuda.empty_cache()
                    prompt = (
                        "Audit one candidate map against an ordered walking-video clip. The map is north-up and its "
                        "highlighted route runs from green S (start) to red E (end). Infer visible street/place names, "
                        "junction sequence, ordered turns, route direction, and endpoint displacement from the frames. "
                        "Ignore minor camera turns and map styling. Score 0-100 how likely this exact S-to-E route is the "
                        "walk. Be strict: readable geography or a direction/topology contradiction should dominate. "
                        "The candidate-blind video evidence is quoted below. Do not reinterpret or embellish it. "
                        "Does this exact S-to-E map route agree with that evidence? Answer exactly YES or NO.\n\n"
                        f"VIDEO ROUTE EVIDENCE:\n{route_evidence}"
                    )
                    content = [{"type": "text", "text": "Candidate map:"},
                               {"type": "image", "image": candidate}]
                    for order, frame in enumerate(frames):
                        content.extend([
                            {"type": "text", "text": f"Ordered frame {order + 1}/{len(frames)}:"},
                            {"type": "image", "image": frame},
                        ])
                    content.append({"type": "text", "text": prompt})
                    messages = [{"role": "user", "content": content}]
                    rendered = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False, preserve_thinking=False,
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(text=[rendered], images=image_inputs, videos=video_inputs,
                                       padding=True, return_tensors="pt").to(model.device)
                    yes_ids = processor.tokenizer.encode("YES", add_special_tokens=False)
                    no_ids = processor.tokenizer.encode("NO", add_special_tokens=False)
                    with torch.inference_mode():
                        logits = model(**inputs).logits[0, -1].float()
                    margin = float(logits[yes_ids[0]] - logits[no_ids[0]])
                    audits.append({"candidate_index": candidate_index, "yes_no_logit_margin": margin,
                                   "map_ocr": map_ocr,
                                   "yes_token_ids": yes_ids, "no_token_ids": no_ids})
                    del inputs, logits
                    torch.cuda.empty_cache()
                valid = [(row["yes_no_logit_margin"], row["candidate_index"]) for row in audits]
                prediction = max(valid)[1] if valid else None
                records.append({
                    "video_id": target, "question_id": str(question_id), "prediction": prediction,
                    "audits": audits, "timestamps": timestamps, "checkpoint": CHECKPOINT,
                    "route_evidence": route_evidence,
                    "frame_count": frame_count, "ground_truth_used_for_inference": False,
                    "public_video_title": public_title,
                    "answer_id": question.get("answer_id"),
                })
                del frames
                torch.cuda.empty_cache()
    result = {"split": split, "records": records, "ground_truth_used_for_inference": False}
    output = ROOT / "experiments" / "qwen38-independent-maps" / split / f"{video_id}-f{frame_count}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@app.local_entrypoint()
def main(split: str = "validation", video_id: str = "__all__", frame_count: int = 16,
         extract_map_ocr: bool = False):
    result = run.remote(split, video_id, frame_count, extract_map_ocr)
    print(json.dumps({"records": len(result["records"]), "results": [
        {"video_id": row["video_id"], "prediction": row["prediction"],
         "answer_id": row.get("answer_id"),
         "yes_no_margins": [round(audit["yes_no_logit_margin"], 4) for audit in row["audits"]]}
        for row in result["records"]
    ]}, indent=2))
