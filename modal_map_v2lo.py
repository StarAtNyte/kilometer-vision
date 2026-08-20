"""Image-only V2LO map-choice control on KilometerVision validation maps.

This deliberately tests candidate-map reading separately from video evidence.
Candidate identities are retained in the prompt text but their display order is
rotated, and validation answers are copied only after inference for reporting.
"""
from __future__ import annotations

import io
import json
import random
import zipfile
from pathlib import Path

import modal

ROOT = Path("/data/2707-kilometer-vision")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
CHECKPOINT = "ybrrraway/V2LO-7B"
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "torchvision", "transformers>=4.49", "accelerate", "pillow", "qwen-vl-utils[decord]")
)
app = modal.App("kilometervision-map-v2lo")


@app.function(image=IMAGE, gpu="H200", volumes={"/data": VOLUME}, timeout=2 * 3600)
def run(split: str = "validation", checkpoint: str = CHECKPOINT,
        output_namespace: str = "map-v2lo"):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    VOLUME.reload()
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(checkpoint)
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    archive_path = ROOT / "manifests" / f"{split}_images.zip"
    records = []
    rng = random.Random(2707)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        for video_id, questions in manifest.items():
            for question_id, question in questions.items():
                refs = question.get("image_paths")
                if not isinstance(refs, list) or len(refs) != 5 or "maps" not in question.get("question", "").lower():
                    continue
                images = []
                for ref in refs:
                    matches = [name for name in members if Path(name).name == Path(str(ref)).name]
                    images.append(Image.open(io.BytesIO(archive.read(matches[0]))).convert("RGB"))
                order = list(range(5)); rng.shuffle(order)
                content = [{"type": "text", "text": (
                    "Five candidate top-down route maps are shown below in a randomized display order. "
                    "Each red route has an S (start) and E (end) marker. Identify which candidate best "
                    "represents the route; inspect the route geometry and ignore any visual ordering bias. "
                    "Reply with exactly one letter A, B, C, D, or E referring to the candidate identity, "
                    "not its display position.\nCandidate identities: A, B, C, D, E."
                )}]
                for display, candidate_index in enumerate(order):
                    content += [{"type": "text", "text": f"Candidate {chr(65 + candidate_index)}:"},
                                {"type": "image", "image": images[candidate_index]}]
                messages = [{"role": "user", "content": content}]
                rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[rendered], images=image_inputs, videos=video_inputs,
                                   padding=True, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
                import re
                tagged = re.search(r"<answer>\s*([ABCDE])\s*</answer>", raw, flags=re.IGNORECASE)
                letter = tagged.group(1).upper() if tagged else next(
                    (c for c in reversed(raw.upper()) if c in "ABCDE"), None
                )
                prediction = "ABCDE".index(letter) if letter else None
                records.append({"video_id": video_id, "question_id": str(question_id),
                                "prediction": prediction, "raw_text": raw, "display_order": order,
                                "checkpoint": checkpoint, "ground_truth_used_for_inference": False,
                                "answer_id": question.get("answer_id")})
                del inputs, generated, images
                torch.cuda.empty_cache()
    result = {"split": split, "checkpoint": checkpoint, "records": records,
              "ground_truth_used_for_inference": False}
    output = ROOT / "experiments" / output_namespace / f"{split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@app.local_entrypoint()
def main(split: str = "validation", checkpoint: str = CHECKPOINT,
         output_namespace: str = "map-v2lo"):
    result = run.remote(split, checkpoint, output_namespace)
    print(json.dumps({"records": len(result["records"]),
                      "predictions": [r["prediction"] for r in result["records"]]}, indent=2))
