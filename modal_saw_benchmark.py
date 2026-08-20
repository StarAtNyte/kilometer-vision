"""Held-out SAW-Bench pilot for KilometerVision route-state diagnostics.

Records are selected deterministically by row position, never by label. Official
questions, options, videos, and answers remain unchanged. Answers are consulted
only after generation to compute aggregate accuracy.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import modal


ROOT = Path("/data/2707-kilometer-vision")
EXTERNAL_ROOT = ROOT / "external" / "saw-bench"
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
CHECKPOINT = "Qwen/Qwen3.8-27B"
TASKS = ("direction", "shape", "revplan")

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1", "accelerate", "pillow",
        "qwen-vl-utils==0.0.14", "huggingface_hub", "pyarrow",
    )
    .pip_install("git+https://github.com/huggingface/transformers.git")
)
app = modal.App("kilometervision-saw-qwen38")


def _letter(text: str, option_count: int) -> int | None:
    patterns = (r"^\s*([A-Z])(?:\s|\.|\)|$)", r"answer\s*(?:is|:)?\s*([A-Z])")
    for pattern in patterns:
        match = re.search(pattern, text.upper())
        if match:
            value = ord(match.group(1)) - ord("A")
            return value if 0 <= value < option_count else None
    return None


def _indices(length: int, count: int) -> list[int]:
    """Even deterministic coverage, independent of all record contents."""
    if count >= length:
        return list(range(length))
    return sorted({round((index + 0.5) * length / count - 0.5) for index in range(count)})


def _frames(video: str, fps: float):
    """Bounded-resolution chronological decode; avoids torchvision full-video RAM use."""
    from PIL import Image
    temporary = tempfile.TemporaryDirectory()
    pattern = Path(temporary.name) / "%06d.jpg"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video,
         "-vf", f"fps={fps},scale=448:-2", "-q:v", "3", str(pattern)],
        check=True, timeout=300,
    )
    images = [Image.open(path).convert("RGB").copy() for path in sorted(Path(temporary.name).glob("*.jpg"))]
    return temporary, images


@app.function(image=IMAGE, gpu="H200", memory=65536, volumes={"/data": VOLUME}, timeout=6 * 3600)
def run(samples_per_task: int = 6, fps: float = 2.0) -> dict:
    os.environ["HF_HOME"] = str(ROOT / "checkpoints" / "huggingface")
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    VOLUME.reload()
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    model = AutoModelForImageTextToText.from_pretrained(
        CHECKPOINT, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
        cache_dir=os.environ["HF_HOME"], attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(CHECKPOINT, cache_dir=os.environ["HF_HOME"])
    records = []
    for task in TASKS:
        parquet = hf_hub_download(
            "ucsbai/SAW-Bench", f"data/{task}.parquet", repo_type="dataset",
            local_dir=EXTERNAL_ROOT,
        )
        rows = pq.read_table(parquet).to_pylist()
        for row_index in _indices(len(rows), samples_per_task):
            row = rows[row_index]
            key = str(row["key"])
            scene = key.split("_", 1)[0]
            video = hf_hub_download(
                "ucsbai/SAW-Bench", f"videos_compressed/Scene_{scene}/{key}.mp4",
                repo_type="dataset", local_dir=EXTERNAL_ROOT,
            )
            options = [str(value) for value in row["options"]]
            rendered_options = "\n".join(
                f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
            )
            prompt = (
                "Answer this observer-centric spatial question from the complete chronological egocentric video. "
                "Track physical translation separately from head/camera pans and accumulate turns in order. "
                "For a reverse route, invert both action order and turn direction. Select exactly one option and "
                "reply with only its letter.\n\n"
                f"Question: {row['question']}\n{rendered_options}"
            )
            temporary, frames = _frames(video, fps)
            content = []
            for frame_index, frame in enumerate(frames):
                content.extend([
                    {"type": "text", "text": f"Chronological frame {frame_index + 1}/{len(frames)}:"},
                    {"type": "image", "image": frame},
                ])
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False, preserve_thinking=False,
            )
            images, videos = process_vision_info(messages)
            inputs = processor(
                text=[rendered], images=images, videos=videos, padding=True, return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            raw_text = processor.batch_decode(
                generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            )[0].strip()
            prediction = _letter(raw_text, len(options))
            # Label access begins here, strictly after prediction.
            answer = int(row["answer"])
            records.append({
                "task": task, "row_index": row_index, "key": key,
                "question": row["question"], "options": options,
                "prediction": prediction, "raw_text": raw_text, "answer": answer,
                "correct": prediction == answer, "fps": fps, "checkpoint": CHECKPOINT,
                "ground_truth_used_for_inference": False,
            })
            del inputs, generated
            temporary.cleanup()
            torch.cuda.empty_cache()
    by_task = {}
    for task in TASKS:
        subset = [row for row in records if row["task"] == task]
        by_task[task] = {
            "correct": sum(row["correct"] for row in subset), "total": len(subset),
            "accuracy": sum(row["correct"] for row in subset) / len(subset) if subset else None,
            "unparsed": sum(row["prediction"] is None for row in subset),
        }
    result = {
        "benchmark": "ucsbai/SAW-Bench", "checkpoint": CHECKPOINT,
        "samples_per_task": samples_per_task, "fps": fps,
        "selection": "even deterministic row positions; label-blind",
        "ground_truth_used_for_inference": False, "by_task": by_task, "records": records,
    }
    output = ROOT / "experiments" / "external-saw" / f"qwen38-n{samples_per_task}-fps{fps:g}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    VOLUME.commit()
    return result


@app.local_entrypoint()
def main(samples_per_task: int = 6, fps: float = 2.0):
    result = run.remote(samples_per_task, fps)
    print(json.dumps({"by_task": result["by_task"], "records": len(result["records"])}, indent=2))
