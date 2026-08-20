"""External-only calibration for DINOv2 landmark presence scores on MMS-VPR."""
from __future__ import annotations

import json
import re
from pathlib import Path

import modal


ROOT = Path("/data/2707-kilometer-vision")
EXTERNAL = Path("/data/allowed-training/mms-vpr-sample")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "transformers>=4.49", "pillow", "huggingface_hub")
)
app = modal.App("kilometervision-landmark-external-calibration")


def _place(path: Path) -> str | None:
    name = path.name
    match = re.search(r"(?<![A-Za-z0-9])(Eh-\d+-\d+|Ev-\d+-\d+|N-\d+-\d+|S-\d+)(?!\d)", name)
    return match.group(1) if match else None


@app.function(image=IMAGE, gpu="A10G", volumes={"/data": VOLUME}, timeout=60 * 60)
def calibrate() -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image as PILImage
    from transformers import AutoImageProcessor, AutoModel

    VOLUME.reload()
    snapshot_download(
        "Yiwei-Ou/MMS-VPR",
        repo_type="dataset",
        local_dir=EXTERNAL,
        allow_patterns=["Sample dataset/01 Images/*.jpg"],
    )
    paths = sorted((EXTERNAL / "Sample dataset" / "01 Images").glob("*.jpg"))
    items = [(path, _place(path)) for path in paths]
    items = [(path, place) for path, place in items if place]
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base", torch_dtype=torch.bfloat16).to("cuda").eval()

    features = []
    for offset in range(0, len(items), 32):
        images = [PILImage.open(path).convert("RGB") for path, _ in items[offset : offset + 32]]
        inputs = processor(images=images, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            patches = model(**inputs).last_hidden_state[:, 1:].float()
        features.extend((patches / patches.norm(dim=-1, keepdim=True).clamp_min(1e-9)).cpu())

    rows = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            similarity = features[i] @ features[j].T
            a = similarity.max(dim=1).values
            b = similarity.max(dim=0).values
            score = float((a.topk(max(1, a.numel() // 10)).values.mean() + b.topk(max(1, b.numel() // 10)).values.mean()) / 2)
            a_to_b = similarity.argmax(dim=1)
            b_to_a = similarity.argmax(dim=0)
            mutual = sum(int(b_to_a[k]) == n for n, k in enumerate(a_to_b.tolist())) / max(1, min(similarity.shape))
            rows.append({"a": i, "b": j, "place_a": items[i][1], "place_b": items[j][1], "same": items[i][1] == items[j][1], "patch_local_similarity": score, "mutual_patch_ratio": mutual})

    # Tune on all places except one, evaluate on pairs touching the held-out place.
    folds = []
    for held_out in sorted({place for _, place in items}):
        train = [row for row in rows if held_out not in {row["place_a"], row["place_b"]}]
        test = [row for row in rows if held_out in {row["place_a"], row["place_b"]}]
        candidates = sorted({row["patch_local_similarity"] for row in train})
        def balanced_accuracy(data, threshold):
            positives = [row for row in data if row["same"]]
            negatives = [row for row in data if not row["same"]]
            tpr = sum(row["patch_local_similarity"] >= threshold for row in positives) / max(1, len(positives))
            tnr = sum(row["patch_local_similarity"] < threshold for row in negatives) / max(1, len(negatives))
            return (tpr + tnr) / 2, tpr, tnr
        threshold = max(candidates, key=lambda value: balanced_accuracy(train, value)[0])
        score, tpr, tnr = balanced_accuracy(test, threshold)
        folds.append({"held_out_place": held_out, "threshold": threshold, "balanced_accuracy": score, "true_positive_rate": tpr, "true_negative_rate": tnr, "pairs": len(test)})

    thresholds = sorted(fold["threshold"] for fold in folds)
    result = {
        "source": "Yiwei-Ou/MMS-VPR Sample dataset; external-only CC-BY-NC-SA-4.0 calibration",
        "checkpoint": "facebook/dinov2-base",
        "images": len(items),
        "places": dict(sorted(__import__("collections").Counter(place for _, place in items).items())),
        "positive_pairs": sum(row["same"] for row in rows),
        "negative_pairs": sum(not row["same"] for row in rows),
        "leave_one_place_out": folds,
        "median_threshold": thresholds[len(thresholds) // 2],
        "mean_balanced_accuracy": sum(fold["balanced_accuracy"] for fold in folds) / len(folds),
        "status": "preliminary_only_four_places",
    }
    output = ROOT / "experiments" / "landmark-dinov2" / "external-mms-vpr-sample-calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(calibrate.remote(), indent=2))
