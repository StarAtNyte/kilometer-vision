"""Train/dev experiment for an external-only visual landmark verifier.

MMS-VPR location classes are split as whole groups, so no place appears in both
training and development. KilometerVision media and labels are never read.
"""
from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

import modal


ROOT = Path("/data/2707-kilometer-vision")
DATA = Path("/data/allowed-training/mms-vpr")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
CPU_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub")
GPU_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "torchvision", "transformers>=4.49", "pillow", "accelerate"
)
app = modal.App("kilometervision-landmark-verifier-train")


def _label(name: str) -> str | None:
    for part in PurePosixPath(name).parts:
        if re.fullmatch(r"(?:Eh|Ev|N)-\d+-\d+|S-\d+", part):
            return part
    match = re.search(r"(?<![A-Za-z0-9])((?:Eh|Ev|N)-\d+-\d+|S-\d+)(?!\d)", name)
    return match.group(1) if match else None


def _fold(label: str) -> str:
    bucket = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 75 else "dev"


@app.function(image=CPU_IMAGE, volumes={"/data": VOLUME}, timeout=2 * 60 * 60)
def inventory(max_per_place: int = 12) -> dict:
    from huggingface_hub import hf_hub_download

    VOLUME.reload()
    DATA.mkdir(parents=True, exist_ok=True)
    archive = Path(
        hf_hub_download(
            "Yiwei-Ou/MMS-VPR",
            "Images.tar.gz",
            repo_type="dataset",
            local_dir=DATA,
        )
    )
    members: dict[str, list[str]] = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if not member.isfile() or Path(member.name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label = _label(member.name)
            if label:
                members.setdefault(label, []).append(member.name)
    selected = {
        label: sorted(names, key=lambda name: hashlib.sha256(name.encode()).digest())[:max_per_place]
        for label, names in members.items()
        if len(names) >= 2
    }
    manifest = {
        "source": "Yiwei-Ou/MMS-VPR Images.tar.gz",
        "split_policy": "sha256(location), buckets 0-74 train, 75-99 dev",
        "max_per_place": max_per_place,
        "archive_bytes": archive.stat().st_size,
        "all_images": sum(len(names) for names in members.values()),
        "all_places": len(members),
        "selected_images": sum(len(names) for names in selected.values()),
        "selected_places": len(selected),
        "fold_place_counts": dict(Counter(_fold(label) for label in selected)),
        "fold_image_counts": dict(Counter(_fold(label) for label, names in selected.items() for _ in names)),
        "overlap_places": [],
        "items": [
            {"archive_member": name, "place": label, "fold": _fold(label)}
            for label, names in sorted(selected.items())
            for name in names
        ],
    }
    output = DATA / "verifier_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    VOLUME.commit()
    return {key: value for key, value in manifest.items() if key != "items"}


@app.function(
    image=GPU_IMAGE,
    gpu="A10G",
    memory=32768,
    volumes={"/data": VOLUME},
    timeout=3 * 60 * 60,
)
def train_projection(epochs: int = 100, embedding_dim: int = 256) -> dict:
    import io
    import random

    import torch
    import torch.nn.functional as F
    from PIL import Image as PILImage
    from transformers import AutoImageProcessor, AutoModel

    VOLUME.reload()
    manifest = json.loads((DATA / "verifier_manifest.json").read_text())
    items = manifest["items"]
    archive = DATA / "Images.tar.gz"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    backbone = AutoModel.from_pretrained(
        "facebook/dinov2-base", torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    descriptors: list[torch.Tensor] = []
    selected_names = {item["archive_member"] for item in items}
    image_bytes: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if member.isfile() and member.name in selected_names:
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise FileNotFoundError(member.name)
                image_bytes[member.name] = extracted.read()
    if len(image_bytes) != len(items):
        raise ValueError(f"Selected image coverage {len(image_bytes)}/{len(items)}")
    for offset in range(0, len(items), 32):
            batch_items = items[offset : offset + 32]
            images = [
                PILImage.open(io.BytesIO(image_bytes[item["archive_member"]])).convert("RGB")
                for item in batch_items
            ]
            inputs = processor(images=images, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                patches = backbone(**inputs).last_hidden_state[:, 1:].float()
                positive = patches.clamp_min(0).pow(3).mean(dim=1).pow(1 / 3)
                negative = -(-patches).clamp_min(0).pow(3).mean(dim=1).pow(1 / 3)
                descriptor = torch.cat((positive, negative), dim=-1)
                descriptor = F.normalize(descriptor, dim=-1)
            descriptors.extend(descriptor.cpu())
    x = torch.stack(descriptors).to("cuda")
    labels = [item["place"] for item in items]
    folds = [item["fold"] for item in items]
    train_places = sorted({label for label, fold in zip(labels, folds) if fold == "train"})
    label_id = {label: index for index, label in enumerate(train_places)}
    train_indices = [index for index, fold in enumerate(folds) if fold == "train"]
    groups = {
        label: [index for index in train_indices if labels[index] == label]
        for label in train_places
    }

    projection = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, embedding_dim),
    ).to("cuda")
    optimizer = torch.optim.AdamW(projection.parameters(), lr=3e-4, weight_decay=1e-3)
    generator = random.Random(2707)
    losses = []
    projection.train()
    for _ in range(epochs):
        chosen_places = generator.sample(train_places, min(64, len(train_places)))
        batch_indices = []
        batch_labels = []
        for place in chosen_places:
            selected = generator.sample(groups[place], 2)
            batch_indices.extend(selected)
            batch_labels.extend([label_id[place], label_id[place]])
        embeddings = F.normalize(projection(x[batch_indices]), dim=-1)
        targets = torch.tensor(batch_labels, device="cuda")
        logits = embeddings @ embeddings.T / 0.07
        eye = torch.eye(len(batch_indices), dtype=torch.bool, device="cuda")
        logits = logits.masked_fill(eye, -1e9)
        positives = targets[:, None].eq(targets[None, :]) & ~eye
        loss = -(F.log_softmax(logits, dim=1) * positives).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    projection.eval()
    with torch.inference_mode():
        projected = F.normalize(projection(x), dim=-1)

    def retrieval(features: torch.Tensor) -> dict:
        recalls = {1: 0, 5: 0}
        queries = 0
        for place in sorted({label for label, fold in zip(labels, folds) if fold == "dev"}):
            indices = [i for i, (label, fold) in enumerate(zip(labels, folds)) if label == place and fold == "dev"]
            ordered = sorted(indices, key=lambda i: hashlib.sha256(items[i]["archive_member"].encode()).digest())
            database = ordered[::2]
            query = ordered[1::2]
            other_database = [
                i for i, fold in enumerate(folds)
                if fold == "dev" and i not in query
            ]
            db_features = features[other_database]
            for index in query:
                ranking = torch.argsort(features[index] @ db_features.T, descending=True)
                ranked_labels = [labels[other_database[int(position)]] for position in ranking[:5]]
                queries += 1
                for k in recalls:
                    recalls[k] += place in ranked_labels[:k]
        return {"queries": queries, **{f"recall_at_{k}": value / queries for k, value in recalls.items()}}

    raw_metrics = retrieval(x)
    projected_metrics = retrieval(projected)

    def make_pairs(fold: str):
        indices = [i for i, value in enumerate(folds) if value == fold]
        by_place = {
            place: [i for i in indices if labels[i] == place]
            for place in sorted({labels[i] for i in indices})
        }
        similarity = x[indices] @ x[indices].T
        position = {index: offset for offset, index in enumerate(indices)}
        pairs = []
        targets = []
        for place, place_indices in by_place.items():
            ordered = sorted(place_indices, key=lambda i: hashlib.sha256(items[i]["archive_member"].encode()).digest())
            for offset, anchor in enumerate(ordered):
                positive_index = ordered[(offset + 1) % len(ordered)]
                pairs.append((anchor, positive_index))
                targets.append(1.0)
                row = similarity[position[anchor]].clone()
                for candidate in indices:
                    if labels[candidate] == place:
                        row[position[candidate]] = -2
                negative_index = indices[int(torch.argmax(row))]
                pairs.append((anchor, negative_index))
                targets.append(0.0)
        return pairs, torch.tensor(targets, device="cuda")

    train_pairs, train_targets = make_pairs("train")
    dev_pairs, dev_targets = make_pairs("dev")

    def pair_features(pairs):
        a = x[[left for left, _ in pairs]]
        b = x[[right for _, right in pairs]]
        return torch.cat((torch.abs(a - b), a * b), dim=1)

    train_pair_x = pair_features(train_pairs)
    dev_pair_x = pair_features(dev_pairs)
    verifier = torch.nn.Sequential(
        torch.nn.Linear(train_pair_x.shape[1], 512),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(512, 1),
    ).to("cuda")
    verifier_optimizer = torch.optim.AdamW(verifier.parameters(), lr=2e-4, weight_decay=1e-3)
    verifier.train()
    pair_losses = []
    for _ in range(50):
        permutation = torch.randperm(len(train_pair_x), device="cuda")
        epoch_losses = []
        for offset in range(0, len(permutation), 256):
            chosen = permutation[offset : offset + 256]
            logits = verifier(train_pair_x[chosen]).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, train_targets[chosen])
            verifier_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            verifier_optimizer.step()
            epoch_losses.append(float(loss.detach()))
        pair_losses.append(sum(epoch_losses) / len(epoch_losses))

    def binary_metrics(scores, targets, threshold):
        predictions = scores >= threshold
        positives = targets.bool()
        negatives = ~positives
        tpr = float((predictions & positives).sum() / positives.sum())
        tnr = float((~predictions & negatives).sum() / negatives.sum())
        return {"balanced_accuracy": (tpr + tnr) / 2, "true_positive_rate": tpr, "true_negative_rate": tnr, "threshold": float(threshold)}

    train_cosine = torch.tensor(
        [float(x[left] @ x[right]) for left, right in train_pairs], device="cuda"
    )
    dev_cosine = torch.tensor(
        [float(x[left] @ x[right]) for left, right in dev_pairs], device="cuda"
    )
    candidates = torch.unique(train_cosine).sort().values
    raw_threshold = max(
        (float(value) for value in candidates),
        key=lambda value: binary_metrics(train_cosine, train_targets, value)["balanced_accuracy"],
    )
    verifier.eval()
    with torch.inference_mode():
        dev_probability = torch.sigmoid(verifier(dev_pair_x).squeeze(1))
    raw_pair_metrics = binary_metrics(dev_cosine, dev_targets, raw_threshold)
    trained_pair_metrics = binary_metrics(dev_probability, dev_targets, 0.5)
    output_dir = ROOT / "models" / "landmark-verifier"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(projection.state_dict(), output_dir / "mms-vpr-dinov2-projection.pt")
    torch.save(verifier.state_dict(), output_dir / "mms-vpr-dinov2-pair-verifier.pt")
    result = {
        "source": manifest["source"],
        "checkpoint": "facebook/dinov2-base",
        "training": "frozen backbone; supervised-contrastive projection",
        "epochs": epochs,
        "images": len(items),
        "train_places": len(train_places),
        "dev_places": len(set(labels)) - len(train_places),
        "place_overlap": 0,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "raw_dino": raw_metrics,
        "trained_projection": projected_metrics,
        "pair_dev_pairs": len(dev_pairs),
        "raw_cosine_pair_verifier": raw_pair_metrics,
        "trained_pair_verifier": trained_pair_metrics,
        "pair_initial_loss": pair_losses[0],
        "pair_final_loss": pair_losses[-1],
        "kilometervision_used": False,
    }
    (output_dir / "mms-vpr-dinov2-projection-metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    VOLUME.commit()
    return result


@app.local_entrypoint()
def main(action: str = "inventory", max_per_place: int = 12, epochs: int = 100) -> None:
    if action == "inventory":
        result = inventory.remote(max_per_place)
    elif action == "train":
        result = train_projection.remote(epochs)
    else:
        raise ValueError("action must be inventory or train")
    print(json.dumps(result, indent=2))
