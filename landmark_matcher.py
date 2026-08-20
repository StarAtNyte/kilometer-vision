"""Reference-image retrieval against question-independent frame sketches."""
from __future__ import annotations
import io, json, math, zipfile
from pathlib import Path
from typing import Any
from route_solver import _image_vector, _distance


def match_reference(manifest_path: Path, archive_path: Path, reference: str) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    members = [name for name in zipfile.ZipFile(archive_path).namelist() if Path(name).name == Path(reference).name]
    if len(members) != 1:
        return {"status": "reference_not_unique", "similarity": 0.0, "timestamp_seconds": None}
    with zipfile.ZipFile(archive_path) as archive:
        data = archive.read(members[0])
    try:
        from PIL import Image
        ref_path = manifest_path.parent / (".reference-" + Path(reference).name)
        ref_path.write_bytes(data)
        ref_vec = _image_vector(ref_path, size=16)
        ref_path.unlink(missing_ok=True)
    except Exception:
        return {"status": "reference_decode_failed", "similarity": 0.0, "timestamp_seconds": None}
    best = {"similarity": 0.0, "timestamp_seconds": None, "frame_index": None}
    for frame in payload.get("frames", []):
        path = manifest_path.parent / frame["path"]
        if not path.is_file(): continue
        sim = 1.0 - float(_distance(ref_vec, _image_vector(path, size=16)) or 1.0)
        if sim > best["similarity"]:
            best = {"similarity": round(sim, 6), "timestamp_seconds": frame["timestamp_seconds"], "frame_index": frame["index"]}
    best["status"] = "matched" if best["timestamp_seconds"] is not None else "no_frames"
    return best
