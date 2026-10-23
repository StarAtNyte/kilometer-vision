"""Estimate sequential camera motion inside each exact question window on Modal.

Unlike the old 60-frame global DA3 artifacts, this lane samples every requested
interval directly from the source video.  It uses adjacent-frame feature geometry,
stores every accepted/rejected step, and makes no metric-scale claim.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import modal

from kilometer_vision.motion_solver import compass_option_index, compass_prediction, summarize_steps
from kilometer_vision.question_planner import classify_question_facets

ROOT = Path("/data/2707-kilometer-vision")
SOURCE_ROOT = Path("/data/shared/walking-tours")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("opencv-python-headless", "numpy").add_local_python_source('kilometer_vision')
app = modal.App("kilometervision-motion")
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _source(video_id: str) -> Path | None:
    directory = SOURCE_ROOT / video_id
    if not directory.is_dir(): return None
    return next((path for path in sorted(directory.iterdir()) if path.suffix.lower() in VIDEO_SUFFIXES and ".partial" not in path.stem), None)


def _extract(source: Path, start: float, end: float, fps: float, directory: Path) -> list[Path]:
    pattern = directory / "%06d.jpg"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-t", str(max(0.0, end-start)), "-i", str(source), "-vf", f"fps={fps},scale=640:-2", "-q:v", "4", str(pattern)]
    subprocess.run(command, check=True, timeout=max(60, int(end-start)*2))
    return sorted(directory.glob("*.jpg"))


def _step(left, right, timestamp: float, focal_scale: float = 0.9) -> dict[str, Any]:
    import cv2
    import numpy as np
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=12)
    kp1, des1 = orb.detectAndCompute(left, None); kp2, des2 = orb.detectAndCompute(right, None)
    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return {"timestamp_seconds": timestamp, "accepted": False, "reason": "insufficient_features"}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    good = [a for a, b in pairs if a.distance < 0.75*b.distance]
    if len(good) < 20:
        return {"timestamp_seconds": timestamp, "accepted": False, "reason": "insufficient_matches", "matches": len(good)}
    p1=np.float32([kp1[m.queryIdx].pt for m in good]); p2=np.float32([kp2[m.trainIdx].pt for m in good])
    height,width=left.shape[:2]; focal=focal_scale*width; center=(width/2.0,height/2.0)
    essential, mask=cv2.findEssentialMat(p1,p2,focal=focal,pp=center,method=cv2.RANSAC,prob=0.999,threshold=1.5)
    if essential is None:
        return {"timestamp_seconds": timestamp, "accepted": False, "reason": "essential_failed", "matches": len(good)}
    inliers, rotation, translation, _=cv2.recoverPose(essential,p1,p2,focal=focal,pp=center,mask=mask)
    yaw=math.degrees(math.atan2(float(rotation[0,2]),float(rotation[2,2])))
    accepted = inliers >= 15 and abs(yaw) <= 50.0
    return {"timestamp_seconds": timestamp, "accepted": accepted, "reason": "ok" if accepted else "geometry_rejected", "matches": len(good), "inliers": int(inliers), "confidence": min(1.0, inliers/max(20,len(good))), "yaw_delta_degrees": yaw, "translation_direction": [float(x) for x in translation.reshape(-1)]}


@app.function(image=IMAGE, volumes={"/data": VOLUME}, cpu=8.0, memory=16384, timeout=6*3600)
def run_motion(split: str = "validation", video_id: str = "__all__", fps: float = 1.0, facet: str = "") -> dict[str, Any]:
    import cv2
    manifest=json.loads((ROOT/"manifests"/f"{split}.json").read_text())
    targets=sorted(manifest) if video_id == "__all__" else [video_id]
    summaries=[]
    for target in targets:
        source=_source(target); records={}
        for key,question in manifest.get(target,{}).items():
            facets=classify_question_facets(question)
            if facet and facet not in facets:
                continue
            if not any(facet in facets for facet in ("compass","map","distance")):
                continue
            start=float(question.get("start_seconds",0.0)); end=float(question.get("end_seconds",start))
            if source is None or end <= start:
                records[str(key)]={"status":"missing_video_or_window","facets":list(facets)}; continue
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    paths=_extract(source,start,end,fps,Path(temporary))
                    images=[cv2.imread(str(path),cv2.IMREAD_GRAYSCALE) for path in paths]
                    steps=[_step(a,b,start+(index+1)/fps) for index,(a,b) in enumerate(zip(images,images[1:])) if a is not None and b is not None]
                summary=summarize_steps(steps)
                records[str(key)]={"status":"ok","facets":list(facets),"fps":fps,"start_seconds":start,"end_seconds":end,"frames":len(paths),"summary":summary,"compass_prediction":compass_prediction(question,summary["net_yaw_degrees"]),"compass_option_index":compass_option_index(question,summary["net_yaw_degrees"]),"steps":steps}
            except Exception as error:
                records[str(key)]={"status":f"error:{type(error).__name__}","facets":list(facets),"error":str(error)}
        # Facet runs must never overwrite one another. The legacy flat path mixed
        # compass/map/distance records and made preserved predictions impossible
        # to reproduce after a later experiment.
        namespace = facet or "all"
        output=ROOT/"motion"/split/namespace/f"{target}.json"; _atomic_json(output,records); VOLUME.commit()
        summaries.append({"video_id":target,"questions":len(records),"ok":sum(row.get("status")=="ok" for row in records.values())})
    return {"split":split,"fps":fps,"facet":facet or None,"videos":summaries}


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=300)
def list_video_ids(split: str) -> list[str]:
    return sorted(json.loads((ROOT/"manifests"/f"{split}.json").read_text()))


@app.local_entrypoint()
def main(split: str="validation", video_id: str="__all__", fps: float=1.0, facet: str=""):
    if video_id != "__all__":
        result = run_motion.remote(split, video_id, fps, facet)
    else:
        ids = list_video_ids.remote(split)
        calls = [run_motion.spawn(split, target, fps, facet) for target in ids]
        partial = [call.get() for call in calls]
        result = {"split": split, "fps": fps, "facet": facet or None,
                  "videos": [row for item in partial for row in item["videos"]]}
    print(json.dumps(result,indent=2))
