"""Answer-blind spatial OCR-anchor audit for candidate maps."""
from pathlib import Path
import argparse, glob, json, math, re, unicodedata
from difflib import SequenceMatcher

STOP = {"and","around","data","downtown","google","map","maps","market","old","part","street","the","tour","town","video","walk","walking","with","world","city","center","centre","copyright","route","filmed","august","osmo","pocket","mic","zoom","uhd","after","there","from","right","our","less"}
def norm(text): return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
def tokens(values): return {t for t in re.findall(r"[a-z]{4,}", norm(" ".join(values))) if t not in STOP}
def similar(a,b): return a == b or (min(len(a),len(b)) >= 5 and SequenceMatcher(None,a,b).ratio() >= .86)
def run(directory, anchors_path, manifest_path, radius):
    anchors = json.loads(anchors_path.read_text())["records"]
    manifest = json.loads(manifest_path.read_text())
    rows=[]
    for path in sorted(glob.glob(str(directory / "*.json"))):
        payload=json.loads(Path(path).read_text()); vid=payload["video_id"]; qid=str(payload["question_id"])
        question=manifest[vid][qid]; anchor_tokens=tokens(anchors[vid].get("route_anchor_candidates",[])+[anchors[vid].get("title","")]); scores=[]
        for record in payload["records"]:
            points=[(record["markers"][key][0]/400, record["markers"][key][1]/400) for key in ("S","E")]
            hits=[]
            for token in record["tokens"]:
                if min(math.hypot(token["x"]-x, token["y"]-y) for x,y in points) <= radius and any(similar(anchor,norm(token["text"])) for anchor in anchor_tokens): hits.append(norm(token["text"]))
            scores.append(len(set(hits)))
        prediction=max(range(5), key=lambda i:scores[i]) if scores else None
        rows.append({"video_id":vid,"question_id":qid,"prediction":prediction,"answer_id":question.get("answer_id"),"scores":scores})
    return {"radius":radius,"rows":rows,"correct":sum(r["prediction"]==r["answer_id"] for r in rows),"questions":len(rows),"ground_truth_used_for_inference":False}
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("directory",type=Path); p.add_argument("--anchors",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--radius",type=float,default=.25); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); result=run(a.directory,a.anchors,a.manifest,a.radius); a.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
