from pathlib import Path
import modal

V = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
A = modal.App("kilometervision-volume-inventory")
I = modal.Image.debian_slim(python_version="3.11")

@A.function(image=I, volumes={"/data": V}, timeout=10 * 60)
def run():
    V.reload()
    root = Path("/data")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path), "bytes": path.stat().st_size})
    return rows

@A.local_entrypoint()
def main():
    import json
    print(json.dumps(run.remote(), indent=2))
