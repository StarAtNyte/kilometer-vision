# KilometerVision 2707 — PATHFINDER

PATHFINDER is an evidence-first spatial reasoning pipeline for the Perception Test 2026 KilometerVision track. It answers compass, distance, map-trace, and landmark questions over walking-tour videos up to about 10 minutes.

## Data policy

Challenge media and derived artifacts belong only in Modal Volume `perception-test-2026-data`. Shared originals are under `/data/shared/walking-tours`; this project will reference them rather than duplicate videos locally. Local files contain code, manifests/schema documentation, and tiny synthetic fixtures only.

The August 11, 2026 organizer release currently reports 13 validation videos / 18 questions and 54 test videos / 986 questions. Re-download updated metadata and images before inference; the organizer explicitly replaced questions.

## Test leaderboard

Current EvalAI test leaderboard plus our best private submission, checked **August 20, 2026**:

| Position | Team / submission | Top-1 | Visibility |
| ---: | --- | ---: | --- |
| 1 | BoundedAccord | 0.454 | Public |
| 2* | StarAtNyte — submission 579590 | **0.250507** | Private; projected position if made public |
| 2 | StarAtNyte — submission 579438 | 0.245436 | Public |
| 3 | Host_37595_Team | 0.196 | Public |

`2*` is a comparison position, not an official EvalAI rank while submission 579590 remains private. It is the production baseline: 247/986 correct and three answers better than submission 579535 after the isolated compass change. Refresh the official public table with:

```bash
evalai challenge 2707 leaderboard 7010
```

## Architecture

See [`RESEARCH.md`](RESEARCH.md). The implementation will build a timestamped route-state lattice with visual place retrieval, OCR, local geometric matching, camera-motion/heading estimates, uncertain trajectory, candidate-map normalization, and specialist option scoring. A VLM is a verifier, not the sole spatial memory.

The strongest geometry-conditioned VLM ablation is available in `modal_bounded_geometry_qwen.py`: Qwen3-VL-32B sees motion-aware exact-window frames, relative-depth views, the DPVO trace, and normalized ordered S→E candidate routes. It is experimental only. On the five validation map questions it scored 1/5 and failed a map-cell layout-invariance control, so it does not feed production or any submission.

Two newer answer-blind feasibility lanes are also rejected from production. `modal_trajloc_feasibility.py` ran the released TrajLoc checkpoint against an 81-cell York satellite gallery, but York and an Istanbul negative retrieved the identical top tile (the negative even had the higher best score). `modal_vggt_slam2.py` ran official VGGT-SLAM 2.0 on the same York map window at 1/2 FPS; it retained 60/84 optimized poses, found zero loop closures, and failed cross-density stability (normalized disparity 0.220, turn correlation -0.067). These are recorded failure modes, not option scorers.

`modal_gemini_geometry.py` tests the stronger native-video route-reasoning hypothesis. Gemini 3.1 Pro was unavailable because the configured API project has zero quota for that model, so the frozen gate used Gemini 3.5 Flash. Across all five validation map questions, the joint full-video/panel/original-map prompt scored 2/5 in base order and 1/5 after a CDEAB display permutation. It preserved candidate identity on 4/5 questions but was both stable and correct on only York (1/5). The model often named the correct streets and landmarks while choosing the wrong near-identical map, so it is not used in production. A candidate-hidden two-stage follow-up stalled reproducibly and yielded no accepted result.

The subsequent isolated-candidate experiments found a more useful boundary. Gemini 3.6 Flash map-only audits scored 3/5 directly; exact reversed-route direction checks and OCR/OSM start-landmark registration explain all 5/5 validation maps, but this is not yet treated as a robust estimate because validation contains only five map questions. `modal_qwen36_independent_maps.py` evaluates the newer native-multimodal `Qwen/Qwen3.8-27B` without candidate-order exposure. Direct generated scores collapsed to `[92,92,92,92,92]`; raw YES/NO logits scored 1/5 on the hard Czech case until candidate-blind route evidence was frozen. With public source-title grounding, the frozen-evidence variant reached 3/5 (Istanbul, York, Czech). Doubling from 16 to 32 frames did not fix Salzburg or Warsaw, and map-only coordinate OCR returned no labels for Salzburg. The accepted conclusion is narrow: Qwen3.8 is useful for candidate-blind semantic/OCR evidence, not for calibrated candidate selection or compass direction. None of these ablations feed a submission.

### External geometry gates (Modal-only)

Two label-independent checks now isolate geometry from KilometerVision. A six-example-per-task SAW-Bench pilot (`modal_saw_benchmark.py`) gave Qwen3.8-27B 2/6 direction, 2/6 shape, and 0/6 reverse-plan accuracy (4/18 overall), confirming that a stronger VLM does not by itself maintain route state. An EgoWalk oracle-pose benchmark (`egowalk_oracle_geometry.py`) gave the ordered scorer 10/10 top-1 on both directed and reversal-free candidate sets, with all ten reversals separated. This validates the mathematical scorer, not visual odometry.

The rendered five-map follow-up (`modal_egowalk_synthetic_maps.py`) found a remaining image-extraction issue: S/E markers were recovered on 8/10 sessions, but the synthetic raster route was not consistently ranked top-1. This is a useful negative gate—the next work is to make marker/route extraction robust before trusting visual motion or submitting a new map system. All external-video/model work runs on Modal; the local machine is limited to lightweight tests and repository operations.

The next research-backed ablation adds an opt-in discrete Fréchet term to the ordered scorer. It improved the thin-route synthetic gate from 2/10 to 4/10 (and reached 4/10 across all tested rendering variants), but remained 1/5 on the tiny KilometerVision validation map set. The default scorer therefore remains unchanged; Fréchet is evidence for further work, not a submission replacement.

An S/E-guided skeleton shortest-path variant was also tested. It matched the Fréchet result externally (4/10) and remained 1/5 on validation; rank fusion did not improve it. The production path therefore stays on the proven DPVO fps1/p16 lane while these alternatives remain explicit ablations.

## Planned commands

```bash
modal run modal_pipeline.py --action bootstrap
modal run modal_pipeline.py --action ingest --split validation
modal run modal_pipeline.py --action ingest --split test
modal run modal_pipeline.py --action build-frames
modal run modal_pipeline.py --action index-split --split validation
modal run modal_pipeline.py --action index-split --split test
modal run modal_pipeline.py --action compile-split --split validation
modal run modal_pipeline.py --action predict --split validation
modal run modal_pipeline.py --action predict --split test
modal run modal_pipeline.py --action audit --split test
```

## Bootstrap and audit commands

The first safe code slice is in [`modal_pipeline.py`](modal_pipeline.py). It mounts
Volume `perception-test-2026-data` at `/data`, creates the canonical
`/data/2707-kilometer-vision/{manifests,frames,route-features,ocr,embeddings,trajectories,evidence,predictions,logs}` tree, and downloads only organizer JSON/image ZIP artifacts. Walking-tour videos remain under `/data/shared/walking-tours` and are not downloaded or copied.

```bash
# Local dependency-light unit tests (no network or Modal required)
python -m unittest discover -s tests -v

# Run the Modal actions; each artifact is written as *.partial then atomically renamed.
modal run modal_pipeline.py --action bootstrap
modal run modal_pipeline.py --action ingest --split validation
modal run modal_pipeline.py --action ingest --split test
modal run modal_pipeline.py --action audit --split validation
modal run modal_pipeline.py --action audit --split test

# Question-independent, bounded 1 FPS low-resolution frame artifacts. Originals
# are read in place from /data/shared/walking-tours and are never copied.
modal run modal_pipeline.py --action build-frames --sample-fps 1 --max-frames 600 --frame-width 320
modal run modal_pipeline.py --action build-routes
```

`index-split` reads only referenced candidate images from the ingested ZIP;
source ZIP members and shared walking-tour videos are never modified or copied.
It writes atomic dependency-light descriptors to
`route-features/{split}_images.json` (SHA-256, byte size, dimensions, and an
optional Pillow RGB thumbnail/perceptual hash) and compiles question-local
family/option differences to `route-features/{split}_questions.json`.
The compiler never reads organizer answer fields. Coverage and image-index
errors are recorded in `logs/{split}_compiler_audit.json`. `compile-split`
regenerates only the question compiler artifacts without decoding images.

`build-frames` writes one manifest and dependency-light SHA-256/size feature
artifact per source under `/data/2707-kilometer-vision/frames`, plus an atomic
run report at `logs/frame_build.json`. Existing artifacts are skipped when the
source fingerprint and sampling configuration match. Missing frames are decoded
individually, so interrupted runs resume without starting over. `ffprobe` and
`ffmpeg` are preferred; OpenCV is used as a dependency-light fallback when those
binaries are unavailable. Sampling is capped by `--max-frames` and timestamps are
validated against probed duration/frame rate.

`modal_geometry.py` now has a DA3 GPU lane. It ran `depth-anything/DA3-LARGE` over all required manifests (136 total, including every validation/test ID) on an A100-80GB and persisted camera/depth summaries under `/data/2707-kilometer-vision/geometry/`. Run one video with `modal run modal_geometry.py --video-id VIDEO_ID --window 8`, or all manifests with `--video-id __all__`. `build-geometry` materializes canonical contracts; `build-routes` produces uncertainty-aware visual novelty and turn-candidate traces from frame manifests. It intentionally leaves absolute heading and metric scale unavailable until calibrated motion/geometry features are added; novelty peaks are candidates, not labels. `build-visual-index` writes bounded frame descriptors under `embeddings/` and optional OCR rows under `ocr/`. `landmark_matcher.py` performs reference-image-to-frame retrieval and feeds distance specialist estimates when artifacts are available.

`geometry_contract.py` turns the raw DA3 poses into the real `geometry_backend.extract_route_signature` contract (turn sequence, segment lengths, uncertainty) that was previously computed but never called anywhere. `modal_pipeline.py --action build-route-signature` materializes `geometry/<video_id>.route_signature.json`, including a short text narrative (e.g. "2 sustained turn(s) detected: right ~85°, left ~40°...") that both feeds the VLM prompt below and is available to specialists.

`modal_vlm.py` is the evidence-card VLM lane: a self-hosted Qwen2.5-VL-7B-Instruct model on Modal A100-80GB reads each question's text/options, sampled frames, reference/candidate images, the route-signature narrative, and OCR text, and returns exactly one letter A-E. Run one video with `modal run modal_vlm.py --video-id VIDEO_ID --split validation`, or a whole split with `--video-id __all__ --split test`. It writes `vlm/<video_id>.json`; `submission.py:select_option` uses a clean ("ok"-status) VLM answer as the primary decision and only falls through to the deterministic specialists/hash fallback when the VLM lane is absent, unparseable, or errored — the "never invent evidence" contract stays intact either way. This is a real GPU-cost, unverified-until-run lane: its prompt/frame-selection/answer-parsing logic (`vlm_solver.py`) is dependency-light and unit tested, but the model call itself has only been exercised by import/syntax checks, not a live run.

`predict` produces a deterministic, schema-safe artifact plus evidence traces. Geometry availability is included in specialist retrieval context; DA3 remains relative-scale evidence and does not claim absolute heading or metric distance. Both artifacts are audited for complete question coverage:

```bash
modal run modal_pipeline.py --action predict --split validation
modal run modal_pipeline.py --action predict --split test
```

Ingestion validates the release counts (13/18 validation and 54/986 test), video IDs,
exactly five options, image references against the matching ZIP, and frame-to-seconds
consistency. Answers, when present in organizer metadata, are retained only in the
organizer manifest and are never copied into the audit report. No challenge labels
are used for training, tuning, or prompting.
