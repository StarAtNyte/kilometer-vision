# KilometerVision 2707 — PATHFINDER

PATHFINDER is an evidence-first spatial reasoning pipeline for the Perception Test 2026 KilometerVision track. It answers compass, distance, map-trace, and landmark questions over walking-tour videos up to about 10 minutes.

## Data policy

Challenge media and derived artifacts belong only in Modal Volume `perception-test-2026-data`. Shared originals are under `/data/shared/walking-tours`; this project will reference them rather than duplicate videos locally. Local files contain code, manifests/schema documentation, and tiny synthetic fixtures only.

The August 11, 2026 organizer release currently reports 13 validation videos / 18 questions and 54 test videos / 986 questions. Re-download updated metadata and images before inference; the organizer explicitly replaced questions.

## Test leaderboard

Current EvalAI test leaderboard, checked **August 21, 2026**:

| Position | Team / submission | Top-1 | Visibility |
| ---: | --- | ---: | --- |
| 1 | BoundedAccord | 0.454 | Public |
| 2 | StarAtNyte — submission 579438 | 0.245436 | Public |
| 3 | Host_37595_Team | 0.196 | Public |

The public table currently exposes three test entries. The top method is explicitly named `bounded-geometry-qwen-zero-shot`; it is the only demonstrated route near 0.5. Refresh the official public table with:

```bash
python refresh_leaderboard.py
```

The original `gpudashboard0` Modal workspace was recovered after a profile mix-up. It still contains the shared source videos and p16 DPVO traces. A p16-only audit now covers 284 image-map test questions across 25 videos; 130 have a top-two margin below 0.005 and 95 below 0.003. The ranking artifact is `experiments/map-geometry-light/test.json`. This is unlabeled test evidence and is used only to target selective verification.

The follow-up motion-variant audit confirms that `fps1-focal0.9-w256-b256-p16` is the only complete strong lane on the five-question validation map set: it scores 4/5; p8 scores 0/5, fps0.5 scores 1/5, fps1.5 scores 0/5, and focal-0.75 variants score 0–1/5 (several older variants are incomplete). A panel-only Qwen3-VL-32B probe, deliberately omitting video and depth to isolate model reasoning, scores 1/5 and does not provide a safe override. Artifacts are `experiments/map-motion-variant-audit/validation.json` and `experiments/qwen-map-panel-probe/validation.json`; neither changes production predictions.

A two-case held-out Gemini 3.6 Flash probe shows why the validated endpoint/evidence gate cannot be replaced by a generic map prompt: it returned the existing p16 choice on one near-tie and a different choice on another, while the same lightweight prompt returned A on the labeled buBU validation map (the correct answer is C). These answer-blind probes are preserved under `experiments/gemini-test-probe/` and are explicitly excluded from production.

The corresponding Qwen3-VL-32B RGB control (geometry panel plus four ordered walking frames, no depth) also scored 1/5, exactly matching the panel-only control. This negative result rules out a simple “add more RGB context” fix; the remaining problem is trajectory/map alignment, not raw visual context. Its artifact is `experiments/qwen-rgb-panel-probe/validation.json`.

Two direct alignment tests were also negative. Anchoring both routes at S and normalizing by the S→E chord scored 3/5 (mirror variant 2/5), below centered ordered Procrustes at 4/5. Projecting raw DPVO camera positions instead of the yaw-integrated trace scored 3/5 for XZ and 2/5 for XY/YZ. The yaw-integrated p16 trace remains the best measured representation; artifacts are under `experiments/map-endpoint-alignment/` and `experiments/map-dpvo3d-projection/`.

Marker-guided S→E extraction plus a Fréchet term fixes buBU under a validation-sized gate, but the same gate changes 55 of 284 unlabeled test map answers (and ungated guided scoring changes 141). This is strong evidence of broad extraction instability, not a safe production improvement. The full audit is `experiments/map-marker-guided-audit/test.json`; it remains rejected.

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

A Fréchet weight sweep (0.1, 0.25, 0.5, 1.0) found 0.25–1.0 tied at 4/10 on the external gate; no coefficient exceeded that ceiling. Chamfer matching also tied at 4/10. This rules out simple metric-weight tuning as the main remaining fix.

Upstream DPVO sweeps were also run remotely: 0.5 FPS scored 1/5, 1.5 FPS scored 0/5, focal scale 0.75 scored 0/5, and the existing 1 FPS/focal 0.9/p16 configuration remained 4/5. A p32 trial exceeded GPU memory on long clips and was not usable. This makes the existing DPVO lane the evidence-backed motion baseline; further gains require loop closure or better visual-motion estimation rather than sampling tweaks.

The official VGGT-SLAM 2.0 runner was then tested remotely on the long `G0fWvIxy88A` validation window. It produced 60 keyframes but no loop-closure events and scored 0/1 on that map question. This is a negative loop-closure result, not a reason to replace DPVO.

DA3-Streaming (DA3NESTED-GIANT-LARGE-1.1, chunk size 60) was also run remotely, with both raw camera-center and gait-smoothed pedestrian trajectories. The available validation map windows scored 0/5 for each DA3 trajectory variant. Independent depth/streaming motion therefore does not currently beat DPVO; the 1 FPS/focal 0.9/p16 lane remains the motion baseline.

Finally, map-conditioned calibration was tested using the scale-invariant route compactness (polyline length divided by S→E chord). It reached only 3/5; a weight sweep from 0.05 to 0.5 never exceeded 3/5. This is another useful negative result: endpoint/length priors overfit the five-question validation set and remain opt-in.

The curvature term was then swept on the actual DPVO 1 FPS/p16 trajectories (weights 0.1, 0.35, 0.5, 1.0, 2.0). The existing 0.35 and 0.5 settings both scored 4/5; higher or lower weights fell to 2–3/5. Turn weighting is therefore already near its useful optimum.

A final representation check compared the DPVO trace mask directly against each candidate's red-route mask using symmetric pixel Chamfer distance. It scored only 1/5 for the DPVO 1 FPS lane, confirming that raster noise and map rendering artifacts make pixel-space matching worse than the ordered skeleton scorer.

The existing ORB/essential-matrix motion lane was then revisited: its per-frame translation directions were integrated with accumulated yaw under all four camera-axis sign conventions. Every convention scored only 1/5. This confirms that raw monocular translation direction is too unstable for the pedestrian route and should not override DPVO.

Finally, DPVO's stored 3-D positions were projected directly on XY, XZ, and YZ axes instead of the current unconstrained PCA projection. The best direct-axis variant reached 3/5, below the PCA lane's 4/5, so vertical-axis contamination is not the dominant failure.

PCA chirality was tested separately by adding reflection penalties from 0.005 through 0.08. Every nonzero penalty dropped DPVO to 2/5, confirming that the route maps do not provide a safe global mirror convention for these relative trajectories.

Candidate-map Tesseract OCR was also scored by label richness, with outlier thresholds from 0 to 8 labels. It reached only 1/5; the correct landmark-rich Czech map is a useful diagnostic example, but OCR richness is not a safe automatic override.

The first positive semantic result is an endpoint-focused Gemini 3.6 Flash duel gate. On all five validation map questions, it preserves the DPVO 1 FPS/p16 answer unless DPVO's top-two margin is below 0.005, Gemini's endpoint-focused duel confidence is at least 0.99, and the answer is unchanged when candidate display order is reversed. That gate changes only the ambiguous Czech case (D→C) and reaches 5/5 versus DPVO's 4/5. A replayed threshold sweep gives 5/5 at both 0.98 and 0.99, but 4/5 at 0.995 because the successful Czech correction is reported at exactly 0.99; 0.99 is therefore the tighter candidate setting. This is promising but still validation-sized and API-dependent; it remains an experiment until tested on held-out/public data.

For the two cases where the gate can change a decision, reversing candidate display order produced the same Gemini answer and 0.99 confidence (Czech C; York E). The evaluator now requires base/reversed agreement before overriding DPVO. This reduces order-prompt risk, but does not remove the small-validation limitation.

Qwen3.8-27B was also rerun remotely at 32 uniformly sampled frames on the three cases where its earlier 16-frame independent-map audit failed. It reached 1/3 (G0 correct; Htw and UK still wrong), so temporal oversampling did not resolve its route/map mismatch and it is not used as an override.

The Gemini model inventory exposes `gemini-3.7-flash`, so the duel runner now accepts an explicit model without overwriting 3.6 artifacts. Initial 3.7 checks are encouraging: it selected the corrected Czech map C in both normal and reversed display order at 0.99, selected the York map E (the other 3.6 gate correction) in both orders at 0.99, selected the Salzburg map A (correct) at 0.98 where 3.6 selected C, and selected the Warsaw map A (correct) at 0.99. The Salzburg and Warsaw reversed calls encountered provider failures (503 and malformed/truncated JSON), so 3.7 is recorded as a promising candidate rather than promoted into the production gate until those order-consistency checks are completed.

The evaluator now prefers a complete Gemini 3.7 base/reverse artifact pair over 3.6, while falling back to 3.6 and then DPVO when the pair is incomplete. Re-running the full remote map evaluator with this preference remains 5/5; the two actual overrides are both 3.7-backed, and the incomplete Salzburg/Warsaw pairs do not override their already-correct DPVO predictions.

A lightweight remote triage of the 431 test map-family questions found 284 image-candidate map questions and 147 text route-instruction questions. All 284 image questions have a DPVO ordered-map ranking; 130 have a top-two margin below 0.005. The 147 text questions are a separate route-instruction bottleneck, not failed image-map extraction. This is a measurement for planning, not a test-label-informed change.

The production evidence audit found 38 route-instruction questions with no Qwen2.5-VL answer (across eight videos). A separate remote Qwen2.5-VL retry produced parseable answers for all 38; 28 differ from the existing deterministic fallback. These predictions are stored under `experiments/vlm-route-retry/` but are not merged into a submission because test labels are unavailable. A 32-frame temporal-stability rerun was stopped after excessive latency without writing an artifact; the 16-frame retry is the usable cost/coverage point.

An 18-question labeled validation audit was then attempted with the same Qwen2.5-VL lane. The restored shared-video mount was unavailable after a Modal volume snapshot race, so 10 text/video questions correctly returned the explicit `no_images` status and were not scored. The eight questions with bundled reference images were parseable but only 2/8 top-1 (map 1/5; landmark 1/3), with a strong E-choice bias. This is a negative control, not evidence for replacing the geometry specialists; the audit artifacts are under `experiments/vlm-validation-audit2/`.

The reversible candidate `submissions/test_submission_route_retry_2026-08-21.json` applies only those 28 route-instruction replacements to the current 986-question baseline. `submission.audit_predictions` confirms the archive is schema-valid (986/986, no missing or extra IDs); it has not been submitted and its leaderboard effect is unknown.

As a stronger-model control, Qwen3.8-27B was run remotely on the nine 4MY route-instruction questions with the same 16-frame budget. It produced nine parseable answers but disagreed with the Qwen2.5 retry on 8/9 questions. This is a negative reliability result; the Qwen3.8 outputs remain an isolated ablation under `experiments/qwen38-route/` and are not merged.

The recent `ybrrraway/V2LO-7B` Video2Layout checkpoint was tested as an image-only map reader on the five labeled validation map questions, with randomized candidate display order and no video/answer access. It achieved 1/5 (raw answers C, E, B, D, B), substantially below the geometry/Gemini route and therefore is not an override. The run is preserved under `experiments/map-v2lo/validation.json`; the parser explicitly extracts the tagged answer rather than matching the `A` in `<answer>`.

The same randomized candidate-panel control with `Qwen/Qwen3-VL-8B-Instruct` and the restored p8 DPVO trace scored 0/5 (C, B, B, C, B). This isolates the failure to the visual route/evidence representation rather than model size; the result is preserved under `experiments/map-qwen3-8b/validation.json` and is not used in production.

To recover the lost p16 traces, `modal_fetch_source_video.py` attempted to rehydrate the official validation source remotely. Both standard and Android yt-dlp clients were rejected by YouTube's bot/sign-in gate; the Hugging Face Walking_Tours repository contains metadata/images and YouTube links, not the videos themselves. No source media or labels were modified.

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
