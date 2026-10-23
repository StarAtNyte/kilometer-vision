# KilometerVision 2707 — research and architecture

Last verified: 2026-08-17. This document is a design baseline; no validation/test answer is used for training, prompting, tuning, or manual labeling.

## Current official state

The EvalAI API challenge 2707 record and the organizer metadata URLs were checked directly:

- Challenge: `https://eval.ai/web/challenges/challenge-page/2707/`
- Updated validation metadata (August 11 release): `https://storage.googleapis.com/dm-perception-test/kilometer_vision_2026_val/kilometervision_valid_20260811.json`
- Updated public test metadata: `https://storage.googleapis.com/dm-perception-test/kilometer_vision_2026_val/kilometervision_test_public_20260811.json`
- Images/maps: matching `*_images_20260811.zip` URLs in the challenge description.

The August 11 organizer notice says both validation and test were updated and must be re-downloaded. The updated metadata currently contains **13 validation videos / 18 questions** and **54 test videos / 986 questions** (not the older 10-video description). Submission format is a JSON object keyed by video ID; each value is a list of `{id, answer, answer_id}`. Top-1 accuracy; all questions are required.

The shared Modal Volume `perception-test-2026-data` already contains the walking-tour originals under `/data/shared/walking-tours/<video_id>/`. A check found the updated 54 test IDs represented there, including completed `.mkv` files. There is no dedicated `/data/2707-kilometer-vision` tree yet. Metadata, image archives, manifests, indexes, and predictions must be added there; originals must remain shared and must not be copied.

Rules: process each test video/question independently; do not use test/validation answers for training, self-supervision, few-shot prompting, or human labeling. Perception Test train and other eligible external data are allowed.

## Task structure

The four answer families are materially different:

1. **Compass:** accumulate camera heading changes from an initial cardinal/intercardinal heading to a queried endpoint.
2. **Distance:** determine whether a supplied landmark/reference image occurs, then estimate route/straight-line/line-of-sight distance.
3. **Map trace:** choose one of five candidate route images.
4. **Landmark:** match a supplied image to a location in the walk and answer distance or occurrence.

A single VLM over uniformly sampled frames is not the winning default: it loses brief signs, accumulates turn errors, confuses head yaw with path turns, and cannot reliably compare candidate maps.

## Proposed winning system: PATHFINDER

**PATHFINDER = Pose-Anchored Temporal Hierarchy for Five-way Inference, Navigation, Distance, and Evidence Retrieval.** The core novelty is to make every answer a comparison over a *route-state lattice* rather than a free-form video summary.

### A. Immutable media and route lattice

For each video, cache question-independent artifacts in Modal:

- exact frame/audio timestamps and a low-resolution proxy;
- 1–2 FPS keyframes plus shot/turn/change-point candidates;
- OCR text and text boxes, place/landmark captions, DINOv2/SigLIP2 and geolocation/place embeddings;
- dense short-window local features for landmark matching;
- optical-flow/feature tracks, camera yaw, horizon/vanishing-point, walking-motion estimates;
- a smoothed 2-D route polyline with uncertainty, plus heading and cumulative distance intervals.

Each lattice node is an event card, not a single frame:

```text
[t0,t1], representative_frames, OCR, place_labels, local/global embeddings,
heading_distribution, displacement_interval, turn_distribution, provenance
```

Adjacent cards are connected by `before/after/near/same-place/turn/loop-closure` edges. Retain raw frame IDs so every prediction has an inspectable certificate.

### B. Question compiler and candidate-difference retrieval

Parse the question and each option into atomic predicates. Use the *differences between options* as retrieval queries, not only the shared question text. Examples:

- compass: endpoint heading and every sustained turn candidate;
- distance: reference-image place match, endpoint/startpoint anchors, and metric type;
- map: route topology, turn sequence, endpoint displacement, and candidate-map image features;
- landmark: image-to-card place matching followed by temporal and distance checks.

Never use answer labels or validation answers to choose routing. The compiler is deterministic and question-local.

### C. Specialist solvers

**Compass solver.** Estimate relative yaw from long-baseline feature tracks and scene geometry, detect sustained path turns with hysteresis, and maintain a particle distribution over heading. Reject short camera/head sweeps unless supported by translation/landmark change. Integrate from the question's stated initial heading; score each option by circular posterior probability.

**Landmark solver.** Use a two-stage image matcher: global place retrieval (DINOv2/SigLIP2) followed by local feature geometric verification (LoFTR/LightGlue-style matching where available). Re-rank with OCR, architecture, storefront text, and temporal neighboring cards. The image question is a search key, never a training example.

**Distance solver.** Prefer route-state displacement and loop closures over monocular depth. Estimate metric scale using walking cadence/speed priors, visible map geometry, known map scale, and repeated-place anchors. Return an uncertainty interval and compare the five options; separate straight-line from traveled-route distance explicitly.

**Map solver.** Decode each candidate map, normalize rotation/scale/translation, extract its polyline/turn graph, and compare topology first, then normalized curvature/turn sequence, then absolute orientation when available. Use the video trajectory posterior as a distribution, not a single brittle path. Candidate maps are treated symmetrically and option order is randomized during internal evaluation.

### D. Certificate-aware compute policy

- Local landmark match: retrieve, then inspect a small temporal neighborhood.
- Compass endpoint: inspect the full queried segment, with extra compute at turn/change candidates.
- Map trace: use the whole route lattice and all five maps.
- Distance/absence: run a global candidate sweep; top-K retrieval cannot prove that a reference image never appeared.

The final scorer combines predicate support, contradiction, uncertainty, and option-order-consistent specialist votes. It must retain timestamps and matching evidence even though only answer fields are submitted.

### E. Model stack

Recommended first implementation uses open, reproducible components: FFmpeg/OpenCV, LightGlue or equivalent local matching, DINOv2/SigLIP2 place embeddings, PaddleOCR, RAFT/optical flow or feature-track odometry, and a Qwen-VL-class short-card verifier on Modal. Add a stronger omni/VLM verifier only for ambiguous cards. Use pretrained weights and external/Perception Test train folds only; do not train on this challenge's examples.

The architecture intentionally avoids depending on one speculative 2026 model. A commercial VLM can be an optional audit lane, but deterministic geometry and candidate-map scoring are the primary path because the benchmark asks spatial relations, not generic captioning.

## Development and validation protocol

1. Bootstrap `/data/2707-kilometer-vision/{manifests,frames,route-features,ocr,embeddings,trajectories,evidence,predictions,logs}`.
2. Ingest the August 11 metadata and image archives atomically; validate IDs, option counts, image references, and frame/time conversions.
3. Build question-independent indexes for shared originals, reusing existing media.
4. Build a synthetic route benchmark from eligible Perception Test train videos and public datasets. Split by source video ID. Tune odometry thresholds, retrieval/reranking, and calibration there only.
5. Run the frozen pipeline on the updated validation set as an external check. Do not revise based on individual validation answers.
6. Run test questions one at a time, export exact schema, and audit completeness before any submission.

Important metrics beyond top-1: landmark Recall@K and geometric verification precision; turn-event F1; heading circular error; route polyline/turn-graph similarity; distance calibration; map choice accuracy; evidence coverage; option-order variance; runtime and Modal cost.

## Main risks

- **No GPS/IMU:** monocular scale and heading drift. Mitigate with loop closures, long-baseline tracks, uncertainty particles, map topology, and option-relative scoring.
- **Head turns mistaken for route turns:** require sustained evidence and translation consistency.
- **Repeated streets/landmarks:** use OCR/local geometry and temporal context, not global embedding alone.
- **Reference image absent:** global sweep and calibrated non-match threshold.
- **Map artwork variation:** topology/turn sequence before pixels.
- **Small validation:** freeze decisions on external/synthetic splits; use validation only for reproducibility and broad failure analysis.

## Research sources

- EvalAI challenge API: `https://eval.ai/api/challenges/challenge/2707/`
- Perception Test official repository: `https://github.com/deepmind/perception_test`
- Challenge/workshop: `https://perception-test-challenge.github.io/`
- Walking Tours dataset: `https://huggingface.co/datasets/shawshankvkt/Walking_Tours`

## Relevant methods reviewed — 2026-08-19

- **AnyLoc** (`arXiv:2308.00688`, `github.com/AnyLoc/AnyLoc`): frozen DINOv2 patch descriptors with unsupervised VLAD aggregation for universal visual place recognition. This is the next low-risk replacement for whole-image SigLIP retrieval.
- **Revisit Anything / SegVLAD** (`arXiv:2409.18049`, `github.com/AnyLoc/Revisit-Anything`): retrieves segmented local regions so viewpoint-dependent non-overlap does not dominate landmark similarity. Primary candidate for landmark presence and revisit/timestamp questions.
- **T\*** (`arXiv:2504.02259`, `github.com/mll-lab-nu/TStar`): iterative coarse-to-fine temporal search for long-video QA. Reported improvements under a 32-frame budget support replacing uniform sparse sampling.
- **VGGT-Long / Map-Long** (`github.com/DengKaiCQ/VGGT-Long`): overlapping chunks, alignment, and loop closure for kilometer-scale RGB sequences; MapAnything integration supports metric SE(3). This directly addresses the current unaligned/capped DA3 failure.
- **VGGT-SLAM 2.0** (`github.com/MIT-SPARK/VGGT-SLAM`): alternative long-sequence dense RGB SLAM with loop closure and current video support.
- **TrajLoc / SeqGeo-VL** (`arXiv:2607.15491`, `github.com/mvrl/TrajLoc`): joint video, route-text, satellite, and trajectory-geometry embeddings. Relevant to both candidate maps and textual route instructions; use as one candidate score rather than a standalone answerer.
- **CrossText2Loc** (`arXiv:2412.17007`): text/scene-description retrieval against satellite or OSM data. Relevant because Walking Tours metadata supplies city/neighborhood context and answer options expose exact street/POI names.
- **FoL / FoL++** (`arXiv:2504.09881` and follow-up): region-aware place recognition and reliability-guided reranking, reinforcing the move away from global image similarity.

### Implication for KilometerVision

The target stack is hierarchical temporal retrieval plus local-region VPR plus aligned long-sequence geometry. A larger sparse-frame VLM is not the main solution: Qwen2.5-VL-32B scored 0/5 validation maps even after removing stale geometry and labeling ordered frames.

Official DA3-Streaming was subsequently run at 1 FPS and 2 FPS on the same sealed map window. Despite using the released nested 1.4B checkpoint, dense half-overlap Sim(3) alignment, and SALAD loop detection, it found no loop constraints and failed cross-density route stability (smoothed-path disparity 0.1008; turn correlation 0.1656; endpoint/path ratio 0.1527 versus 0.4132). The important pedestrian-video failure is upstream of candidate decoding: camera-centre topology itself changes with sampling, and a moving-average body-path proxy cannot repair chunk orientation errors. Do not treat DA3-Streaming's KITTI/TUM ranking as evidence that it transfers to head-turn-heavy walking tours.

Two answer-blind follow-ups also failed. The released TrajLoc checkpoint was loaded strictly (with only its upstream `traj_aggregator` -> `video_film` key rename) and queried using two interleaved seven-frame York sequences against 81 overlapping Esri satellite mosaics. Both York queries and two Istanbul negative-control queries returned exactly tile `(32571, 21056)`; Istanbul-even scored 0.7771 versus York-even 0.7720. Its video-only representation therefore acts as a generic urban-domain score in this bounded gallery, not trustworthy localization. Official VGGT-SLAM 2.0 was then run at 1/2 FPS on the sealed 215-second York window. Optical-flow selection retained 60/84 poses from 215/430 frames, but SALAD found zero loop closures. Timestamp-matched 3-D Sim(3) comparison gave normalized disparity 0.2203, turn correlation -0.0669, and endpoint/path ratios 0.0504 versus 0.0936. Do not feed either signal into candidate selection.

The strongest available native-video audit also failed the full gate. The configured Gemini API project exposed 3.1 Pro but assigned it zero request/input-token quota; the frozen experiment therefore used `gemini-3.5-flash`. It received each exact map-question video window at 4 FPS, a labelled DPVO/candidate comparison panel, and all five original maps. Base-order accuracy was 2/5; after rotating display order from ABCDE to CDEAB it was 1/5. Stable identity agreement was 4/5, but stable-and-correct coverage was only 1/5. The useful intermediate behavior was semantic localization: responses repeatedly recovered exact streets, POIs, and route order. The failure was candidate discrimination, especially precise S/E placement among near-identical maps. A principled candidate-hidden video-extraction then map-only-selection experiment was attempted twice, but its video-only API request stalled reproducibly even with a documented 180-second SDK transport timeout; no partial result was accepted. Native Gemini output therefore remains research evidence, not a production override.

### Map-route research update (2026-08-19)

The rejected Hu-moment matcher is structurally mismatched to the problem. Global image moments are useful for coarse silhouette comparison but discard path order, signed turns, endpoint identity, branch structure, and traversal reversal. Candidate routes should instead be decoded as ordered graphs and compared to a posterior over video trajectories.

- **TrajLoc / SeqGeo-VL (ECCV 2026):** directly retrieves a matching satellite tile from sequential street video, route language, and optional waypoint/headings. Its released TrajMod encodes `(Δx, Δy, heading)` and reported video-to-satellite R@1 is 12.09%, substantially above comparable sequential baselines. Use it as an independent candidate score, not an oracle; rendered Google-map candidates are out of its training domain.
- **GASM shape-only map matching (GeoInformatica 2025):** represents trajectories and road networks symbolically and reports that shape can support map matching without absolute coordinates. The transferable lesson is to preserve ordered shape/topology rather than compare raster moments.
- **Sketch-map alignment research:** topology, ordering, qualitative location relations, and bidimensional alignment remain stable under scale/position distortion; combining order relations with geometric alignment is more robust than one global shape statistic.
- **VGGT-Long / Map-Long:** overlapping chunks, alignment, and loop closure extend feed-forward reconstruction to kilometer sequences; MapAnything-backed metric SE(3) is available. This is the expensive fallback when current motion traces, rather than candidate decoding, are the limiting factor.
- **SD-map localization work:** recent methods fuse noisy trajectory/visual evidence with map priors rather than forcing a single deterministic path. For this challenge, score all five candidates against a trajectory distribution and retain the candidate-relative margin.

Experiment result: ordered route graphs improved Hu from 1/5 to 2/5, but ORB-derived paths were unstable across 0.5/1/2 fps. Exact-window OCR also failed. Camera-only VGGT global reconstructions scored 2/5 at both 30 and 60 frames but changed predictions; overlap-aligned 16-frame chunks over 120 frames produced continuous trajectories and also scored 2/5. A fixed equal-rank consensus of the three VGGT views reached 3/5. The next gate is official long-sequence loop closure / Pi-Long comparison and cross-density stability—not validation-tuned weighting.
- DINOv2: `https://github.com/facebookresearch/dinov2`
- SigLIP2: `https://huggingface.co/google/siglip2-base-patch16-224`
- LightGlue: `https://github.com/cvg/LightGlue`
- Qwen3-VL/Omni families: `https://github.com/QwenLM/Qwen3-VL`, `https://github.com/QwenLM/Qwen3-Omni`

These sources support component selection; leaderboard claims and future-paper claims are not assumed without a reproducible artifact and a license check.

## Frontier sweep and score-target plan — 2026-08-24

The current external leader is reported as **0.6075** on the 986-question test set. That is approximately 599 correct answers, so a net gain of only one answer would clear the target; broad unverified rewrites are therefore high risk. The local repository records an older 0.57505 Gemini submission, so the 0.6075 artifact/configuration must be treated as an external run until its exact manifest, settings, and checksum are archived.

### Research findings that transfer directly

- **Temporal Chain of Thought** (`arXiv:2507.02001`) uses iterative VLM-guided frame retrieval instead of one uniform sample. It reports that a small, selected context can outperform a much larger uniform context on long videos. Apply this as a question-local retrieval pass: first select turn/landmark/map-evidence frames, then answer from the selected set.
- **NeuS-QA** (`arXiv:2509.18041`) builds a temporal-logic video automaton from frame propositions and submits only logic-verified segments to the VLM. For this challenge, the useful subset is answer-blind: `start -> turn sequence -> endpoint`, `reference appears -> nearest timestamp`, and `route begins/ends at map pins`.
- **TRACE** (`arXiv:2603.23404`, ACL 2026) induces a textual allocentric representation containing meta-context, camera trajectory, and entities before spatial QA. This is almost exactly the missing contract in the current direct Gemini lane: produce a route certificate with ordered turns, crossings, endpoint displacement, visible street/POI entities, and uncertainty, then perform candidate selection.
- **ViSRA** (`arXiv:2605.10106`) is a training-free agent that composes explicit expert spatial cues with an MLLM and reports gains on spatial-video tasks. The transferable idea is orchestration, not the headline score: feed a VLM route/geometry/OCR/track evidence cards and require it to resolve only residual semantic ambiguity.
- **DINOv3** (`arXiv:2508.10104`, Meta model card) exposes dense patch features intended for retrieval, correspondence, segmentation, and tracking without fine-tuning. It is a direct upgrade path for landmark/reference retrieval and loop-closure proposals. Use a small/base model for all frames and a larger model only for top candidates.
- **Revisit Anything / SegVLAD** (`arXiv:2409.18049`) retrieves segmented image regions rather than whole-image descriptors, specifically addressing viewpoint changes and partial overlap. This is a strong fit for storefronts, signs, statues, and facades in walking tours.
- **M-LLM frame selection** (`arXiv:2502.19680`) shows a practical two-level selector: cheap per-frame relevance plus text-level temporal selection over captions. A deterministic approximation can use OCR, novelty, turn energy, and reference-image similarity before spending a Gemini call.
- **Qwen3-VL** (`arXiv:2511.21631`, official HF models) adds explicit timestamp alignment, interleaved multimodal context, and long video support. The 235B model is too expensive for a blind full replacement here; a 32B thinking/instruct probe is worthwhile only with TRACE-style route text and strict candidate-order controls.
- **Gemini’s current video documentation** states that File API video is sampled at 1 FPS by default, low resolution costs about 66 tokens/frame, high resolution about 258 tokens/frame, and fast motion may be missed. Explicit video FPS metadata is therefore a required reproducibility field for every experiment.

### Ranked experiments to beat 0.6075

1. **Reproduce the leader contract.** Archive the exact leader settings, then rerun a bounded validation control with `fps=2`, `scale=320`, `thinking=low`, explicit video FPS metadata, and the same model. This is the highest-confidence engineering lever because the repository’s validated batch used 2 FPS while production defaulted to 0.5 FPS.
2. **TRACE-style Gemini route certificate, map-only.** First call: answer-blind route certificate. Second call: candidate comparison using the frozen certificate and all five maps. Run only on map questions; do not pay two calls for distance/landmark/compass until validation shows a benefit.
3. **Question-adaptive frame selection.** Build a cheap selector over cached frames using temporal novelty, optical-flow/turn energy, OCR/street text, reference-image similarity, and candidate-map route relevance. Give the final model the selected neighborhood plus a sparse global scaffold. This should improve brief-turn and landmark recall without increasing every clip to high FPS.
4. **Compass-only temporal ensemble.** Combine the existing motion specialist with corrected-FPS Gemini only when the two route certificates agree on sustained turns; abstain to the existing motion lane on disagreement. The validation evidence supports Gemini as a partial compass specialist, not a universal replacement.
5. **DINOv3/SegVLAD landmark lane.** Replace the current thumbnail/color fallback for the 112 landmark and 297 distance questions. Retrieve top temporal neighborhoods, then verify with local patch geometry and OCR. Merge only landmark/distance changes and preserve answer-blind evidence.
6. **Candidate-symmetric map adjudication.** Randomize map display order, run route-certificate comparison twice with different candidate permutations, and accept only stable identity. This addresses the known Gemini positional bias; a single changed answer is not evidence of a gain.
7. **Only then test new spatial VLMs.** Probe SpatialAxiom-35B, Qwen3-VL-32B/235B, or InternVL3.5 with the structured evidence contract. Do not promote raw video-only outputs: the existing local Qwen and SpatialAxiom map probes already show that generic map selection is the wrong interface.

### Promotion gates

No test answer labels may be used. Promote a lane only if it improves a relevant validation family, is permutation-stable for map choices, has complete answer coverage, and changes no unrelated family. Given the 0.6075 target, the practical objective is a **single net correct answer**, but the candidate must still pass the validation and schema gates before submission.
