# PATHFINDER experiment plan

## Objective
Improve spatial-question top-1 accuracy and produce reproducible, schema-safe submissions without using held-out answer labels.

## Execution lanes

1. **Contract safety:** unit tests, metadata audits, answer/answer_id consistency, exact archive/schema checks.
2. **Visual retrieval:** bounded color/edge descriptors first; optional semantic embeddings and local geometric verification on Modal.
3. **Route geometry:** DA3 relative pose, calibrated only when evidence supports it; no invented absolute heading or meters.
4. **Specialists:** compass, distance, landmark, and map decisions use question-local evidence and preserve uncertainty.
5. **Submission:** regenerate validation/test artifacts, audit completeness, checksum outputs, and record EvalAI results.

## Boundaries

- Never copy challenge videos into the repository.
- Never train, prompt, tune, or manually label from validation/test answers.
- Do not treat relative monocular geometry as absolute north or metric distance.
- Do not claim a score improvement without a fresh reproducible submission record.

## Research-backed incremental experiments

Promote a lane only when it improves the relevant validation family without changing unrelated answers. Keep failed runs under `experiments/`; production artifacts remain split-namespaced.

1. **AnyLoc baseline:** replace whole-image SigLIP retrieval with frozen DINOv2 patch descriptors and VLAD-style aggregation. Compare maximum similarity, temporal coherence, and retrieval timestamp on the three validation reference-image questions.
2. **SegVLAD reranking:** if DINOv2 has useful recall, segment shortlisted reference/frame pairs and aggregate only overlapping local regions. Verify the top candidates geometrically before deciding presence.
3. **T* temporal search:** replace uniform six/twelve-frame sampling with coarse-to-fine temporal search. Use reference similarity for landmarks/revisits and OCR/street-name relevance for route questions.
4. **Long-sequence geometry:** replace the obsolete 60-frame DA3 artifacts with VGGT-Long or Map-Long over overlapping exact-window chunks, alignment, and loop closure. Prefer MapAnything metric SE(3); otherwise retain explicit Sim(3) scale uncertainty.
5. **Map localization:** evaluate TrajLoc video/text/geometry embeddings against the five candidates, then fuse with video metadata, dense OCR, candidate POIs/street names, and recovered trajectory evidence.
6. **Metric answers:** derive endpoint distance only after localization or metric trajectory recovery. Walking-time remains a weak bound, never the primary straight-line estimate.

### Gates

- Validation labels are used only for final evaluation and failure analysis, never training or parameter fitting.
- Do not promote a threshold inferred from the three landmark examples; use pretrained defaults, external calibration data, or label-independent score statistics.
- Record per-family coverage, changed answers, confidence, runtime, model/checkpoint, and artifact hash.
- Do not submit an experiment that merely shifts aggregate validation accuracy through unrelated fallback changes.

## Current next experiment

1. **Completed:** regenerated compass motion under facet-namespaced paths, reproduced 146/146 preserved choices, and submitted the strict one-family ablation. Submission 579590 is the new private-test best at 0.2505071.
2. Preserve submission 579590 / SHA-256 `4adccee09cd10102bc4b754b9809efb3dacc0ea7f29290fb3fc682ff1ef7c3c6` as the production answer baseline; future artifacts must report family-isolated diffs against it.
3. Do not use DPVO winner margin or same-trace resampling stability as a production gate; the validation false positive is confident and stable. Evaluate candidate-map scoring on an eligible external/synthetic route-to-map benchmark instead.
4. **Completed reconnaissance:** public video metadata covers 54/54 test videos and yields route anchors for 52/54; candidate-map OCR covers all 284 image-map questions. Exact/fuzzy anchor overlap uniquely separates only 12/284 choices, so keyword-only map replacement is rejected.
5. **Completed/rejected as raw geometry:** multi-anchor geocoding recovered coherent route corridors, but official OSMLoc positions were highly sensitive to unknown focal length. Retain metadata/OSM as semantic candidate evidence only, not as a metric trajectory.
6. **Completed/rejected as a VLM selector:** Qwen3-VL-32B jointly consumed motion-aware frames, relative-depth views, DPVO trajectory, and normalized S→E maps, but scored 1/5 validation maps and changed 2/5 predictions when only panel placement changed. Retain the evidence-card renderer for inspection; do not promote its answers.
7. **Completed/rejected:** official LoGeR failed the label-free 30-vs-60-frame route-stability gate on York. Do not score or tune a candidate-map override from this trace.
8. **Completed/rejected:** official DA3-Streaming with loop detection failed the label-free 1-vs-2-FPS route-stability gate on York (pedestrian-path disparity 0.1008, turn correlation 0.1656, endpoint ratio 0.1527 vs 0.4132); neither density produced a loop constraint. Do not inspect candidate correctness or expand this configuration.
9. Next evaluate semantic candidate evidence that does not require camera intrinsics: candidate-map POI/street extraction against jointly geocoded metadata anchors, with abstention unless a unique choice has multiple independent matches. Continue the eligible train-only compass/landmark specialist lane independently; keep KilometerVision validation/test media sealed from fitting. Do not upload experimental artifacts automatically.

## Map recovery plan

1. **Completed:** ordered skeleton/route decoding, multi-scale curvature DTW, Procrustes alignment, reversal/reflection enumeration, and S/E marker decoding. Hu 1/5; best single ordered lane 2/5.
2. **Completed/rejected:** ORB map trajectories at 0.5/1/2 fps are unstable (1/5, 1/5, 0/5). Keep ORB only for its separately verified compass use.
3. **Completed/rejected:** candidate/video OCR provides no reliable localization signal on the five reporting maps.
4. **Promoted leader:** native-continuity DPVO at 1 fps / 16 patches / 256 px with reflection-free SVD-axis ambiguity scores 4/5 validation maps. Test coverage is 431/431; H100 handles ordinary windows and H200 handles 285–292-frame tails.
5. **Ready artifact:** the DPVO submission changes 222/431 map choices versus the compass-only artifact, with no unrelated-family changes. The blind EvalAI result is the next promotion gate.
6. If needed after the blind gate, add loop closure or an external-data-calibrated confidence fusion with VGGT; do not select a fusion rule from the five validation labels.
