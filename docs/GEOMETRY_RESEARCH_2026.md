# Geometry research for KilometerVision (verified through August 2026)

## What “geometry” means here

Geometry is not “make a 3-D model because it looks impressive.” For this benchmark it means recovering the **relative spatial state of the camera over time**:

```text
frame i -> camera pose (R_i, t_i), depth/points, uncertainty
sequence -> trajectory graph, turns, loop closures, heading changes
reference image -> place/time correspondence
candidate map -> route graph / polyline correspondence
```

The important distinction is **metric geometry** versus **up-to-scale geometry**. A monocular video can often estimate shape and relative motion but not absolute meters. Therefore:

- Compass needs relative yaw integrated from a stated initial heading.
- Map questions mainly need topology and turn sequence; scale is secondary.
- Landmark questions need place correspondence and temporal position.
- Distance questions need scale or a calibrated walking-speed/scene prior.

A useful pose is `T_i = [R_i, t_i]`, where `R_i` is camera orientation and `t_i` is camera position. A 3-D point from pixel `(u,v)` and depth `d` is `X_i = d K^-1 [u,v,1]`; transform it to a common route frame with `X_world = T_i X_i`. The final system must carry confidence and failure modes, not silently convert uncertain depth into meters.

## Strong model families and what they contribute

### 1. VGGT and VGGT-Omega — global multi-view geometry

- VGGT, CVPR 2025: [paper](https://arxiv.org/abs/2503.11651), [GitHub](https://github.com/facebookresearch/vggt), [HF](https://huggingface.co/facebook/VGGT-1B).
- VGGT predicts cameras, depth, point maps, and tracks from one/few/many views in one feed-forward pass. It is a strong baseline for short windows and global consistency.
- VGGT-Omega, 2026: [paper](https://arxiv.org/abs/2605.15195), [GitHub](https://github.com/facebookresearch/vggt-omega), [HF](https://huggingface.co/facebook/VGGT-Omega).
- Omega exposes camera/depth predictions, handles substantially more frames per A100, and has a text-alignment checkpoint. Use it for 25–200 frame windows, not the entire test inventory in one context.
- License/checkpoint access must be checked before competition use; the original VGGT and commercial checkpoint have different terms.

### 2. Depth Anything 3 / DA3-Streaming — long-video geometry

- [paper](https://arxiv.org/abs/2511.10647), [GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3), [HF models](https://huggingface.co/depth-anything), [streaming code](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/main/da3_streaming).
- DA3 uses a depth-ray representation and supports monocular depth, multi-view depth, pose estimation, pose-conditioned depth, and metric/nested variants.
- DA3-Streaming is especially relevant: sliding-window inference for ultra-long video with low memory, persistent state, reference-view selection, and loop utilities.
- Best role: generate dense local geometry and candidate camera poses over sequential windows; do not trust its raw monocular scale without calibration.

### 3. MapAnything — unified metric reconstruction and model adapter

- [paper](https://arxiv.org/abs/2509.13414), [GitHub](https://github.com/facebookresearch/map-anything), [HF](https://huggingface.co/facebook/map-anything).
- Apache model option: `facebook/map-anything-apache`.
- Supports image-only and multi-modal metric reconstruction and exposes a unified interface for VGGT, VGGT-Omega, DUSt3R, MASt3R, MUSt3R, Pi3-X, and DA3.
- This is valuable engineering-wise: PATHFINDER can benchmark several geometry backends using one output contract instead of coupling the whole system to Qwen or one model.
- Use MapAnything’s confidence masks and metric outputs for distance; use pose/point outputs for route graph construction.

### 4. DUSt3R / MASt3R / MUSt3R / CUT3R / Fast3R

- DUSt3R: [GitHub](https://github.com/naver/dust3r), [paper](https://arxiv.org/abs/2312.14132).
- MASt3R: [GitHub](https://github.com/naver/mast3r), [paper](https://arxiv.org/abs/2406.09756).
- CUT3R, CVPR 2025: [GitHub](https://github.com/CUT3R/CUT3R), [paper](https://arxiv.org/abs/2501.12387). It maintains persistent state across frames and is a good sequential baseline.
- Fast3R, CVPR 2025: [GitHub](https://github.com/facebookresearch/fast3r), [paper](https://arxiv.org/abs/2501.13938). It targets 1000+ images in one forward pass, useful for sparse route windows.
- These models output relative 3-D/poses/point maps. They are useful as independent geometric votes and for cases where one global model drifts, but memory and license/checkpoint differences matter.

### 5. Metric-scale hybrid: VIDAR-style design

- VIDAR (2026): [arXiv](https://arxiv.org/abs/2607.17171).
- Its core lesson is important for walking tours: pair a motion/odometry backbone with a dense foundation model. Odometry supplies frame/scale; the foundation model supplies local dense geometry.
- KilometerVision has no IMU, so use a visual-only approximation: long-baseline feature tracks + robust pose estimation + learned depth/pose, with a walking-speed prior only as a weak scale factor. Never present learned depth scale as ground truth.

## Place recognition and landmark retrieval

### DINOv3 and dense descriptors

- DINOv3, 2025/2026 releases: [paper](https://arxiv.org/abs/2508.10104), [GitHub](https://github.com/facebookresearch/dinov3), [HF docs](https://huggingface.co/docs/transformers/model_doc/dinov3).
- DINOv3 provides strong global and dense patch features. It is preferable to our current thumbnail/color sketches for scene and landmark retrieval.
- Practical model ladder: DINOv3 ViT-S/B for cost, ViT-L for final candidate re-ranking; use patch descriptors for local sign/building matching rather than only CLS/global vectors.

### AnyLoc / MegaLoc / GeM/SALAD family

- AnyLoc: [paper](https://arxiv.org/abs/2308.00688), [GitHub](https://github.com/AnyLoc/AnyLoc). DINO features plus VLAD aggregation remain a strong training-free VPR baseline.
- MegaLoc: [paper](https://arxiv.org/abs/2502.17237), [GitHub](https://github.com/gmberton/MegaLoc), [HF](https://huggingface.co/gberton/MegaLoc). It is designed for broad place retrieval and reports strong results across VPR datasets.
- SALAD: [GitHub](https://github.com/serizba/salad). Use as an alternative global descriptor/aggregation lane.
- EfficientVPR, CVPR 2026: [paper](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_EfficientVPR_Toward_Efficient_Visual_Place_Recognition_via_Scene-Aware_Prompt_Tuning_CVPR_2026_paper.html). Relevant if DINO-scale inference is too expensive.
- Trajectory-aware cross-view geo-localization (ECCV 2026): [paper](https://arxiv.org/abs/2607.15491), [GitHub](https://github.com/mvrl/TrajLoc). The key transferable idea is to retrieve from a **sequence**, not independent frames: route order and repeated observations disambiguate visually similar streets.

## Candidate-map matching: the right abstraction

Do not compare candidate map PNGs to video frames with a VLM and ask “which looks similar.” Convert each candidate map to a route signature:

```text
map_signature = {polyline, endpoints, turn_angles, segment_lengths, loop_count,
                 branch_count, orientation, image-space scale/rotation}
```

Convert the video reconstruction to the same signature, allowing an unknown similarity transform. Score:

```text
map_score = topology_match
          + turn_sequence_match
          + normalized_curvature_match
          + endpoint/displacement_match
          + orientation_match
          - uncertainty_penalty
```

Topology should dominate pixels. A visually different map drawing can still be the correct route; a visually similar drawing can have the wrong turn sequence.

## Recommended PATHFINDER-Ω architecture

1. **Question-independent lattice:** 1 FPS frames, 0.25 FPS long-range anchors, OCR, DINOv3/MegaLoc descriptors, raw media refs.
2. **Geometry windows:** DA3-Streaming or CUT3R for sequential windows; VGGT-Omega/MapAnything for 25–100-frame anchor windows.
3. **Pose fusion:** robustly align overlapping windows with SE(3)/Sim(3), reject low-confidence matches, add loop-closure edges from DINOv3/MegaLoc retrieval.
4. **Uncertainty route graph:** preserve heading posterior, turn candidates, displacement intervals, and scale confidence.
5. **Reference-image matcher:** global DINOv3/MegaLoc retrieval -> local patch matching -> temporal neighborhood verification.
6. **Map compiler:** image segmentation/vectorization -> route signature -> all-five candidate graph matching.
7. **Specialists:** compass, landmark, distance, and map solvers score atomic option predicates; a VLM only resolves semantic ambiguity from cited evidence cards.
8. **Audit:** question-type-specific certificate. Counts/absence require global coverage; local landmark matches can use a bounded certificate.

## Priority order and expected payoff

| Priority | Change | Why |
|---|---|---|
| P0 | Replace thumbnail descriptors with DINOv3-S/B or MegaLoc | Current landmark retrieval is weak and non-semantic |
| P0 | Add DA3-Streaming or CUT3R on short sequential windows | Current route solver has novelty only, no pose |
| P1 | Add VGGT-Omega/MapAnything anchor windows and Sim(3) alignment | Reduces accumulated drift and estimates trajectory shape |
| P1 | Implement map vectorization/topology matching | Map questions are currently essentially guessed |
| P1 | Use sequence-level place retrieval/loop closure | Prevents repeated-street and viewpoint confusion |
| P2 | Train a route-signature reranker on eligible external/train data | Learns uncertainty/calibration without challenge leakage |
| P2 | Add VLM evidence-card adjudication | Handles wording, landmark semantics, and ambiguous options after geometry |

## What not to do

- Do not send the whole 10-minute video to one VLM and trust its route narrative.
- Do not average monocular depth into meters without a scale model.
- Do not use validation answers to tune the map/geometry pipeline.
- Do not use a single global embedding as geometric proof.
- Do not compare map artwork pixel-wise as the primary decision.

## Implementation status

`src/kilometer_vision/jobs/modal_geometry.py` is now a working optional DA3 lane. On 2026-08-18 it ran `depth-anything/DA3-LARGE` over 60 cached frames on an A100-80GB and persisted camera/depth summaries in the Volume. The metric checkpoint is intentionally not used for pose because its API returned no extrinsics; use DA3-LARGE for poses and a separate calibrated metric model only after explicit scale validation.

## Bottom line

For KilometerVision, geometry is the **state estimator and proof layer**: camera motion, relative heading, depth/point correspondence, route topology, and uncertainty. The best non-copying direction is a model-agnostic geometry ensemble: DA3-Streaming/CUT3R for continuity, VGGT-Omega/MapAnything for global anchor windows, DINOv3/MegaLoc for loop closure/place retrieval, and a specialized graph matcher for the five map choices. Qwen—or another VLM—should only read the resulting evidence cards.
