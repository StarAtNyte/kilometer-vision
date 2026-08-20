# DA3 geometry upgrade run — 2026-08-18

- Backend: `depth-anything/DA3-LARGE`
- Execution: Modal `kilometervision-geometry`, A100-80GB
- Input: all required validation/test frame manifests plus shared inventory; after reconciliation this is 136 manifests, including all 54 test IDs.
- Output: `/data/2707-kilometer-vision/geometry/<video_id>.da3.json` and canonical `<video_id>.json` contracts; 136 video artifacts after adding the 30 previously absent test IDs.
- Per-video output: 60 timestamped frame records containing DA3 relative extrinsics, intrinsics, depth summaries, and confidence when provided.
- Contract materialization: `modal_pipeline.py --action build-geometry` copied DA3 artifacts to canonical `<video_id>.json` paths and recorded relative scale explicitly.
- Prediction artifacts regenerated after geometry context integration:
  - `/data/2707-kilometer-vision/predictions/validation_submission.json` — 18/18 audit coverage
  - `/data/2707-kilometer-vision/predictions/test_submission.json` — 986/986 audit coverage
- Metadata audits: validation 13 videos / 18 questions; test 54 videos / 986 questions; both valid with no warnings.
- Validation submission: EvalAI **579377**, private, Top-1 **0.4444444444** (18/18 coverage), preserving the previous best validation gate.
- Baseline private test submission: EvalAI **579376**, Top-1 **0.2434077079** (240/986).
- Geometry-integrated private test submission: EvalAI **579417**, Top-1 **0.2454361055**, finished 2026-08-18 01:12:40 UTC.
- Geometry-integrated validation submission: EvalAI **579418**, private, Top-1 **0.4444444444**.
- Current prediction artifact SHA-256 checksums: validation `8d0f3651ce213d8e188bee0b17d7f40d564dfbbbb6a456816d429e0bf1941a2c`; geometry-integrated test `e45f2536ea9890eb08fa02bda6145abb309be5a515bb3469da356f17cab6989d`.

## Leaderboard comparison

- Geometry-integrated private test score: **0.2454361055**.
- Baseline private test score: **0.2434077079**.
- Improvement: **+0.0020283976** absolute.
- Public leaderboard reference leader: approximately **0.4544** Top-1.
- Gap to leader: approximately **0.2089638945** absolute (**20.9 percentage points**).
- Current score is approximately **54.0%** of the leader score.
- Exact rank is unavailable because submission **579417** was private and therefore does not appear on the public leaderboard.

## Limitations

DA3-LARGE pose coordinates are relative and each bounded inference window has its own reference frame. The scorer therefore does not claim absolute north or metric meters; geometry is exposed as timestamped retrieval/evidence context. Metric distance and absolute heading require a separately calibrated fusion lane.

No validation/test answer labels were used for training, tuning, prompting, or manual labeling. The test submission contains only required prediction fields.

## Follow-up experiment — relative-yaw compass lane

The submission layer now extracts a bounded relative yaw delta from DA3 extrinsic rotations and uses it only when the question supplies an initial compass anchor (for example, “starting facing north”). The result is a weak posterior over the endpoint direction; neighboring directions absorb residual uncertainty. Options are normalized from phrases such as `the east`, rather than requiring exact option strings. Questions without an explicit anchor continue through the prior fallback path, so unknown geometry cannot invent north.

Validation: 28 dependency-light unit tests pass, including a synthetic 90-degree pose rotation (`north` -> `east`). This is an offline contract test, not a leaderboard claim. Before any submission, run the validation/test Modal prediction actions and record the resulting EvalAI score separately; DA3 pose-axis convention and per-window frame alignment remain risks.

## Follow-up experiment — color/edge retrieval and landmark presence

Replaced the grayscale-only thumbnail descriptor with a bounded RGB-channel-mean, centered-color, and edge descriptor. This preserves scene color cues while retaining viewpoint-tolerant low-resolution structure. Landmark questions with an available reference match now select semantic presence/absence options when the match confidence clears the explicit threshold; opaque options still use the prior fallback.

Validation: 30 dependency-light unit tests pass, including same-luminance red/blue separation. This remains a local retrieval experiment; rerun `build-visual-index` and `predict` on Modal before comparing leaderboard scores.
