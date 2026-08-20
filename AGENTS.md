# Challenge 2707 — KilometerVision

Read this file before making changes in this directory.

## Official links

- EvalAI: https://eval.ai/web/challenges/challenge-page/2707/overview
- Workshop: https://perception-test-challenge.github.io/
- Walking Tours dataset: https://huggingface.co/datasets/shawshankvkt/Walking_Tours

## Schedule

- Validation phase: June 5–August 27, 2026, 23:59:59 UTC.
- Test phase: July 15–August 27, 2026, 23:59:59 UTC.
- Technical report deadline: August 31, 2026, 23:59:59 UTC.
- Winner announcement: September 3, 2026.

## Task

Answer five-choice spatial-reasoning questions about walking-tour videos up to approximately 10 minutes long and generally covering about one kilometer.

Question families include:

- Distance estimation between locations or moments.
- Compass orientation after accumulated camera/person turns.
- Map understanding and selecting the route map matching the video.
- Landmark recognition and determining whether a reference image appeared in the route.

Supported formats include:

1. Text question with five text answers.
2. Text question with five image answers, such as candidate maps.
3. Text-and-image question with five text answers.

## Data

| Split | Videos | Questions | Additional assets |
| --- | ---: | ---: | --- |
| Validation | 10 | 18 | Questions JSON, video CSV, images/maps archive |
| Test | Organizer release | Organizer release | Check challenge page |

The validation set is too small to support robust tuning. Use it to verify parsing, media alignment, output formatting, and broad failure modes—not to fit task-specific parameters.

## Evaluation and submission

- Primary metric: top-1 accuracy.
- Return exactly one of five choices for every question.
- Correctly package reference images and candidate-map images with their questions.
- Preserve exact IDs, option ordering, schema, and archive layout.
- Record internal evidence, estimated route state, and confidence for analysis.

## Recommended implementation

Use a structured route representation rather than relying only on a single VLM context:

1. Decode timestamped keyframes and detect scene/turn boundaries.
2. Extract landmark embeddings and searchable visual descriptors.
3. Estimate relative camera motion, turns, and approximate travel distance.
4. Build a trajectory graph with timestamps, landmarks, headings, and uncertainty.
5. Route each question to a specialist solver: compass, map, landmark, or distance.
6. Compare candidate answers against the structured route and use a VLM for semantic reasoning where helpful.

For landmark questions, retrieve the closest keyframes to the supplied image and verify geometrically or semantically. For compass questions, accumulate turns with uncertainty. For map questions, normalize scale and rotation before matching trajectory shape. For distance questions, combine visual odometry or map cues with calibrated uncertainty rather than trusting monocular depth alone.

Modal A100 80GB access is confirmed. VRAM is sufficient; the main risks are accumulated motion error, ambiguous scale, long-video preprocessing, and weak spatial consistency. Cache keyframes and route features so experiments do not repeatedly decode videos or consume unnecessary GPU credits.

## Rules

- Do not manually label test videos.
- Process each test video/question pair independently; do not exploit correlations among test questions.
- Do not train, self-supervise, or prompt on validation or test data.
- Eligible external datasets and pretrained models may be used.
- Public VLM APIs are permitted; log their exact versions and never commit credentials.
- A public test submission and technical report are required for prize eligibility.

