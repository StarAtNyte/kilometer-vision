# PATHFINDER · KilometerVision

Spatial reasoning over walking-tour videos: compass orientation, distance, route maps, and landmarks.

Question planning → cached frames and geometry → route and landmark specialists → answer selection and submission audit.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --extra dev
uv run pytest
uv run modal run -m kilometer_vision.jobs.modal_pipeline --help
```

The local test suite uses synthetic inputs. Model inference runs in Modal;
GPU frameworks and model-specific dependencies are defined in each job's image,
so local setup does not download model weights.

## Running the pipeline

Authenticate with Modal and configure the profile and any model/API secrets required
by the selected job. The commands below launch remote work and may incur cloud costs.

```bash
uv run modal run -m kilometer_vision.jobs.modal_pipeline --action bootstrap
uv run modal run -m kilometer_vision.jobs.modal_pipeline --action audit --split validation
```

Use module paths for every job: `uv run modal run -m kilometer_vision.jobs.<module>`.
Local tools also run as modules:

```bash
uv run python -m kilometer_vision.tools.validation_diagnostic --help
```

## Code layout

| Path | Purpose |
| --- | --- |
| [`src/kilometer_vision/`](src/kilometer_vision/) | Shared components: `question_planner.py`, geometry backends, route/map/motion solvers, landmark matching, and `submission.py` |
| [`src/kilometer_vision/jobs/`](src/kilometer_vision/jobs/) | Modal data jobs, inference pipelines, and experiments |
| [`src/kilometer_vision/tools/`](src/kilometer_vision/tools/) | Local evaluation and operational utilities |
| [`tests/`](tests/) | Synthetic regression tests |
| [`docs/`](docs/README.md) | Research, architecture, experiment history, and recorded results |

The previous root-level Python modules now live under `src/kilometer_vision/`.
Imports use the package namespace, and Modal images include that package for remote execution.

## Data and evaluation

Persistent storage: `perception-test-2026-data`, with track data under `/data/2707-kilometer-vision` and shared videos under `/data/shared/walking-tours`. Keep raw media, extracted frames, model weights,
credentials, and new prediction artifacts out of Git. Existing checked-in experiment
records are retained for traceability.

Read [`AGENTS.md`](AGENTS.md) for challenge-specific constraints. Preserve organizer
IDs, option order or track identities, and the submission schema. Validation and test
examples must not become training or few-shot prompt data. Grounded outputs contain
box tracks; multiple-choice outputs contain the selected option.

## Development and project records

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for verification and contribution guidance.
Build a distributable package with `uv build`. CI runs the synthetic test suite and
package build without cloud credentials.

Research notes and score tables in [`docs/`](docs/README.md) are dated experiment
records; they do not represent a freshly verified leaderboard. See the
[official challenge](https://eval.ai/web/challenges/challenge-page/2707/overview)
for evaluation details.
