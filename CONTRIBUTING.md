# Contributing

## Setup and checks

Use Python 3.11+ and run these commands from the repository root:

```bash
uv sync --frozen --extra dev
uv run pytest
uv build
```

## Code organization

Put reusable logic in `src/kilometer_vision/`, Modal entrypoints in its `jobs/`
subpackage, and local analysis or validation utilities in `tools/`. Use absolute
package imports. When changing remote imports, include the package in the relevant
Modal image with `add_local_python_source("kilometer_vision")`.

Update the README and affected commands when changing entrypoints. Record experiment
methods, model versions, and measured results in `docs/`; distinguish observations
from proposed work. Keep changes focused and preserve existing submission schemas.

## Data and credentials

Follow the challenge rules in [AGENTS.md](AGENTS.md). Use synthetic fixtures for
local tests. Keep dataset media, model checkpoints, credentials, and new generated
predictions outside version control. Do not use validation or test examples as
training or few-shot prompt data. Remote inference and organizer submissions are
separate from local checks and require the appropriate account configuration.
