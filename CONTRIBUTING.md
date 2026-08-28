# Contributing

Issues and pull requests that improve reproducibility, correctness, documentation, test coverage,
or compatibility are welcome after the repository owner has selected a software licence.

For a local development setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ruff check src tests
```

Please keep changes focused, add tests for changed behaviour, and state whether a modification is
intended to preserve the published experiment or introduce a method change. Do not commit datasets,
logs, local paths, virtual environments, or credentials.

