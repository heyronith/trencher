#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv sync --all-extras
uv run ruff check .
uv run pytest
uv run trencher --version
uv run trencher dune doctor
