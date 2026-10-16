# Conventions

## Code

- Formatting and lint: ruff, line length 120 (config in `pyproject.toml`). Run
  `uv run ruff format .` before committing.
- `from __future__ import annotations` at the top of every module.
- Every module starts with a docstring saying what it is for and, if useful, what does
  *not* belong in it.
- Frozen dataclasses for value objects (`Pen`, `Style`, `Page`, `Elements`). Derive
  variants with `dataclasses.replace`, don't mutate.
- Points are numpy arrays shaped `(N, 2)`. Validate at the public boundary
  (`scene._polyline`, `scene._xy`), trust them inside.
- Angles are radians in code; degrees only in user-facing labels.
- Comments explain *why*, not what. No commented-out code.
- Private helpers start with `_`. Only names in `__all__` are public.

## Tests

- pytest, one test module per source module (`tests/test_<module>.py`).
- Pure geometry and layout get numeric tests. Anything visual gets an image test with
  `@pytest.mark.mpl_image_compare` and a baseline in `tests/baseline/`.
- Example scripts in `examples/` double as image-test fixtures; keep them small.
- Random things (stipple) are seeded so images are reproducible.

## Git and PRs

- Branch per task: `<area>/<JIRA-KEY>-<short-name>` (e.g. `render/KAN-12-hatch-angle`).
- Commit messages: imperative summary line under ~70 chars, then a short why.
- PR title starts with the Jira key; body: what changed, and a rendered image when the look changes.
- Keep PRs small. A PR that touches more than one area should say why.
