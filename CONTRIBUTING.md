# Contributing

Thanks for considering a contribution to **chest-xray-classifier**. This
document is the short, opinionated version of the workflow — the CI and
pre-commit hooks enforce the rest.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) 0.4 or newer (we use it to manage the
  virtualenv and lockfile — `pip install`ing things by hand will drift the
  environment away from what CI runs).
- Python **3.13** (the value in `.python-version`; CI also runs the suite on
  3.12 via the matrix).
- [`pre-commit`](https://pre-commit.com/) (installed as a dev dependency, so
  `uv sync --all-groups` already ships it).

## Dev setup

```bash
# 1. Install everything, including dev + docs groups.
uv sync --all-groups

# 2. Wire up the git hooks (ruff/mypy/deptry/bandit/interrogate/codespell).
uv run pre-commit install

# 3. Run the full local gate before opening a PR.
uv run pre-commit run --all-files
uv run pytest
```

## Branch naming

Branches are short, lower-kebab-case, and prefixed with the type of change:

- `feat/<short-slug>` — new capability.
- `fix/<short-slug>` — bug fix.
- `docs/<short-slug>` — documentation-only change.
- `chore/<short-slug>` — tooling, CI, or refactor with no behavioural change.

## Commit style

We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
(`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `build:`, `ci:`).
The subject is imperative mood, under 72 characters, and does **not** end with
a period.

Do **not** append a `Co-Authored-By:` trailer for AI assistants (Claude, Copilot,
etc.). AI-assisted changes are still authored by the human on record.

## Pull-request process

1. CI must be green — the `lint-test` matrix (Python 3.12 + 3.13), the
   `docker-build` job, and `actionlint` all need to pass.
2. Keep the PR focused — one logical change per PR, with a one-line summary
   and a short *What / Why* body.
3. If you changed behaviour, update the [`CHANGELOG.md`](CHANGELOG.md) entry
   under `## [Unreleased]`.
4. Reviewer defaults to `@kiselyovd`.
5. Squash-merge only; the merged message should stay Conventional Commits
   compliant.

## Code style

- Formatting and linting: `ruff format` + `ruff check` (configured in
  `pyproject.toml`, enforced in CI and pre-commit).
- Type checking: `mypy src scripts` — all code under `src/` must type-check.
- Docstrings: `interrogate` (configured in `pyproject.toml`) enforces a
  minimum docstring coverage on public code.
- Security: `bandit -r src scripts -c pyproject.toml` must pass.
- Unused deps: `deptry .` must pass.

Running `uv run pre-commit run --all-files` applies all of the above in one
step; the CI job is deliberately the same command stack so a clean local run
is a strong signal that CI will pass too.
