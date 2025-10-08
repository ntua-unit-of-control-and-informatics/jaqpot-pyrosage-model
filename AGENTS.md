# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: FastAPI app entrypoint exposing `/health` and `/infer`.
- `src/model.py`: Pyrosage service (loading AttentiveFP models, inference).
- `src/loggers/`: JSON logger and request middleware.
- `models/`: Model weights grouped by `classification/` and `regression/`.
  - Naming: `*_attentivefp_best.pt` (auto‑discovered).
- `deploy_pyrosage_model.py`: Helper for Jaqpot deployment.
- `requirements.txt`, `Dockerfile`, `README.md`, `test_local.py`.

## Build, Test, and Development Commands
- Local setup:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Run API locally:
  - `python main.py` (serves on `http://0.0.0.0:8000`).
- Docker build/run:
  - `docker build -t jaqpot-pyrosage-model .`
  - `docker run -p 8000:8000 jaqpot-pyrosage-model`
- Smoke tests:
  - `curl http://localhost:8000/health`
  - `python test_local.py` (hits `/infer` with sample payloads).

## Coding Style & Naming Conventions
- Python 3.10+, 4‑space indentation, PEP 8.
- Use type hints (see `src/model.py`).
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Logging: prefer `src.loggers.logger` over `print`; attach context via `extra={"req":..., "res":...}`.
- Model files: keep under `models/{classification,regression}` using the `*_attentivefp_best.pt` pattern.

## Testing Guidelines
- Current tests are script‑based (`test_local.py`) and curl checks.
- When adding tests, prefer `pytest` under `tests/` with `test_*.py` names; include minimal fixtures and request examples.
- Target: keep new logic covered; validate error paths (invalid SMILES, unknown model, no bonds).

## Commit & Pull Request Guidelines
- Use Conventional Commits (observed): `feat: ...`, `fix: ...`.
- PRs should include:
  - Clear description, motivation, and scope.
  - Steps to run locally (commands and example requests).
  - Any model/endpoint changes and README updates.
  - Screenshots or sample JSON for `/infer` responses when applicable.

## Security & Configuration Tips
- Do not commit secrets. Model weights are public artifacts here; large new weights should follow the existing structure and naming.
- Keep API surface stable (`/health`, `/infer`); document any changes in `README.md` and this file as needed.
