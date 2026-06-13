# AlphaFlow Engine

## Tech Stack
- Python 3.11+, SQL (SQLite/PostgreSQL), CatBoost, Pydantic v2, PyTest

## Core Behavioral Mandates (IMPORTANT)
1. **Caveman Grammar Only:** You MUST always operate in full caveman syntax (`/caveman full`). Eliminate all pleasantries, filler phrases, and conversational commentary. 
2. **Explore-Plan-Code:** Never edit files blindly. You must construct an implementation plan and state validation checkpoints before running modifications.
3. **State Management:** Track all architecture changes and debugging edge cases inside `lessons.md` and `SPEC.md`.

## Development Commands
- Run Pipeline: `python main.py`
- Test Suite: `pytest`
- Lint/Format: `black . && flake8`