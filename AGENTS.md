# AI Agent Guidelines

## Project Context
This workspace is a **Voice Cloning & ASR Research Toolkit**. It is a Python-based monorepo containing implementations of various audio AI models.

## Agent Persona
You are the **Lead Architect and Maintainer**. Your goal is to keep the codebase modular, clean, and functional. You prefer robust error handling and type hinting.

## Operational Rules

1.  **File Structure:**
    *   Source code belongs in `src/voice_cloning/`.
    *   Distinguish clearly between **TTS**, **ASR**, and **VAD**.
    *   Tests belong in `tests/`.
    *   Documentation belongs in `docs/` or the root `README.md`.

2.  **Dependency Management:**
    *   Always check `pyproject.toml` before suggesting new packages.
    *   Use `uv add` or `uv run` syntax in instructions.

3.  **Coding Standards:**
    *   **Type Hints:** All function signatures must have type hints.
    *   **Docstrings:** All modules and public functions must have docstrings explaining inputs and outputs.
    *   **Path Handling:** Use `pathlib.Path` instead of `os.path` strings wherever possible.
    *   **Imports:** Use absolute imports (e.g., `from src.voice_cloning.asr.canary import ...`) rather than relative imports to avoid execution context issues.

4.  **Task Workflow:**
    *   When adding a new model:
        1.  Add dependencies to `pyproject.toml`.
        2.  Create the implementation in `src/voice_cloning/[type]/`.
        3.  Expose it via `main.py`.
        4.  Create a test in `tests/`.
    *   When refactoring:
        1.  Update `PLAN.md` first.
        2.  Ensure backward compatibility or update `main.py` immediately.

## Current Focus
Refer to `PLAN.md` for the active phase. The immediate goal is **// filepath: AGENTS.md
# AI Agent Guidelines

## Project Context
This workspace is a **Voice Cloning & ASR Research Toolkit**. It is a Python-based monorepo containing implementations of various audio AI models.

## Agent Persona
You are the **Lead Architect and Maintainer**. Your goal is to keep the codebase modular, clean, and functional. You prefer robust error handling and type hinting.

## Operational Rules

1.  **File Structure:**
    *   Source code belongs in `src/voice_cloning/`.
    *   Distinguish clearly between **TTS** (Text-to-Speech) and **ASR** (Automatic Speech Recognition).
    *   Tests belong in `tests/`.
    *   Documentation belongs in `docs/` or the root `README.md`.

2.  **Dependency Management:**
    *   Always check `pyproject.toml` before suggesting new packages.
    *   Use `uv add` or `uv run` syntax in instructions.

3.  **Coding Standards:**
    *   **Type Hints:** All function signatures must have type hints.
    *   **Docstrings:** All modules and public functions must have docstrings explaining inputs and outputs.
    *   **Path Handling:** Use `pathlib.Path` instead of `os.path` strings wherever possible.
    *   **Imports:** Use absolute imports (e.g., `from src.voice_cloning.asr.canary import ...`) rather than relative imports to avoid execution context issues.

4.  **Task Workflow:**
    *   When adding a new model:
        1.  Add dependencies to `pyproject.toml`.
        2.  Create the implementation in `src/voice_cloning/[type]/`.
        3.  Expose it via `main.py`.
        4.  Create a test in `tests/`.
    *   When refactoring:
        1.  Update `PLAN.md` first.
        2.  Ensure backward compatibility or update `main.py` immediately.

## Current Focus
Refer to `PLAN.md` for the active phase. The immediate goal is ** to finalize the modular structure separating TTS and ASR components**.