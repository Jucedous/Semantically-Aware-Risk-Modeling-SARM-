# interactive_llm_feedback

This repository contains research code for **LLM-assisted compilation of semantic safety constraints** from object-only scenes, with **persistent user feedback overrides**.

At a high level, the pipeline:

1. Takes a scene described by named objects (name/kind/tags/radius and optionally pose).
2. Uses an LLM to infer which object pairs are **semantically hazardous categories** (independent of current spacing).
3. Converts hazardous pairs into weighted, “soft clearance” pairwise rules suitable for downstream controllers (e.g., CBF-style constraints).
4. Allows users to provide feedback (“this should/should not be dangerous”), stores it, and resolves future hazards using those stored preferences.

This README documents the **core library and API service**. (The repository also contains experimental/demo code that is intentionally not covered here.)

---

## Repository layout

- `cbf/` – core logic: semantic hazard compilation, rule instantiation, preference handling, and conflict resolution.
- `services/` – minimal OpenAI-compatible Chat Completions wrapper with optional on-disk caching.
- `server/` – FastAPI service exposing compilation and feedback endpoints.
- `models/` – example assets and local state (kind cache, user preference store, optional LLM cache directory).

---

## Requirements

- Python **3.10+**
- An OpenAI-compatible API key (see below)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
