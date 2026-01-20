# interactive_llm_feedback

Research code for **LLM-assisted compilation of semantic safety constraints** from object-only scenes, with **persistent user feedback overrides**.

High level:
1. Input: a scene described by named objects (name/kind/tags/radius, and optionally pose).
2. The system uses an LLM to infer which object pairs are **semantically hazardous categories** (independent of current spacing).
3. Hazards are converted into **pairwise clearance rules** suitable for downstream controllers (CBF-style constraints).
4. Users can provide feedback (“this should / should not be dangerous”), which is stored and applied in future compilations.

This README documents:
- The **Python API service** (`server/`) and core library (`cbf/`, `services/`).
- The **Unity integration package** (`Safety.zip` → `Safety/` scripts) that calls the API and visualizes/scoring hazards.

> Note: The repository also contains an `app/` folder used for internal demos/testing. **This README intentionally does not include usage instructions for `app/`.**

---

## Repository layout

- `server/` – FastAPI service exposing:
  - `GET /health`
  - `POST /v1/compile`
  - `POST /v1/feedback`
- `cbf/` – core logic: semantic hazard compilation, rule instantiation, preference handling, conflict resolution.
- `cbf/graphs/` – feedback graph + hazard graph utilities.
- `services/` – minimal OpenAI-compatible Chat Completions wrapper with optional on-disk caching.
- `models/` – example assets and local state:
  - `models/kind_cache.json`
  - `models/user_prefs.json` (default preference store)
  - `models/llm_cache/` (optional on-disk LLM cache, when enabled)

---

## Requirements

- Python **3.10+**
- An OpenAI-compatible API key (see below)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
