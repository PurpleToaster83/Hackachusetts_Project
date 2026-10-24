# How Kiro was used in this project

This document records how the Kiro IDE was used to build the Messy Desk
simulator and the Messy Desk Lab app. It describes only what actually happened.

## Kiro IDE as the primary development environment

All development in this repository was done inside Kiro, an AI-powered IDE. Kiro
read and edited files, ran the test suite and the evaluation harness, and
inspected `git` state throughout.

## Two modes of work: vibe-coded engine, spec-driven app

The simulation engine under `src/` (`scene.py`, `robot.py`, `planner.py`,
`sim.py`, `log.py`, `compare.py`, `run_eval.py`, `render.py`) was built as
conversational "vibe" coding — direct, iterative requests with tests added
alongside each module.

The Streamlit app ("Messy Desk Lab") was built with spec-driven development. A
single spec lives in `.kiro/specs/messy-desk-lab/` and follows the design-first
workflow:

- `design.md` — the technical design (written first; an inspection-only pass
  over the engine that assessed the interfaces the app needs and proposed two
  behavior-preserving engine changes).
- `requirements.md` — EARS-format requirements derived from the design.
- `tasks.md` — an implementation plan ordered for earliest deployability, built
  and executed one stage at a time (logic layer → minimal Streamlit shell →
  deploy prep → trace core + Live View → planner rationale → Trial Explorer →
  bounded batch + error polish).

Stages were executed individually with a review checkpoint between them, rather
than running the whole plan unattended.

## Agent hook: automatic change log

`.kiro/hooks/ai-change-logger.json` defines a `Stop`-triggered agent hook named
"AI Change Logger". After each agent turn it inspects the `git` diff and appends
a dated entry to `AI_CHANGELOG.md` describing the agent's own changes (and
writes nothing when a turn made no code change). `AI_CHANGELOG.md` at the repo
root is the running output of that hook.

## Steering docs

Three always-included steering files in `.kiro/steering/` shaped how Kiro
worked:

- `product.md` — product scope, what the challenge is judged on, and the
  honesty rule (never describe the simulation as real vision; keep the README
  limitations accurate).
- `tech.md` — stack, simulation parameters, commands, and the build order and
  per-step verification workflow.
- `structure.md` — module layout, stable module contracts, and conventions
  (headless core with no pygame, all randomness from a seeded generator,
  generated output only under `out/`).

## Tests

The suite is run with `pytest` and currently reports 151 passing tests (one test
file per module under `tests/`). The tests are example-based — hand-picked seeds
and configs via `pytest.mark.parametrize`, using `pytest.approx`/`pytest.raises`,
not property-based generative testing — but they assert strong invariants:
determinism per seed, the occlusion/coverage geometry math, grasp-probability
behavior, dataset-schema conformance, confidence-interval math matching the
engine definition, and — for the headless trace core promoted out of the
renderer — `tests/test_trace.py` parity tests proving the trace reproduces
`run_episode`'s outcome exactly (so that refactor changed no behavior).

## MCP and Kiro powers

No MCP servers and no Kiro powers were used in this project. Development relied
only on Kiro's built-in file, terminal, spec, hook, and steering capabilities.

## One thing Kiro helped most with, and one limitation

- Helped most: keeping the app strictly layered over the unchanged simulation
  engine — spec-driven staging plus the trace-parity tests made it safe to
  refactor shared code and add the UI without altering any `EpisodeResult`.
- Limitation: Kiro cannot verify genuinely interactive or external outcomes on
  its own — it could not click through the Streamlit UI in a browser or perform
  the Streamlit Community Cloud deployment, so those steps were left for a human
  and reported honestly rather than claimed.


## Explored

Two larger features were built on separate branches and merged to main only after they passed the test gate, so both are now live in the app:

- **Shape model** — item footprints default to axis-aligned boxes (matching the published results) with an experimental toggle for rotated convex polygon footprints. Started as a time-boxed spike on an `experiment-shapes` branch, then test-covered and merged.
- **Learned linear policy** — a `LearnedPlanner` whose coverage/distance/crowding weights come from a reproducible seeded search evaluated on held-out seeds. Also built on a branch and merged after test-gating.

Both are selectable in the Streamlit app (shape-model dropdown and the "learned" strategy), with the axis-aligned box engine kept as the default so the reported baseline results stay tied to the validated path.