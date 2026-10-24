# Implementation Plan: Messy Desk Lab

## Overview

This plan builds the Messy Desk Lab Streamlit app so it is **deployable as early as possible**. The stages are ordered to reach a live, public URL at Stage 3, then layer richer sections on top. Every task is independently completable and MUST leave `pytest -q` green. No task changes simulation behavior; the two engine touches (Trace_Core promotion in Stage 4, `planner.last_reason` in Stage 5) are behavior-preserving and each carries a determinism/parity test proving no `EpisodeResult` changes.

Language: **Python 3.10+** (the design and engine are concrete Python; no pseudocode language choice is needed).

Hard constraints honored throughout:
- `app/logic.py` imports neither `streamlit`, `pygame`, nor `render`. Flow is `UI -> logic -> src/ core`.
- The simulation core under `src/` stays headless; only `render.py` imports pygame.
- `robotics/` and `data/` are read-only; generated output goes only under `out/` (gitignored).
- CI math reuses the engine definition: `Z_95 = 1.96`, sample std with `n-1` denominator, `0.0` for `n < 2` (matches `src/compare.py::_ci95` and `src/run_eval.py`).
- No fabricated results, no parameter tuning.
- One test file per new module under `tests/`, matching the existing `tests/test_*.py` convention. Every code task writes/updates its tests and runs `pytest -q`.

## Tasks

- [x] 1. Stage 1 — Headless logic layer: data models, batch runner, CI stats
  - [x] 1.1 Create `app/` package and the typed data models in `app/logic.py`
    - Create `app/__init__.py` and `app/logic.py`.
    - Define the dataclasses from the design: `ExperimentConfig`, `TrialConfig` (exactly `planner_name, seed, n_items, clutter`), `TrialRow`, `Stat`, `PlannerStat`, `ExperimentResult`.
    - Import `EpisodeResult` from `src.sim`; import nothing from `streamlit`, `pygame`, or `render`.
    - Add `available_planners() -> list[str]` and `make_planner(name, seed)` that construct a FRESH planner by name (`RandomPlanner(seed=seed)`, `NearestPlanner()`, `LeastCoveredPlanner()`, `NearestRetryPlanner()`, `OcclusionPlanner()` from `src.planner`) WITHOUT importing `render.py`'s `PLANNER_FACTORIES`; raise `ValueError` for an unknown name.
    - Add `planner_description(name) -> str` and `build_scene(seed, n_items, clutter)` delegating to `src.scene.make_scene`; clamp `clutter` to `[0, 1]`.
    - _Requirements: 1.4, 1.1, 5.1, 6.1, 6.3, 13.1, 12.4_
  - [x] 1.2 Implement the CI statistics functions in `app/logic.py`
    - Add `Z_95 = 1.96`, `mean(values)`, `ci95(values)` (= `Z_95 * sample_std(n-1) / sqrt(n)`, `0.0` for `n < 2`), and `overlap(a, b)` over two `Stat` intervals.
    - Mirror `src/compare.py::_ci95` and `src/run_eval.py` exactly; do not re-derive or duplicate the engine math beyond the shared definition.
    - _Requirements: 3.2, 3.3, 10.1, 10.2, 10.3_
  - [x] 1.3 Implement `run_experiment`, `aggregate_stats`, and the output-location guard in `app/logic.py`
    - `run_experiment(config)` loops each planner over each seed building a FRESH `make_scene(...)` + FRESH `make_planner(...)` per trial and calling `src.sim.run_episode(scene, planner, seed=seed)`; collect `TrialRow`s paired with the real `EpisodeResult`.
    - `aggregate_stats(rows) -> list[PlannerStat]` computes mean ± 95% CI per metric (ticks=`steps`, sorted=`items_sorted`, failed=`failed_grasps`, pushes=`pushes`) per planner using `ci95`.
    - Add an output-path helper that resolves any generated file under `out/` only and rejects paths outside `out/` (never `data/` or `robotics/`).
    - _Requirements: 1.5, 1.6, 3.1, 3.5, 5.2, 5.3, 6.3, 12.1, 12.2, 12.3_
  - [x]* 1.4 Write unit tests for Stage 1 in `tests/test_logic.py`
    - CI math: `ci95` on a known list equals the hand-computed value; `ci95([x])` and `ci95([])` return `0.0`; `overlap` on known intervals. _Requirements: 3.2, 3.3, 10.1, 10.2, 10.3_
    - Determinism: `run_experiment` with equal config yields identical rows/stats; same seed → identical results. _Requirements: 1.5, 5.2, 5.3_
    - Rows carry real `EpisodeResult`; `aggregate_stats` matches a hand aggregation. _Requirements: 1.6, 3.1, 3.5_
    - `make_planner` raises `ValueError` for an unknown name; `available_planners()` is stable. _Requirements: 1.4, 13.1_
    - No forbidden imports: assert `app.logic` has not imported `streamlit`, `pygame`, or `render` (via `sys.modules` / source inspection). _Requirements: 6.1_
    - Output guard rejects paths outside `out/`. _Requirements: 12.1, 12.2_
    - _Requirements: 1.4, 1.5, 1.6, 3.1, 3.2, 3.3, 5.2, 5.3, 6.1, 10.1, 10.2, 10.3, 12.1, 12.2, 13.1_

- [x] 2. Stage 2 — Minimal runnable Streamlit shell (Experiment + Results)
  - [x] 2.1 Create `app/requirements.txt` for headless deployment
    - List `numpy`, `pandas`, `matplotlib`, `streamlit`, and `pillow` (only if `st.image` needs it). Exclude `pygame`, `imageio`, `pytest`.
    - Leave the root `requirements.txt` unchanged.
    - _Requirements: 11.1, 11.2_
  - [x] 2.2 Create `app/streamlit_app.py` with the Experiment and Results sections
    - UI-only module: imports `streamlit` and `app.logic` only; never imports `src/` math directly for computation and never imports `render.py`.
    - Force `matplotlib.use("Agg")` before any `pyplot` import at module top.
    - Experiment section: controls for `n_items` (clamped to the supported range), `clutter` slider in `[0, 1]`, trial/seed count, and a planner multiselect populated from `logic.available_planners()`; a Run button calling `logic.run_experiment` inside `st.spinner`; cache the `ExperimentResult` in `st.session_state`.
    - Results section: per-planner mean ± 95% CI table for ticks, items sorted, failed grasps, pushes, derived only from the cached `EpisodeResult` rows; show a tie note when two planners' intervals overlap (via `logic.overlap`).
    - _Requirements: 1.1, 1.2, 1.7, 3.1, 3.4, 3.5, 6.2, 11.3_
  - [x]* 2.3 Write import-boundary and smoke tests in `tests/test_streamlit_app.py`
    - Assert `app/streamlit_app.py` imports `streamlit` + `app.logic` only (no direct `src/` math import, no `render`). _Requirements: 6.2_
    - Assert the module forces the Agg backend before `pyplot` (source inspection). _Requirements: 11.3_
    - _Requirements: 6.2, 11.3_

- [ ] 3. Stage 3 — Deploy to Streamlit Community Cloud (manual verification)
  - [ ]* 3.1 Prepare deployment artifacts and a deploy checklist (agent-prepared)
    - Verify `app/streamlit_app.py` runs top-to-bottom without a pygame/display dependency and that `app/requirements.txt` excludes `pygame`/`imageio`/`pytest`.
    - Write a short `app/DEPLOY.md` with the Community Cloud settings: main file `app/streamlit_app.py`, requirements `app/requirements.txt`, Python 3.10+.
    - _Requirements: 11.1, 11.3, 11.4_
  - [ ]* 3.2 Connect the repo and verify the public URL (HUMAN / MANUAL — agent cannot complete)
    - Human action: connect the GitHub repo in Streamlit Community Cloud, set main file `app/streamlit_app.py` and requirements `app/requirements.txt`, deploy, then confirm the public URL loads and a batch run completes with no pygame/display error.
    - _Requirements: 11.4, 11.5_

- [ ] 4. Stage 4 — Trace_Core promotion + Live View replay
  - [ ] 4.1 Promote the trace structures and stepping logic into a headless core (`src/trace.py`)
    - Move `ItemState`, `ContainerState`, `TickFrame`, `Trace`, and the `build_trace` stepping logic from `src/render.py` into a new headless `src/trace.py` (no pygame import).
    - Behavior-preserving: keep the exact `numpy.random.default_rng(seed)` draw order, `effective_grasp_prob`, `travel_ticks`, `FAILED_GRASP_TICKS`, `push`, and the `10 * items_total` default budget so outcomes stay bit-for-bit identical to `src/sim.run_episode`.
    - Make `src/render.py` re-export `build_trace`, `Trace`, `TickFrame`, `ItemState`, `ContainerState` (and keep `PLANNER_FACTORIES`) from `src/trace.py` so its public API is unchanged.
    - _Requirements: 7.1, 7.4_
  - [ ]* 4.2 Write the Trace_Core parity/determinism test in `tests/test_trace.py`
    - **Property 2: Trace matches sim** — for a sweep of seeds/planners/configs, trace final counters equal the paired `run_episode` `EpisodeResult`: `sorted_count == items_sorted`, `failed_count == failed_grasps`, `push_count == pushes`, last `tick == steps`.
    - **Validates: Requirements 7.2, 7.3**
    - Also assert two `build_trace` calls with equal config yield identical frames. **Validates: Requirement 5.4**
    - Confirm existing `src/` and `render` tests still pass (run `pytest -q`). **Validates: Requirement 7.4**
    - _Requirements: 5.4, 7.2, 7.3, 7.4_
  - [ ] 4.3 Implement replay + headless frame rendering in `app/logic.py`
    - Add `ReplayFrame`, `ReplayResult`, `ContainerBox`, `ItemBox`, `FrameImage` dataclasses.
    - `replay_trial(config) -> ReplayResult`: rebuild a FRESH scene + FRESH planner from `TrialConfig`, call the `src.trace` core (never `render.py`), map `TickFrame`s to `ReplayFrame`s, attach containers and the paired `EpisodeResult`.
    - `draw_frame(frame, containers, width_px, height_px)` renders one `(height_px, width_px, 3)` uint8 RGB array with matplotlib Agg; `render_frames(frames, containers)` renders the whole trace once, returning `FrameImage`s with alt text.
    - _Requirements: 2.1, 2.3, 2.4, 2.6, 5.2, 5.3, 5.5, 6.4, 13.3_
  - [ ]* 4.4 Write replay/render unit tests in `tests/test_logic.py`
    - **Property 1: Determinism per config** — `replay_trial` twice with equal `TrialConfig` yields identical frames. **Validates: Requirement 5.4**
    - **Property 2: Trace matches sim** at the app layer — replay final counters equal the paired `run_episode` `EpisodeResult`. **Validates: Requirement 5.5**
    - `draw_frame` returns `(height_px, width_px, 3)` uint8; a `fail` event marks only `event_item`. **Validates: Requirements 2.6, 2.4**
    - `n_items == 0` yields a zero-/single-frame replay with `items_total == 0`. **Validates: Requirement 2.5**
    - _Requirements: 2.4, 2.5, 2.6, 5.4, 5.5_
  - [ ] 4.5 Add the Live View section to `app/streamlit_app.py`
    - Pick one trial from the cached experiment; call `logic.replay_trial` then `logic.render_frames` once (cache frames in `st.session_state`).
    - Provide play/pause/step/speed over a session-state frame index using `st.image`; never open a pygame window.
    - For an empty replay (`n_items == 0`) show a "no items on this desk" message.
    - _Requirements: 2.1, 2.2, 2.5, 11.4_

- [ ] 5. Stage 5 — Optional planner rationale (`last_reason`)
  - [ ] 5.1 Add the optional `last_reason` attribute to the planners in `src/planner.py`
    - Set `self.last_reason` just before returning the chosen action in each `next_action` branch, per the design's mapping (e.g. RandomPlanner "random push"/"random grasp", NearestPlanner "nearest item", LeastCoveredPlanner "least-covered item", NearestRetryPlanner "nearest (non-skipped)"/"nearest (all skipped)", OcclusionPlanner "stubborn-retry: pushing coverer"/"singulating: clearing crowder pays off"/"nearest exposed item"/"all covered: pushing least-covered's coverer").
    - Must not change the return type or which action is chosen. Default to `""` when unset.
    - Make the Trace_Core read `getattr(planner, "last_reason", "")` after `next_action` returns and store it on the frame's `reason` field.
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - [ ]* 5.2 Write the rationale behavior-preserving test in `tests/test_planner.py`
    - **Property 4: Rationale is behavior-preserving** — for every seed/planner/config, enabling rationale reading leaves every `EpisodeResult` identical to a run with it disabled, and `last_reason` never alters the selected action.
    - **Validates: Requirements 8.4, 8.5**
    - _Requirements: 8.4, 8.5_

- [ ] 6. Stage 6 — Trial Explorer
  - [ ] 6.1 Implement filter + inspect in `app/logic.py`
    - `filter_trials(rows, predicate)` filters `TrialRow`s by a pure predicate over planner, seed, and metric thresholds.
    - `inspect_trial(config) -> TrialDetail`: rebuild the scene from `TrialConfig` and report occluders, per-item covered fraction, and crowding neighbors by calling `src.scene.occluders`, `covered_fraction`, `crowding_neighbors` (never reimplementing the math).
    - Reuse `replay_trial` for deterministic replay of a chosen trial.
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 6.3_
  - [ ]* 6.2 Write Trial Explorer unit tests in `tests/test_logic.py`
    - `filter_trials` returns only rows matching the predicate. **Validates: Requirement 4.1**
    - `inspect_trial` readouts equal direct `src.scene` calls on the same rebuilt scene. **Validates: Requirements 4.2, 4.4**
    - Replaying a chosen trial twice is deterministic. **Validates: Requirement 4.3**
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [ ] 6.3 Add the Trial Explorer section to `app/streamlit_app.py`
    - Filter the cached trial rows; show inspection readouts (occluders / covered fraction / crowding); replay a chosen trial deterministically by reusing the Live View render path.
    - _Requirements: 4.1, 4.2, 4.3_

- [ ] 7. Stage 7 — Bounded-batch cap and error polish
  - [ ] 7.1 Implement `cap_batch_size` and wire it into `run_experiment`
    - `cap_batch_size(trials, n_items, n_planners) -> int` clamps the trial count by a fixed time budget, never above the request and never below 2, shrinking as `n_items`/`n_planners` grow (per the design pseudocode).
    - `run_experiment` applies the cap before looping and records whether the run was capped.
    - _Requirements: 9.1, 9.2, 9.3_
  - [ ]* 7.2 Write `cap_batch_size` unit tests in `tests/test_logic.py`
    - Never exceeds the request; never below 2; shrinks as `n_items`/`n_planners` grow. _Requirements: 9.1, 9.2, 9.3_
  - [ ] 7.3 Add capped-run notice and error handling to `app/streamlit_app.py`
    - Show a caption stating the run was capped to N trials to stay within the time budget when capping occurred.
    - Wrap `run_experiment`/`replay_trial`/`render_frames` in `try/except`, surfacing the message via `st.error` and staying responsive for retry.
    - Run batches synchronously inside `st.spinner`; no background workers. Keep `matplotlib.use("Agg")` forced before `pyplot`.
    - _Requirements: 9.4, 9.5, 11.5, 13.2, 13.3_

- [ ] 8. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional (test and manual/deploy sub-tasks) and can be skipped for a faster MVP; core implementation tasks are never optional.
- Stage 3's connect-and-verify step (3.2) requires human action in Streamlit Community Cloud and cannot be completed autonomously by the agent; the agent prepares everything needed (3.1).
- The two engine touches are behavior-preserving: Stage 4 (Trace_Core) carries a parity test (4.2) and Stage 5 (`last_reason`) carries a determinism test (5.2), each proving no `EpisodeResult` changes.
- Every code task writes/updates its tests under `tests/` and must leave `pytest -q` green.
- Each task references specific requirement sub-clauses for traceability.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "2.1", "4.1"] },
    { "id": 2, "tasks": ["1.4", "2.2", "3.1", "4.2", "5.1"] },
    { "id": 3, "tasks": ["2.3", "3.2", "4.3", "5.2", "7.1"] },
    { "id": 4, "tasks": ["4.4", "4.5", "6.1", "7.2"] },
    { "id": 5, "tasks": ["6.2", "6.3", "7.3"] }
  ]
}
```
