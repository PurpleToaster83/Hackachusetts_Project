# Requirements Document

## Introduction

Messy Desk Lab is a Streamlit web app layered over the existing Messy Desk simulation engine under `src/` and deployed to Streamlit Community Cloud. It presents four sections — Experiment, Live View, Results, and Trial Explorer — over the reused engine. The app is a thin presentation layer: Streamlit UI calls live in `app/streamlit_app.py`, and all testable logic lives in `app/logic.py`, which imports neither Streamlit, pygame, nor the renderer. The simulation core is reused as-is and never duplicated.

These requirements are derived from the approved design document (design-first workflow). They cover the four sections, deterministic replay from a minimal config key, the GUI-independent logic boundary, two proposed behavior-preserving engine changes (a shared headless trace core and an optional planner rationale), reuse of the engine's confidence-interval definition, Streamlit Community Cloud deployment, and the project constraints (read-only `robotics/`, output only under `out/`, no fabricated results, no parameter tuning). Scope is limited to what the design specifies.

## Glossary

- **Lab**: The Messy Desk Lab Streamlit application as a whole.
- **UI_Layer**: `app/streamlit_app.py`, the Streamlit entry point. Holds Streamlit calls, layout, session state, spinners, progress, and error surfacing only; contains no simulation math.
- **Logic_Layer**: `app/logic.py`, the pure, typed, headless module holding all non-Streamlit logic. Imports only `src/`, numpy, matplotlib (Agg), PIL, and the standard library.
- **Engine_Core**: The existing simulation core under `src/` (`scene`, `robot`, `planner`, `sim`, `log`, `compare`, `run_eval`), reused unchanged except for the proposed behavior-preserving additions.
- **Trace_Core**: The proposed shared headless stepping core (`src/trace.py` or an extended `sim.py`) that both `run_episode` and the trace builder call, so metrics and per-tick frames come from one code path.
- **Experiment_Section**: The section that configures `n_items`, `clutter`, trial count, seeds, and planners and runs a batch.
- **Live_View_Section**: The section that replays one trial with play/pause/step/speed using pre-rendered headless frames.
- **Results_Section**: The section that compares metrics across planners with 95% confidence intervals and an overlap/tie interpretation.
- **Trial_Explorer_Section**: The section that filters, inspects, and deterministically replays individual trials.
- **TrialConfig**: The minimal replay key carrying exactly `planner_name`, `seed`, `n_items`, and `clutter`.
- **EpisodeResult**: The engine's aggregate episode outcome `(items_sorted, items_total, steps, failed_grasps, pushes)` from `src/sim.py`.
- **ReplayFrame**: One simulated tick of a replay, mirroring the Trace_Core per-tick frame (gripper position, event, event item, optional reason, tick, cumulative counters).
- **Confidence_Interval**: The 95% interval half-width defined as `Z_95 * sample_std(n-1) / sqrt(n)` with `Z_95 = 1.96`, returning `0.0` for `n < 2`, identical to `run_eval`/`compare`.
- **Agg**: The matplotlib non-interactive backend used for headless rendering, forced before importing pyplot.
- **n_items**: The number of items on the desk for a trial, in pixels on a 640x480 desk.
- **clutter**: The clutter level for scene generation, a value in `[0, 1]`.

## Requirements

### Requirement 1: Configure and run an experiment batch

**User Story:** As a user, I want to configure object count, clutter, trial count, seeds, and planners and run a batch, so that I can compare planners over repeated trials.

#### Acceptance Criteria

1. THE Experiment_Section SHALL present controls for `n_items`, `clutter`, trial count, a seed set, and a planner selection.
2. WHERE the user sets `n_items` outside the supported range, THE Experiment_Section SHALL clamp `n_items` to the supported range before running.
3. WHERE the user sets `clutter` outside `[0, 1]`, THE Logic_Layer SHALL clamp `clutter` to `[0, 1]` before running.
4. WHEN the user selects a planner, THE Experiment_Section SHALL offer only planner names returned by the available-planner list.
5. WHEN the user runs the batch, THE Logic_Layer SHALL run each selected planner over each seed using a fresh scene and a fresh planner for each trial.
6. WHEN a batch completes, THE Logic_Layer SHALL return per-trial rows paired with each trial's real `EpisodeResult` and per-planner aggregates.
7. WHEN a batch completes, THE UI_Layer SHALL cache the batch result in session state so other sections reuse it without recomputing.

### Requirement 2: Replay one trial in Live View

**User Story:** As a user, I want to replay a single trial with play, pause, step, and speed controls, so that I can watch how a planner clears the desk without opening a native window.

#### Acceptance Criteria

1. WHEN the user selects one trial, THE Live_View_Section SHALL replay that trial using pre-rendered headless frames.
2. THE Live_View_Section SHALL provide play, pause, step, and speed controls over a session-state frame index.
3. WHEN rendering a replay, THE Logic_Layer SHALL render the trace to RGB frames once per replay and index into the cached frames for scrubbing.
4. THE Logic_Layer SHALL render frames using the matplotlib Agg backend and SHALL NOT open a pygame window.
5. WHERE a selected trial has `n_items` equal to 0, THE Live_View_Section SHALL present a zero-frame replay with a message that there are no items on the desk.
6. THE Logic_Layer SHALL produce each rendered frame as an `(height_px, width_px, 3)` uint8 RGB array with accompanying alt text.

### Requirement 3: Compare planner results with confidence intervals

**User Story:** As a user, I want to compare ticks, items sorted, failed grasps, and pushes across planners with 95% confidence intervals, so that I can judge which planner performs better and whether a difference is meaningful.

#### Acceptance Criteria

1. THE Results_Section SHALL present, per planner, the mean and 95% confidence interval for ticks, items sorted, failed grasps, and pushes over the experiment's seeds.
2. THE Logic_Layer SHALL compute each 95% confidence interval as `Z_95 * sample_std(n-1) / sqrt(n)` with `Z_95` equal to `1.96`.
3. IF a metric has fewer than 2 samples, THEN THE Logic_Layer SHALL return a confidence interval of `0.0` for that metric.
4. WHEN two planners' 95% intervals for a metric overlap, THE Results_Section SHALL report the comparison as a tie at the current sample size.
5. THE Results_Section SHALL derive its metrics only from the real `EpisodeResult` rows produced by the Experiment batch.

### Requirement 4: Filter, inspect, and replay trials in Trial Explorer

**User Story:** As a user, I want to filter trials, inspect occlusion and crowding readouts, and replay a chosen trial, so that I can understand individual outcomes in detail.

#### Acceptance Criteria

1. THE Trial_Explorer_Section SHALL filter trial rows by a predicate over planner, seed, and metric thresholds.
2. WHEN the user inspects a trial, THE Logic_Layer SHALL rebuild the trial's scene and report occluders, per-item covered fraction, and crowding neighbors.
3. WHEN the user replays a chosen trial, THE Logic_Layer SHALL deterministically rebuild and replay that trial from its `TrialConfig`.
4. THE Logic_Layer SHALL compute occluders, covered fraction, and crowding readouts by calling the Engine_Core functions and SHALL NOT reimplement that math.

### Requirement 5: Deterministic replay from the minimal config key

**User Story:** As a user, I want replays to be exactly reproducible from a trial's saved configuration, so that the same trial always looks and scores identically.

#### Acceptance Criteria

1. THE TrialConfig SHALL carry exactly `planner_name`, `seed`, `n_items`, and `clutter`.
2. WHEN replaying a trial, THE Logic_Layer SHALL construct a fresh scene and a fresh planner from the TrialConfig for each replay.
3. THE Logic_Layer SHALL NOT reuse a planner instance or a mutated scene across replays.
4. WHEN `replay_trial` is called twice with an equal TrialConfig, THE Logic_Layer SHALL produce identical trace frames.
5. WHEN a trial is replayed, THE paired `run_episode` call for the same config SHALL produce an identical `EpisodeResult`.

### Requirement 6: GUI-independent logic boundary

**User Story:** As a developer, I want all testable logic separated from the UI and the engine kept as the single source of simulation behavior, so that the app stays maintainable, testable, and headless.

#### Acceptance Criteria

1. THE Logic_Layer SHALL import neither `streamlit`, `pygame`, nor `render`.
2. THE UI_Layer SHALL import `streamlit` and the Logic_Layer only and SHALL NOT import the Engine_Core math directly for computation.
3. THE Logic_Layer SHALL perform all simulation work by calling the Engine_Core and SHALL NOT duplicate simulation behavior.
4. WHERE a module other than the renderer needs trace frames, THE Logic_Layer SHALL obtain them from the Trace_Core and SHALL NOT import `render`.

### Requirement 7: Shared headless trace core (proposed, behavior-preserving)

**User Story:** As a developer, I want the per-step trace and the aggregate metrics to come from one shared code path, so that replay and simulation cannot drift and no episode outcome changes.

#### Acceptance Criteria

1. THE Trace_Core SHALL produce per-tick frames and the aggregate `EpisodeResult` from the same stepping code path.
2. THE Trace_Core SHALL NOT alter any `EpisodeResult` for any seed, planner, or config relative to the current engine.
3. WHEN a trial is traced, THE Trace_Core SHALL produce final counters equal to the paired `EpisodeResult`, such that `sorted_count` equals `items_sorted`, `failed_count` equals `failed_grasps`, `push_count` equals `pushes`, and the last `tick` equals `steps`.
4. WHERE `render.py` reuses the promoted trace structures and builder, THE renderer SHALL keep its public API unchanged by re-exporting them.

### Requirement 8: Optional planner rationale (proposed, behavior-preserving)

**User Story:** As a user, I want a short rationale for each planner action during replay, so that I can understand why the planner chose each move.

#### Acceptance Criteria

1. WHERE a planner sets a rationale, THE planner SHALL expose it as an optional `last_reason` attribute set just before returning its chosen action.
2. WHEN the Trace_Core reads a rationale, THE Trace_Core SHALL read `last_reason` after `next_action` returns and store it on the corresponding frame.
3. WHERE a planner does not set a rationale, THE Trace_Core SHALL record an empty rationale for that action.
4. THE optional rationale SHALL NOT change which action a planner selects.
5. WHEN rationale reading is enabled, THE engine SHALL produce the same `EpisodeResult` for every seed, planner, and config as it produces with rationale reading disabled.

### Requirement 9: Bounded synchronous batch execution

**User Story:** As a user, I want large batches to stay responsive, so that the app does not hang on a heavy run under Streamlit's rerun model.

#### Acceptance Criteria

1. WHEN a batch is requested, THE Logic_Layer SHALL cap the trial count by a fixed time budget before running.
2. THE Logic_Layer SHALL cap the trial count to no more than the requested count and no fewer than 2.
3. WHILE `n_items` or the planner count increases, THE Logic_Layer SHALL reduce the capped trial count accordingly.
4. WHEN the trial count is capped, THE UI_Layer SHALL display that the run was capped to the computed trial count to stay within the time budget.
5. WHEN a batch runs, THE UI_Layer SHALL run it synchronously inside a spinner and SHALL NOT rely on background workers.

### Requirement 10: Confidence-interval definition reuse

**User Story:** As a developer, I want the app's statistics to match the engine's existing definition, so that Lab results are consistent with `run_eval` and `compare`.

#### Acceptance Criteria

1. THE Logic_Layer SHALL define the 95% confidence interval identically to `run_eval` and `compare`, using `Z_95` equal to `1.96` and sample standard deviation with `n-1` denominator.
2. IF fewer than 2 samples are provided, THEN THE Logic_Layer SHALL return a confidence interval of `0.0`.
3. THE Logic_Layer SHALL report two planners' intervals as overlapping when neither interval lies entirely outside the other.

### Requirement 11: Streamlit Community Cloud deployment

**User Story:** As a maintainer, I want the app to deploy cleanly to Streamlit Community Cloud, so that it runs on a headless container without pulling display libraries.

#### Acceptance Criteria

1. THE Lab SHALL provide an app-level `app/requirements.txt` that excludes `pygame`, `imageio`, and `pytest`.
2. THE Lab SHALL leave the root `requirements.txt` unchanged for full-project local work.
3. THE UI_Layer SHALL force the matplotlib Agg backend before importing pyplot.
4. THE Live_View_Section SHALL render replays from headless frames and SHALL NOT open a pygame window on the deployment container.
5. WHILE handling a long batch, THE UI_Layer SHALL process it synchronously under the time-budget cap with caching and SHALL NOT claim background workers.

### Requirement 12: Output location and project constraints

**User Story:** As a maintainer, I want the app to respect the project's data and honesty constraints, so that inputs stay read-only and results stay trustworthy.

#### Acceptance Criteria

1. WHEN the Lab writes any generated output, THE Logic_Layer SHALL write it only under `out/`.
2. THE Lab SHALL NOT write to `data/` or `robotics/`.
3. THE Lab SHALL derive displayed metrics and tables only from real `run_episode` and `run_eval` output and SHALL NOT fabricate results.
4. THE Lab SHALL NOT tune simulation parameters.

### Requirement 13: Error handling and input validation

**User Story:** As a user, I want clear handling of invalid inputs and run failures, so that the app stays responsive and tells me what went wrong.

#### Acceptance Criteria

1. IF a planner name is not in the available-planner list, THEN THE Logic_Layer SHALL raise a `ValueError`.
2. IF `run_experiment`, `replay_trial`, or `render_frames` raises an exception, THEN THE UI_Layer SHALL surface the error message and remain responsive for a retry.
3. WHEN the matplotlib backend is required, THE Logic_Layer SHALL force the Agg backend before importing pyplot so that no display is required.
