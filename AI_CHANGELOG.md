# AI Changelog

## 2026-10-24 - Fix scene occlusion and determinism bugs, add scene tests

### Changes
- Fixed `occluders()` so it only reports items that are actually covered, letting `len()` count covered items.
- Made same-seed scenes compare equal so determinism is verifiable.
- Added a scene test suite covering determinism, overlap math, and clutter behavior.
- Added a root `conftest.py` so tests can import `src.scene` without an installed package.

### Files Modified
- `src/scene.py` — occlusion fix plus value comparability for `Scene`.
- `tests/test_scene.py` — new test module for the scene model.
- `conftest.py` — new pytest config placing the project root on `sys.path`.

### Technical Details
- `occluders()` previously assigned an entry for every item (empty list when nothing covered it), so `len(occluders(scene))` counted all items. It now skips items with zero/negative area and only adds an entry when at least one higher-`z` item exceeds the 0.25 overlap threshold.
- `Scene.rng` was an equality-participating field, so two scenes built from the same seed were not `==`. Marked it `field(repr=False, compare=False)`; `make_scene` already used a single `np.random.default_rng(seed)` and `Item` already used plain floats, so scenes and items now compare by value.
- The occlusion threshold is strictly-greater-than 0.25 (an exact 0.25 overlap is not occluding), which the overlap-math test pins down.

### Testing
- Ran `.venv/bin/pytest -q` (Python 3.12, pytest 9.1.1): 6 passed in 0.06s.
- Sanity check of clutter margin: average covered items over 20 seeds was 2.40 at clutter 0.1 and 8.65 at clutter 0.9.

## 2026-10-24 - Add grasp model, episode runner, and random planner

### Changes
- Added the grasp success model as a pure function.
- Added the episode runner with its `EpisodeResult` dataclass.
- Added a `RandomPlanner` baseline (nearest-first and occlusion-aware intentionally deferred to the next build step).
- Added test suites for the grasp model and the episode runner/planner.

### Files Modified
- `src/robot.py` — new module with `grasp_success_prob(covered_fraction)`.
- `src/sim.py` — new module with `EpisodeResult` and `run_episode`.
- `src/planner.py` — new module with `RandomPlanner` only.
- `tests/test_robot.py` — new tests for the grasp formula and clamping.
- `tests/test_sim.py` — new tests for determinism, empty scene, and the no-overlap sorting guarantee.

### Technical Details
- `grasp_success_prob` returns `1 - 0.8 * covered_fraction`, clamped to `[0.05, 1.0]`. At full coverage (1.0) the value is 0.2, still above the 0.05 floor; the floor only engages for coverage beyond 1.0.
- `run_episode(scene, planner, logger=None, seed=0)` draws from a seeded `numpy.random.default_rng(seed)`. Each step the planner returns an item id; the grasp succeeds when `rng.random() < grasp_success_prob(covered_fraction(scene, item_id))`, reusing the existing `covered_fraction` helper in `scene.py`. Success removes the item from `scene.items` and increments `items_sorted`; failure increments `failed_grasps`. The loop stops when the desk is clear or `steps` reaches `10 * items_total`. The `logger` argument is accepted for interface stability but unused for now.
- `RandomPlanner` holds its own seeded generator, independent of the scene's rng, and returns a uniformly random remaining item id (or `None` when empty).
- Planner interface used is `next_action(scene)`; the steering doc lists `next_action(scene, gripper)`, but `Gripper` does not exist yet (it is the next build step), so the signature was kept minimal and will be aligned when the gripper lands.
- Correction to the task premise: `make_scene(clutter=0)` does not guarantee a non-overlapping layout (clutter only controls clustering toward center; `seed=11, n_items=15, clutter=0.0` still produced occluders). The no-overlap test therefore uses a hand-built scene with items spaced 60 px apart and asserts `occluders(scene) == {}` and zero coverage before running.

### Testing
- Ran `.venv/bin/pytest -q` this turn: 13 passed in 0.05s.
- Intermediate runs caught two faulty test assumptions I then fixed: a float-equality assertion on `0.2` (switched to `math.isclose`), and the mistaken assumption that `make_scene(clutter=0)` yields no overlaps (switched to a hand-built spaced scene).

## 2026-10-24 - Add gripper, push primitive, tick-based timing, and action protocol

### Changes
- Added a `Gripper` dataclass and `Grasp`/`Push` action types, and a `push` primitive that moves an item and reduces what it covered.
- Made episode time tick-based: each action costs ticks proportional to gripper travel distance (capped speed, 15 Hz), and a failed grasp costs an extra tick.
- Changed the planner interface to `next_action(scene, gripper)`; planners now return a `Grasp` or `Push` action. Updated `RandomPlanner` to match.
- `EpisodeResult.pushes` is now populated.
- Extended the robot and sim test suites (travel cost, push coverage reduction, push counting, determinism).

### Files Modified
- `src/robot.py` — added `Gripper`, `Grasp`, `Push`, `travel_ticks`, `push`, and `TICK_HZ`/`MAX_SPEED` constants alongside the existing grasp model.
- `src/sim.py` — reworked `run_episode` around the action protocol, gripper travel, tick budget, and push counting.
- `src/planner.py` — `RandomPlanner.next_action(scene, gripper)` returns a random grasp or (20% of the time) a push.
- `tests/test_robot.py` — added tests for the gripper default, travel-cost monotonicity/floor, push coverage reduction, and desk clamping.
- `tests/test_sim.py` — switched the no-overlap guarantee to a grasp-only planner, added push-counting and travel-cost tests, kept determinism/empty-scene tests.

### Technical Details
- `travel_ticks(distance) = max(1, ceil(distance / MAX_SPEED))` with `MAX_SPEED = 120.0` px/tick and `TICK_HZ = 15`. Any action (even zero-distance) costs at least one tick.
- `run_episode` now constructs a `Gripper(0, 0)`, travels to the target item (adding travel ticks), then applies the action. `Grasp` uses the unchanged `grasp_success_prob(covered_fraction(...))`; a failed grasp adds one tick and increments `failed_grasps`. `Push` calls `robot.push`, increments `pushes`, and moves the gripper to follow the item.
- `push(scene, item_id, dx, dy)` shifts the item and clamps its center to the desk. Coverage is geometric, so moving a covering item away reduces (or zeroes) the covered fraction of items beneath it — no explicit coverage bookkeeping needed.
- Semantic change: `EpisodeResult.steps` now counts ticks rather than actions, so the `10 * items_total` cap is a tick budget. Flagged to the user for review; could revert to an action budget if preferred.
- The no-overlap sorting test previously asserted `steps == n_items`; because `RandomPlanner` now sometimes pushes and steps are tick-based, that exact-count assertion no longer holds. The test now uses a deterministic grasp-only planner to keep the sorting guarantee checkable, with travel cost and push counting covered by separate tests.
- Planner interface now matches the steering contract `next_action(scene, gripper)`.

### Testing
- Ran `.venv/bin/pytest -q` this turn: 20 passed in 0.06s.
- Manual episode smoke this turn (`RandomPlanner`, n_items=15, clutter=0.6): seed 3 sorted 15/15 in 39 ticks with 6 pushes and 0 failed grasps; seed 7 sorted 15/15 in 30 ticks with 3 pushes and 0 failed grasps.
- `run_eval.py --quick` was not run because that module does not exist yet (it is a later build step).

## 2026-10-24 - Add nearest/occlusion planners, failed-grasp penalty, and comparison harness

### Changes
- Made a failed grasp cost 3 extra ticks via a named `FAILED_GRASP_TICKS` constant (was a hardcoded 1).
- Added `NearestPlanner` (grasp nearest remaining item, coverage-blind) and `OcclusionPlanner` (grasp nearest exposed item; singulate by pushing coverers when needed).
- Added `src/compare.py`, a headless harness that sweeps the three planners over 30 seeds x {5,15,25} items x clutter {0.3,0.9} and prints a metrics table.
- Added an optional `max_ticks` budget parameter to `run_episode` so the harness can measure items sorted within a tick budget.
- Extended the robot, sim, and planner test suites.

### Files Modified
- `src/robot.py` — added `FAILED_GRASP_TICKS = 3`.
- `src/sim.py` — charge `FAILED_GRASP_TICKS` on a failed grasp; added optional `max_ticks` budget (defaults to the prior `10 * items_total`).
- `src/planner.py` — added `NearestPlanner` and `OcclusionPlanner`, plus `_distance`/`_nearest` helpers and retry/push constants.
- `src/compare.py` — new comparison harness (self-bootstraps `sys.path` to run as `python src/compare.py`).
- `tests/test_robot.py` — assert the failed-grasp penalty constant is 3.
- `tests/test_sim.py` — test that one failed grasp costs travel + `FAILED_GRASP_TICKS`.
- `tests/test_planner.py` — new tests for nearest selection, occlusion never grasping a covered item while an exposed one exists, push-on-all-covered and push-after-two-failures paths, and per-planner determinism.

### Technical Details
- `OcclusionPlanner` computes `occluders(scene)` each call. Uncovered items (not in that map) are candidates; it grasps the nearest to the gripper. When no uncovered item remains, it picks the item with the fewest coverers and pushes its top coverer (highest z) away: direction is coverer-center minus item-center, normalized, magnitude about one coverer width.
- Failure inference without extra sim plumbing: the planner records the id it last grasped; on the next call, if that item is still present the grasp must have failed, so it bumps a per-item attempt counter (a sorted item is gone, so its counter is retired). After two failures on an item that has a coverer, it pushes the coverer before retrying. This keeps the `next_action(scene, gripper)` contract unchanged.
- `run_episode(max_ticks=...)` overrides the stopping budget; left unset it keeps the original `10 * items_total` behavior, so existing callers and tests are unaffected.
- `compare.py` budget is `round(2.7 * n_items)` ticks: 5->14, 15->40, 25->68, matching the "40 ticks at 15 items" spec. It runs each cell twice per seed — once unbudgeted for ticks-to-clear/failed/pushes, once capped for items-sorted-in-budget.

### Testing
- Ran `.venv/bin/pytest -q` this turn: 29 passed in 0.06s.
- Ran `python src/compare.py` this turn. Key real numbers (mean over 30 seeds):
  - clutter 0.9, 25 items: occlusion cleared in 33.4 ticks, 1.53 failed grasps, sorted 25.0/25 within a 68-tick budget; nearest took 88.2 ticks, 15.23 failed grasps, sorted only 17.0/25.
  - clutter 0.9, 15 items: occlusion 19.3 ticks / 0.47 failed / 15.0 sorted vs nearest 34.1 ticks / 4.17 failed / 13.7 sorted.
  - Failed grasps for coverage-blind planners rise with clutter (nearest at 25 items: 3.03 at clutter 0.3 -> 15.23 at 0.9), confirming grasp success depends on coverage.
  - Occlusion beats nearest at 15+ items in every tested cell, satisfying the workflow guardrail.
- Verified the harness output is byte-identical across two runs (deterministic). Removed the temp files used for the diff check.

## 2026-10-24 - Add clearance penalty, two baselines, value-based singulation, and CI table

### Changes
- Added a crowding clearance penalty: a grasp on an item with another item within 15 px of its edges is multiplied by 0.7.
- Added two baseline planners: `LeastCoveredPlanner` (grasp smallest covered fraction, ties by distance) and `NearestRetryPlanner` (nearest, but skip an item for 5 actions after 2 failed grasps).
- Updated `OcclusionPlanner` to singulate a crowded target only when the expected failed-grasp saving beats the push cost (rule stated in its docstring; constants not tuned to results).
- Updated `compare.py` to cover all five planners and print mean +/- 95% interval for ticks-to-clear and failed grasps, plus the pushes column.
- Extended robot and planner test suites.

### Files Modified
- `src/scene.py` — added `_edge_gap`, `crowding_neighbors`, and `is_crowded` geometry helpers.
- `src/robot.py` — added `CLEARANCE_MARGIN = 15.0`, `CLEARANCE_PENALTY = 0.7`, and `effective_grasp_prob(scene, item_id)` combining coverage and crowding.
- `src/sim.py` — grasp success now uses `effective_grasp_prob` instead of coverage alone.
- `src/planner.py` — added `LeastCoveredPlanner` and `NearestRetryPlanner`; reworked `OcclusionPlanner` with a value-based declutter push and expected-tick helpers.
- `src/compare.py` — five planners, 95% CI columns, pushes column.
- `tests/test_robot.py` — clearance penalty constant, crowded/uncrowded probability, and push-removes-penalty tests (with corrected geometry).
- `tests/test_planner.py` — least-covered selection, nearest-retry skip and return, value-based singulation behavior, and five-planner determinism.

### Technical Details
- Crowding test: `_edge_gap(a, b)` is 0 when boxes touch/overlap, else the larger per-axis gap. `crowding_neighbors` lists other items within `CLEARANCE_MARGIN` px; `is_crowded` is the boolean form.
- `effective_grasp_prob` = `grasp_success_prob(covered_fraction) * (0.7 if crowded else 1.0)`, floored at `MIN_GRASP_PROB`. Because coverage and crowding are recomputed from live geometry each grasp, a push that moves either item clear automatically restores the probability — no penalty bookkeeping.
- OcclusionPlanner singulation rule (fixed): expected wasted ticks before a success is `FAILED_GRASP_TICKS * (1 - p) / p`; clearing a crowd raises p from `base * 0.7` to `base`. Push only when that saving strictly exceeds push cost `travel_ticks(dist_to_neighbour) + 1`. For a fully exposed crowded item (base 1.0) the saving is ~1.29 ticks, below the 2-tick minimum push cost, so the declutter push never fires there by design.
- Confirmed this is not a bug: OcclusionPlanner still pushes via the coverer-clearing paths (stubborn retry / all-covered), just rarely.
- `compare.py` `_ci95` uses the sample std (n-1) and returns 0 for n<2.

### Testing
- Ran `.venv/bin/pytest -q` this turn: 37 passed in 0.07s. (Two robot tests failed on a first run due to my own wrong test coordinates — a 30 px gap instead of the intended 10 px; fixed the geometry and reran green.)
- Ran `python src/compare.py` this turn (30 seeds, clutter 0.3 and 0.9). Sample real numbers at clutter 0.9, 25 items: nearest 164.5 +/- 17.1 ticks / 34.50 failed / 9.0 sorted@68t; leastcov 66.2 +/- 5.6 / 9.70 / 22.9; occlusion 69.4 +/- 6.0 / 10.47 / 22.0; random 113.0 +/- 9.8 / 15.40 / 12.8.
- Counted OcclusionPlanner pushes directly this turn: 11 pushes across all 180 episodes, only in the 25-item cells (1 at clutter 0.3, 10 at clutter 0.9); none from the new declutter rule.
- Verified `compare.py` output is byte-identical across two runs (deterministic). Removed temp files.

## 2026-10-24 - Add session handoff notes

### Changes
- Added `HANDOFF.md`, a repository handoff summary for picking the project up in a fresh session.

### Files Modified
- `HANDOFF.md` — new document: project purpose, git/test status, module-by-module summary of the simulation core (scene, robot, sim, planner, compare), real benchmark results so far, steering conventions, suggested next build steps, setup commands, and watch-outs.

### Technical Details
- Documentation only; no source code, tests, or behavior changed.
- Captures current state: HEAD at commit fb8861e, clean tree, 37 tests passing, build steps 1-3 done (scene/robot/planners) with log.py, run_eval.py, render.py, and the RL stretch still to build.
- File is untracked; committing it is left to the user.

### Testing
- No tests were run this turn (documentation-only change). The 37-passing status quoted in HANDOFF.md reflects the previous turn's `pytest -q` run, not a run this turn.

## 2026-10-24 - Add dataset logger and wire it through the episode runner

### Changes
- Added `src/log.py`, the dataset layer that records each episode as frames and writes them to disk in the BridgeData V2 / DROID shape, so simulation runs double as a reusable robotics dataset (build order step 4).
- Wired the previously-unused `logger` parameter through `src/sim.run_episode` so one frame is logged per gripper action when a logger is supplied.
- Added `tests/test_log.py` covering the schema, frame/reward/done semantics, dataset layout, and determinism.

### Files Modified
- `src/log.py` — new module: `Frame` dataclass, `EpisodeLogger(task)` with `log_frame(...)` and `to_frame()`, `write_dataset(loggers, out_dir="out")`, and the `FRAME_COLUMNS` / `EPISODE_COLUMNS` schema constants.
- `src/sim.py` — `run_episode` now logs a frame per action (grasp and push) via the `logger`; updated the docstring to describe the logging behavior. Grasp branch captures the success flag so reward can be recorded.
- `tests/test_log.py` — new test file (8 tests).

### Technical Details
- Dataset layout under `out_dir`: `dataset/meta/episodes.csv` with columns `episode,task,frames` (matching `robotics/bridge-v2/data.csv`), and one `dataset/data/episode_{index:06d}.csv` per episode with columns `frame_index,timestamp,state_x,state_y,holding,action_type,action_x,action_y,reward,done`.
- One frame is logged per action, recording the post-travel gripper state. `timestamp` is `tick / TICK_HZ` seconds (15 Hz). `holding` of `None` is stored as -1. For a grasp, `action_x/action_y` is the item center and `reward` is 1.0 on success else 0.0; for a push, they are the push delta `(dx, dy)` and `reward` is 0.0.
- `to_frame()` sets `done=True` only on the last frame and returns a schema-columned DataFrame (empty but still columned when no frames were logged).
- `write_dataset` creates `meta/` and `data/` as needed, indexes episodes by list position, and records each episode's frame count in the meta file. Output goes under `out/` per the generated-files convention; `out/` is gitignored.
- Core stays headless: `log.py` imports only `pandas`, `pathlib`, and `src.robot.TICK_HZ`; no pygame. Passing no logger leaves `run_episode` behavior and `compare.py` output unchanged.

### Testing
- Ran `.venv/bin/python -m pytest -q` this turn: 45 passed in 0.32s (37 prior + 8 new in `test_log.py`).
- Ran `python src/compare.py` this turn to confirm no regression: OcclusionPlanner still beats NearestPlanner at 15+ items (clutter 0.9, 25 items: nearest 164.5 +/- 17.1 ticks / 34.50 failed vs occlusion 69.4 +/- 6.0 / 10.47), matching the prior handoff numbers.
- Ran an end-to-end smoke write this turn (3 episodes, OcclusionPlanner) into `out/dataset/`: `meta/episodes.csv` produced the `episode,task,frames` rows and each `episode_{i:06d}.csv` carried the full 10-column schema.
- Determinism test (`test_write_dataset_is_deterministic`) confirms the same seed produces byte-identical episode data files.

## 2026-10-24 - Add run_eval.py evaluation sweeps, plots, and results.md

### Changes
- Added `src/run_eval.py`, the evaluation driver (build order step 5): runs the five planners across randomized desks, writes four figures to `out/plots/` and a markdown table file to `out/results.md`, all from real `run_episode` numbers.
- Added `tests/test_run_eval.py` with a `--quick` smoke test plus constant-override and determinism tests.

### Files Modified
- `src/run_eval.py` — new module. CLI `--seeds N` (default 30) and `--quick` (5 seeds, item counts trimmed to 5/15). Builds four artifacts: ticks-to-clear vs n_items and failed-grasps vs n_items (both two-panel for clutter 0.3/0.9, all five planners, 95% interval error bars); a sensitivity grouped-bar plot at 15 items / clutter 0.9 sweeping FAILED_GRASP_TICKS in {1,3,5} x CLEARANCE_PENALTY in {1.0,0.7,0.5} over Occlusion/LeastCovered/NearestRetry; and a push ablation (OcclusionPlanner pushes on vs off) at 25 items / clutter 0.9. Writes `out/results.md` as markdown tables.
- `tests/test_run_eval.py` — new test file (3 tests): quick run writes all four PNGs and results.md non-empty with expected headings; `override_constants` applies then restores globals on all three modules; two quick runs produce byte-identical results.md.

### Technical Details
- Headless plotting: `matplotlib.use("Agg")` is set before importing `pyplot`, so no display and no GUI/pygame dependency. Verified at runtime that the backend is Agg and `pygame` is not in `sys.modules` after import.
- Constants are overridden by argument, not by editing globals: `override_constants(failed_grasp_ticks, clearance_penalty)` is a context manager that rebinds `FAILED_GRASP_TICKS` on `src.robot`, `src.sim`, and `src.planner` (all three imported it by name) and `CLEARANCE_PENALTY` on `src.robot` and `src.planner` (where it is actually read), then restores every saved attribute on exit. This matches how the sim and planner reference those names, so the overrides take effect without touching source globals.
- Fresh scene and fresh planner per seed in `evaluate`, since scenes and the occlusion/retry planners carry mutable state that reuse would corrupt.
- Push ablation uses a `NoPushPlanner` wrapper that delegates to an inner planner and converts any returned `Push` into a `Grasp` of the same item id, leaving grasps and `None` untouched. Chosen over forking the planner so the comparison isolates exactly the push behavior.
- `results.md` is generated entirely from the swept `Stat` means/intervals (no hand-entered numbers) and includes a computed note on whether the sensitivity winner is stable across settings.
- Statistics mirror `compare.py`: mean and 1.96*std/sqrt(n) with sample std (n-1), 0 for n<2.

### Testing
- Ran `.venv/bin/python -m pytest -q` this turn: 48 passed in 1.75s (45 prior + 3 new).
- Ran `python src/run_eval.py --quick` this turn: wrote `out/plots/{ticks_vs_items,failed_vs_items,sensitivity,ablation}.png` and `out/results.md`; exited cleanly.
- Real numbers from the quick run (5 seeds, wide intervals): clutter 0.9, 15 items ticks — nearest 72.6 +/- 39.0, leastcov 39.0 +/- 12.8, occlusion 38.6 +/- 13.7, nearretry 49.4 +/- 17.0.
- Occlusion vs LeastCovered: occlusion wins outright in the harsher-penalty sensitivity rows (e.g. ft=5/cp=0.5: occlusion 55.2 +/- 4.0 vs leastcov 91.8 +/- 26.2); tie at default ft=3/cp=0.7; leastcov wins when cp=1.0. Ranking does not hold across all settings — by design, occlusion's edge appears when crowding is penalized.
- Pushes do fire but rarely: ablation pushes_on averaged 0.60 +/- 0.78 pushes/episode at 25 items/clutter 0.9 (pushes_off 0.00); disabling pushes cost ~4 ticks and ~1 extra failed grasp per episode at this sample size.
- Confirmed Agg backend and no pygame import at runtime this turn.

## 2026-10-24 - Add render.py demo renderer (side-by-side pygame video)

### Changes
- Added `src/render.py`, the demo renderer (build order step 6) and the only module that imports pygame. It replays planners on one seed and writes an animated side-by-side video.
- Added `tests/test_render.py` with a replay-correctness check and a headless smoke render.

### Files Modified
- `src/render.py` — new module. CLI: `--seed`, `--n-items 15`, `--clutter 0.9`, `--planners random,occlusion`, `--out out/demo.mp4`, `--headless`. Runs each named planner on the same seed and draws them in one row: desk, items as colored rectangles with letter labels (P/K/M/T) in z-order, labelled container boxes at the right edge, gripper as a circle, and a per-panel status line (planner, tick, sorted, failed, pushes). Animates each tick with interpolated gripper motion and highlights grasp (green ring), failed grasp (red item flash), and push (amber ring). Encodes with imageio to mp4, falling back to gif on failure.
- `tests/test_render.py` — new test file (2 tests): `build_trace` reproduces the sim's sorted/failed/push counts for a seed; a headless render of a small scene writes a non-empty `.mp4`-or-`.gif` file.

### Technical Details
- Keeps pygame out of the headless core: pygame is imported locally inside `_render_pygame`/`_draw_panel_pygame`, never at module top level. `--headless` sets `SDL_VIDEODRIVER=dummy` via `os.environ.setdefault` before `pygame.init()`, so no window opens.
- Replay fidelity (honesty rule): `build_trace` mirrors `run_episode` step for step — same `numpy.random.default_rng(seed)` draw order, same `effective_grasp_prob` grasp model, same `push` primitive, same tick accounting (travel ticks by distance, failed grasp adds `FAILED_GRASP_TICKS`). The animation therefore reproduces the real episode outcome rather than inventing one; `test_build_trace_matches_sim_outcome` asserts the final counters equal the sim's.
- Side-by-side sync: `_frame_pairs` yields `FRAMES_PER_TICK` (2) interpolation steps between tick frames and holds the final frame when one planner's run is shorter, so both panels share a video length.
- pygame surface is converted to RGB numpy via `pygame.surfarray.array3d` + transpose (pygame is width-major, numpy row-major).
- Fallback path: if pygame cannot be imported, `_render_matplotlib` (Agg backend) draws the same logical content (desk, containers, labelled items, gripper, status title) so the demo still renders without pygame.
- Encoding: `write_video` tries the requested path (mp4) first; on any encoder error it logs and retries with a `.gif` of the same stem. mp4 uses `fps`; the gif (pillow) writer takes per-frame `duration` in ms (`1000/FPS`), which also cleared an imageio `fps`-deprecation warning.

### Testing
- Ran `.venv/bin/python -m pytest -q` this turn: 50 passed (48 prior + 2 new); warnings dropped from 3 to 2 after switching the gif writer to `duration` (remaining warnings are unrelated imageio plugin notices).
- Ran `python src/render.py --headless --seed 3 --out out/demo.mp4` this turn: used pygame via the SDL dummy driver; mp4 encoding failed with ValueError because this environment has no ffmpeg backend (no `imageio_ffmpeg` installed), so it fell back to `out/demo.gif` at 122,610 bytes (~120 KB).
- Verified the rendered frame visually: both `random` and `occlusion` panels on seed 3 show the desk, z-ordered labelled items, right-edge containers, the gripper circle, and the status line.
- Note: mp4 output requires an ffmpeg backend that is not installed here; `.mp4` requests always fall back to `.gif` in this environment by design. Installing `imageio-ffmpeg` would enable mp4 (a new dependency, not added without asking).

## 2026-10-24 - Interval-aware winner column in the sensitivity table

### Changes
- Changed the sensitivity table in `src/run_eval.py` so the `winner` column reports `tie` when the 95% intervals of the two best planners overlap, instead of always naming the lowest mean.
- Replaced the single "Best planner is leastcov in the first setting..." sentence with a per-planner count of clear wins plus the number of ties across all settings.

### Files Modified
- `src/run_eval.py` — added `_sensitivity_winner(row, names)` helper; rewired the results-markdown loop to use it, tally clear wins per planner and ties, and emit the new summary line; updated the column-description sentence to explain the tie rule.

### Technical Details
- Winner rule: planners are ordered by mean ticks-to-clear (lower is better). The best and second-best are declared a `tie` when their 95% intervals overlap, i.e. `best.mean + best.ci >= second.mean - second.ci` (the best already has the smaller mean). Otherwise the lowest-mean planner's name is returned.
- Summary line now reads, e.g., "Across 9 settings: clear wins by planner (occlusion 2, leastcov 2, nearretry 0); ties 5." Counts are computed from the same per-row decision, so the table and summary cannot disagree.
- Only the sensitivity section changed; the ticks/failed tables and the ablation table are untouched. All numbers still come from the real sweep.

### Testing
- Ran `python src/run_eval.py` this turn (full 30-seed sweep): regenerated `out/results.md`. The winner column now shows `tie` in 5 of 9 settings and names a planner where intervals separate (e.g. ft=3/cp=0.5 -> occlusion 52.9 +/- 3.8 vs leastcov 65.3 +/- 6.9; ft=3/cp=1.0 -> leastcov 17.4 +/- 0.2 vs occlusion 19.3 +/- 1.4). Summary line: "Across 9 settings: clear wins by planner (occlusion 2, leastcov 2, nearretry 0); ties 5."
- Ran `.venv/bin/python -m pytest -q` this turn: 50 passed, 2 warnings.

## 2026-10-24 - Remove stray virtualenvs (# and Intel) and gitignore them

### Changes
- Removed two accidental virtualenvs committed at the repo root, `#/` and `Intel/`, from both version control and disk. They were leftover `python3.12 -m venv` environments (confirmed via each `pyvenv.cfg`), not project code; the real environment is `.venv/`.
- Added ignore rules so a future stray venv with either name is not re-added.

### Files Modified
- `.gitignore` — added `/#/` and `/Intel/` under a comment noting they are stray virtualenvs.
- `#/` and `Intel/` — deleted. 982 tracked files staged as deletions via `git rm -r`; the remaining untracked `.pyc` caches (406 per dir) were removed with `rm -rf`, and the directory shells are gone from disk.

### Technical Details
- Used `git rm -r '#' Intel` to stage the deletions, which removed the tracked files but left untracked bytecode caches behind; those were cleaned up with `rm -rf '#' Intel`.
- Anchored the ignore patterns to the repo root (`/#/`, `/Intel/`) so they match only these top-level dirs, not any nested path.
- This removes the venvs going forward only; they remain in prior commit history. Purging them from history would require a destructive rewrite (git filter-repo + force-push) and was intentionally NOT done.
- Verified after the change: 0 files remain tracked under either path, both directories are gone from disk, and `git check-ignore` confirms `#/` and `Intel/` are now ignored.
- Not committed or staged by me beyond the `git rm` staging inherent to the removal; `.venv/` and the project source are untouched. No commit was made.

### Testing
- No code tests were run this turn (no source code changed). Verification was limited to git/filesystem checks described above: staged deletions = 982, tracked files under the paths = 0, both dirs absent from disk, and both paths reported ignored by `git check-ignore`.

## 2026-10-24 - Add requirements.txt, make src a package, write project README

### Changes
- Added `requirements.txt` at the repo root listing the runtime and test dependencies.
- Added an empty `src/__init__.py` so `src` is a proper importable package.
- Replaced the root `README.md` with a project README (problem, approach, dataset format, results, limitations, run instructions, structure). Did NOT touch `robotics/README.md` (the hackathon kit file).

### Files Modified
- `requirements.txt` — new. numpy, pandas, matplotlib, imageio, imageio-ffmpeg, pygame, pytest. No version pins (none needed for tests to pass). Used `pygame` (not `pygame-ce`) because `.venv/bin/pip list` shows `pygame` 2.6.1 installed.
- `src/__init__.py` — new, empty; marks `src` as a package.
- `README.md` — rewritten. All result numbers copied verbatim from `out/results.md` (no invented or rounded figures); plots and `out/demo.gif` embedded with relative paths; only commands verified to exist are listed.

### Technical Details
- conftest.py: left in place but is now redundant. With `src/__init__.py` present, the suite passes without conftest.py (verified by moving it aside and rerunning, then restoring). Not deleted, per instruction.
- README applies the "different only when 95% intervals don't overlap" rule throughout. Key figures and their `out/results.md` sources: nearest 164.5 ± 17.1 ticks / 34.50 ± 4.41 failed, occlusion 69.4 ± 6.0 / 10.47 ± 1.50, leastcov 66.2 ± 5.6 / 9.70 ± 1.39 (clutter 0.9, 25-item columns); default tie occlusion 40.2 ± 5.0 vs leastcov 40.1 ± 5.0 (sensitivity row 3/0.7); sensitivity summary occlusion 2 / leastcov 2 / ties 5 of 9; ablation pushes_on 69.4 ± 6.0 / 10.47 ± 1.50 / 0.33 ± 0.22 and pushes_off 69.7 ± 5.5 / 10.60 ± 1.38 / 0.00.
- Dataset-format section documents the `src/log.py` columns (meta: episode/task/frames; data: frame_index, timestamp, state_x/y, holding, action_type, action_x/y, reward, done).
- Verified the CLI commands via `--help` (`run_eval.py`, `render.py`) and confirmed `compare.py` takes no arguments before listing them in the README.

### Testing
- Ran `.venv/bin/python -m pytest -q` this turn: 50 passed, 2 warnings.
- Ran the same suite with conftest.py temporarily moved aside this turn: 50 passed (confirming conftest.py is no longer required once `src/__init__.py` exists); conftest.py was restored.
- Ran `.venv/bin/pip install -r requirements.txt --dry-run` this turn: resolves cleanly; all satisfied except `imageio-ffmpeg-0.6.0`, which pip reports it "would install".
- Confirmed all README-referenced artifacts exist: `out/plots/{ticks_vs_items,failed_vs_items,sensitivity,ablation}.png` and `out/demo.gif`.

## 2026-10-24 - Point README images at docs/ instead of gitignored out/

### Changes
- Repointed the five embedded image links in `README.md` from `out/plots/*.png` and `out/demo.gif` to the committed copies under `docs/`, so the plots and demo render on GitHub (the `out/` directory is gitignored and would not display).

### Files Modified
- `README.md` — changed the five image paths: `ticks_vs_items.png`, `failed_vs_items.png`, `sensitivity.png`, `ablation.png` now resolve under `docs/`, and `demo.gif` points to `docs/demo.gif`. No other content changed; the Results section still notes `out/` is generated and gitignored.

### Technical Details
- The `docs/` folder (pre-existing this turn, not created by me) holds copies of the four plots and the demo gif. Image filenames there are flat (`docs/sensitivity.png`, etc.), unlike the `out/plots/` subpath, so the links were adjusted accordingly.
- Only Markdown image references changed; all result numbers and prose are untouched.

### Testing
- No code tests were run this turn (documentation-only change). Verified each of the five `docs/` image paths referenced in README.md resolves to an existing file on disk (all five present).

## 2026-10-24 - Add Messy Desk Lab Streamlit app design spec (design only)

### Changes
- Added a design-only spec for "Messy Desk Lab", a four-section Streamlit app (Experiment, Live View, Results, Trial Explorer) layered over the existing simulation engine. No source code or simulation behavior was changed this turn.

### Files Modified
- `.kiro/specs/messy-desk-lab/design.md` — new technical design document (high-level + low-level).
- `.kiro/specs/messy-desk-lab/.config.kiro` — new spec config (design-first feature).

### Technical Details
- Documented the minimal engine interface surface the app needs, cited per module (scene, robot, planner, sim, log, compare, run_eval, render).
- Gap analysis: EpisodeResult carries only aggregate counts (no per-step scrub), so the design proposes promoting trace structures + build_trace stepping logic out of the pygame-bound render.py into a headless core (new src/trace.py or extended sim.py) with a single shared stepping path so sim and trace cannot drift.
- Confirmed deterministic replay is possible from (planner_name, seed, n_items, clutter) alone, provided a fresh planner and fresh scene are rebuilt each replay (flagged the instance-reuse risk for stateful planners).
- Recommended an optional, behavior-preserving planner.last_reason attribute (read after next_action) for action rationale, over widening the next_action return signature.
- Deployment: separate app/requirements.txt without pygame/imageio/pytest, matplotlib Agg before pyplot, headless frames for Live View, and a synchronous-capped + cached batch approach under Streamlit's single-threaded rerun model.
- All proposed engine changes flagged as proposed, optional, and behavior-preserving with determinism/parity test requirements.

### Testing
- No tests were run this turn (design document only; no code changed). `validate_spec_format` on design.md reported no errors, only expected design-first warnings about properties lacking requirement references.

## 2026-10-24 - Remove stale streamlit-demo spec

### Changes
- Deleted the superseded `streamlit-demo` spec. It was an earlier three-tab demo design replaced by the broader four-section `messy-desk-lab` spec. No source code or simulation behavior changed.

### Files Modified
- `.kiro/specs/streamlit-demo/design.md` — deleted (stale design document).
- `.kiro/specs/streamlit-demo/.config.kiro` — deleted (stale spec config).

### Technical Details
- The `messy-desk-lab` spec is the active successor and is the four-section superset (Experiment, Live View, Results, Trial Explorer) of the removed three-tab demo (Run, Compare, About).
- Noted to the user that `messy-desk-lab/.config.kiro` currently reuses the same specId (`61807b9b-...`) that the deleted spec carried; offered to regenerate a unique ID. No ID change was made this turn.

### Testing
- No tests were run this turn (spec file deletion only; no code changed).

## 2026-10-24 - Add Messy Desk Lab requirements and implementation plan

### Changes
- Added the requirements document and the implementation task list for the "messy-desk-lab" spec, derived from the existing design (design-first workflow). No source code or simulation behavior changed.

### Files Modified
- `.kiro/specs/messy-desk-lab/requirements.md` — new EARS-format requirements (13 requirements with Introduction and Glossary).
- `.kiro/specs/messy-desk-lab/tasks.md` — new implementation plan ordered for earliest deployability.

### Technical Details
- Requirements cover the four app sections (Experiment, Live View, Results, Trial Explorer) plus cross-cutting decisions: deterministic replay from a minimal TrialConfig, the UI->logic->core boundary, the two proposed behavior-preserving engine changes (shared Trace_Core, optional planner.last_reason), CI reuse (Z_95=1.96, sample std n-1, 0.0 for n<2), Community Cloud deployment, and project constraints (robotics/ read-only, output under out/, no fabricated results, no tuning).
- Tasks follow a user-specified deployability-first sequence: Stage 1 headless logic layer + CI stats + tests; Stage 2 minimal Streamlit shell (Experiment + Results) + app/requirements.txt without pygame/imageio/pytest; Stage 3 deploy + manual public-URL verification; Stage 4 Trace_Core promotion (src/trace.py, render.py re-exports) + Live View; Stage 5 planner rationale; Stage 6 Trial Explorer; Stage 7 batch cap + error polish.
- Both engine touches (Stages 4 and 5) are flagged behavior-preserving, each gated by a determinism/parity test proving no EpisodeResult changes. Every code task is required to leave pytest -q green. A Task Dependency Graph places same-file tasks in separate waves.
- Note: the requirements-detailing refinement pass from the prior turn was cancelled, so requirements.md holds the original (complete) acceptance criteria; the task breakdown traces cleanly to them.

### Testing
- No tests were run this turn (planning documents only; no code changed). Both documents passed `validate_spec_format` with no issues.

## 2026-10-24 - Messy Desk Lab Stage 1: headless logic layer + tests

### Changes
- Added the GUI-independent logic layer for Messy Desk Lab (`app/logic.py`) and its unit tests, implementing the experiment batch runner, confidence-interval statistics, input clamping, and an output-location guard. No simulation behavior in `src/` was changed.

### Files Modified
- `app/__init__.py` — new app package marker.
- `app/logic.py` — new headless logic layer (data models, planner registry, batch runner, CI stats, output guard).
- `tests/test_logic.py` — new unit tests for the logic layer.
- `.kiro/specs/messy-desk-lab/tasks.md` — marked Stage 1 (task 1 and sub-tasks 1.1–1.4) complete.

### Technical Details
- `app/logic.py` imports only `src/`, numpy, and the standard library — never streamlit, pygame, or render (verified via sys.modules inspection and static grep).
- Data models: `TrialConfig` (exactly planner_name, seed, n_items, clutter), `ExperimentConfig`, `TrialRow`, `Stat`, `PlannerStat`, `ExperimentResult`.
- Planner registry mirrors render's factories WITHOUT importing render; `make_planner` builds a fresh instance each call and raises `ValueError` (listing valid names) for unknowns. `run_experiment` builds a fresh scene and fresh planner per trial via `src.sim.run_episode`, so mutable planner state never leaks between trials.
- CI math mirrors `src/compare._ci95` and `src/run_eval`: `Z_95=1.96`, sample std with n-1, `0.0` for n<2; a test cross-checks `ci95` against `src.compare._ci95`. `overlap` matches the run_eval tie rule.
- `build_scene` clamps clutter to [0,1] and n_items to a documented [0,30] range before delegating to `make_scene`.
- Output guard `resolve_output_path` rejects relative paths whose first segment is a read-only input tree (`data/`, `robotics/`) and any path that resolves outside `out/`. During development the guard raised a correct-but-differently-worded error for `data/` paths than the test regex expected; fixed by making the read-only-tree rejection explicit and tightening the test assertions.
- No dedicated file-writing tool was available, so files were created via shell heredocs and read back to verify.

### Testing
- Ran `.venv/bin/python -m pytest -q` (Python 3.12.15, pytest 9.1.1): 67 passed, 2 warnings. The 2 warnings are pre-existing imageio deprecations in `tests/test_render.py`, unrelated to this change.
- Verified the import boundary independently: importing `app.logic` leaves streamlit/pygame/src.render absent from `sys.modules`, and a static grep finds no forbidden imports.
- Confirmed via `git status` that no `src/` file was modified.

## 2026-10-24 - Messy Desk Lab Stage 2: minimal Streamlit app (Experiment + Results)

### Changes
- Added the minimal runnable Streamlit app with Experiment and Results sections, a headless app-level requirements file, and static import-boundary tests. No simulation behavior in `src/` was changed.

### Files Modified
- `app/streamlit_app.py` — new UI-only Streamlit module (Experiment controls + Results table).
- `app/requirements.txt` — new app/deploy dependency list (streamlit, numpy, pandas, matplotlib, pillow; excludes pygame/imageio/pytest).
- `tests/test_streamlit_app.py` — new static (AST) boundary tests for the UI module.
- `.kiro/specs/messy-desk-lab/tasks.md` — marked Stage 2 (task 2 and sub-tasks 2.1–2.3) complete.

### Technical Details
- `app/streamlit_app.py` imports only streamlit, app.logic, matplotlib (to force the backend), and pandas — no src/ math, pygame, or render. All computation goes through app.logic.
- Forces `matplotlib.use("Agg")` at import time, before anything can import pyplot, so the app stays headless on Streamlit Community Cloud.
- Experiment section: sidebar controls for n_items (3–30), clutter (0–1), trial count, base seed, and a planner multiselect from `logic.available_planners()`; a Run button calls `logic.run_experiment` inside `st.spinner` and caches the ExperimentResult in `st.session_state`. A try/except surfaces run failures via `st.error`.
- Results section: per-planner table of `mean +/- 95% CI` for ticks/items sorted/failed grasps/pushes using the Stage 1 `ci95`; overlapping intervals labeled "tie" via `logic.overlap`, otherwise the lower planner is named. Real batch numbers only; accessibility via text labels (not colour) and explanatory captions.
- Tests use `ast` source inspection rather than importing the module, because streamlit is an app/deploy dependency and is intentionally not installed in the test environment. During development the boundary test did not recognize `from app import logic` as `app.logic`; fixed by having the AST helper also record `pkg.name` forms.

### Testing
- Ran `.venv/bin/python -m pytest -q` (Python 3.12.15, pytest 9.1.1): 70 passed, 2 warnings. The 2 warnings are pre-existing imageio deprecations in `tests/test_render.py`, unrelated to this change.
- Verified `app/streamlit_app.py` imports no src/pygame/render (static grep) and confirmed via `git diff --stat` that no `src/` file was modified.

## 2026-10-24 - Fix Streamlit Cloud import of app/src modules

### Changes
- Fixed `app/streamlit_app.py` so `from app import logic` (and logic.py's own `import src.*`) resolves both locally and on Streamlit Community Cloud, which runs the script from a different working directory and otherwise raised `ModuleNotFoundError: No module named 'app'`. No simulation behavior or `src/` file was changed.

### Files Modified
- `app/streamlit_app.py` — prepend a sys.path bootstrap that adds the repo root before importing logic.

### Technical Details
- Added, immediately after `from __future__ import annotations` and before the matplotlib/logic imports, a bootstrap that computes the repo root as the parent of this file's `app/` directory via `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` and inserts it at the front of `sys.path` (guarded by an `if not in` check).
- Kept `from app import logic` rather than switching to `import logic`: a single repo-root entry resolves both `app` (for the UI import) and `src` (for logic.py's `import src.*`), since both packages live under the repo root. Using `abspath(__file__)` makes it independent of cwd.
- `import os` / `import sys` are stdlib, so the UI import-boundary test (no src/pygame/render) still holds.

### Testing
- Ran `.venv/bin/python -m pytest -q` (Python 3.12.15, pytest 9.1.1): 70 passed, 2 warnings. The 2 warnings are pre-existing imageio deprecations in `tests/test_render.py`, unrelated to this change.
- Simulated the Cloud scenario: loaded `app/streamlit_app.py` by file path from `/tmp` with no PYTHONPATH (streamlit/matplotlib/pandas stubbed); the module loaded and both `src` and `src.sim` resolved — the case that previously failed. Did not launch a live browser server (manual step; streamlit not installed in the venv).
- Confirmed `git diff --stat -- src/` shows no changes.

## 2026-10-24 - Messy Desk Lab Stage 4: trace core promotion + Live View

### Changes
- Promoted the per-tick trace structures and stepping logic out of `src/render.py` into a new headless `src/trace.py` (behavior-preserving), with `render.py` re-exporting the moved names so its public API is unchanged. Added a deterministic replay + headless frame renderer to `app/logic.py` and a Live View section to the Streamlit app. No simulation behavior in `src/sim.py`, `src/robot.py`, `src/scene.py`, or `src/planner.py` was changed.

### Files Modified
- `src/trace.py` — new headless trace core (ItemState, ContainerState, TickFrame, Trace, PLANNER_FACTORIES, build_trace), moved from render.py; no pygame import.
- `src/render.py` — removed the moved definitions; now re-exports them from `src.trace` (13 insertions, 168 deletions).
- `app/logic.py` — added ReplayFrame/ReplayResult/ContainerBox/ItemBox/FrameImage, `replay_trial`, `draw_frame` (matplotlib Agg), and `render_frames`.
- `app/streamlit_app.py` — added the Live View section (trial picker, cached render, play/pause/step/speed, live counters).
- `tests/test_trace.py` — new parity/determinism tests for the trace core.
- `tests/test_logic.py` — added replay/render tests.
- `.kiro/specs/messy-desk-lab/tasks.md` — marked Stage 4 (task 4 and sub-tasks 4.1–4.5) complete.

### Technical Details
- The trace block (dataclasses + `build_trace`) was moved verbatim into `src/trace.py` so the stepping logic lives in the headless core; `render.py` re-exports `PLANNER_FACTORIES`, `ItemState`, `ContainerState`, `TickFrame`, `Trace`, `build_trace` (verified to be the same objects), so `from src.render import build_trace, render` still works and the existing render parity test still passes.
- Deliberate decision: `run_episode` was NOT rewired to route through the trace core, to avoid any risk of changing simulation behavior (forbidden by the plan). Instead `build_trace` is the single trace implementation and a parity test locks it to `run_episode`'s output across 5 seeds × 5 planners × 3 configs. Flagged to the user that a true shared stepping generator touching sim.py would need sign-off.
- `app/logic.py` `replay_trial` rebuilds a fresh scene + fresh planner from TrialConfig, builds the trace via `src.trace` (never render), and pairs it with a direct `run_episode` result. `draw_frame` uses matplotlib Agg and encodes each kind by letter + colour + edge style (colour never the only cue); `render_frames` renders once with per-frame alt text. The app imports no streamlit/pygame/render (verified via sys.modules); the one grep hit is the word "render.py" in a docstring.
- Live View renders frames once and caches them keyed by TrialConfig in `st.session_state`, so scrubbing indexes into the list; play advances by the speed value and reruns until the end; empty-desk trials show a "no items" message.

### Testing
- Ran `.venv/bin/python -m pytest -q` (Python 3.12.15, pytest 9.1.1): 108 passed, 2 warnings (up from 70; the parity sweep and replay/render tests added the rest). The 2 warnings are pre-existing imageio deprecations in `tests/test_render.py`, unrelated to this change.
- Confirmed via `git diff --quiet` that `src/sim.py`, `src/robot.py`, `src/scene.py`, and `src/planner.py` are unchanged (no grasp/tick/constant logic touched).
- Verified `app/logic.py` loads none of streamlit/pygame/src.render into sys.modules, and `app/streamlit_app.py` has no src/pygame/render imports.

## 2026-10-24 - Live View renderer: recognizable item glyphs + two-jaw gripper

### Changes
- Improved the Live View headless frame renderer so items read as everyday desk objects (pen/key/mug/trash) and the gripper reads as a two-jaw claw. Presentation only — no simulation behavior, geometry, coverage, z-order, or any `src/` file was changed; all EpisodeResult values stay identical.

### Files Modified
- `app/logic.py` — rewrote item and gripper drawing in `draw_frame`; added `_draw_item_glyph` and `_draw_gripper` helpers and a per-kind accent palette.
- `app/streamlit_app.py` — added the Live View caption "Top-down view of a kinematic sorting simulation (no 3D physics or real hardware)."

### Technical Details
- Glyphs are matplotlib-drawn shapes (no emoji/unicode/fonts, so they render reliably on a headless Cloud container): pen = rounded body + nib along the item's longer axis; key = round bow with hole + blade with two teeth; mug = rounded square + handle arc; trash = crumpled polygon blob with scribble strokes.
- Each glyph is scaled to fit strictly within the item's existing bounding box `(s.w, s.h)`, and a faint outline of the true box is kept so the real sim footprint stays visible. The accessible kind label (P/K/M/T) is retained at the box corner as a fallback; distinct per-kind colours are preserved; the fail highlight still recolours only `event_item`.
- The trash blob's vertices are jittered by a per-item `numpy.random.default_rng(s.id)` created locally in the renderer — it does not touch or share the simulation's rng, so sim determinism is unaffected.
- The gripper is now two parallel jaw rectangles on a base bar; jaws close in and highlight on a grasp, keeping the existing grasp/fail/push/idle ring colours.
- The renderer still forces `matplotlib.use("Agg")` before importing pyplot (verified Agg precedes the pyplot import) and imports no pygame or render.

### Testing
- Ran `.venv/bin/python -m pytest -q` (Python 3.12.15, pytest 9.1.1): 108 passed, 2 warnings — unchanged count, confirming no behavior changed. The 2 warnings are pre-existing imageio deprecations in `tests/test_render.py`.
- Smoke-rendered a full 25-frame replay (occlusion, seed 3, 20 items, clutter 0.9) exercising move/grasp/fail events across all glyph kinds with no errors; sample frame shape (320, 420, 3) uint8.
- Confirmed via `git status`/`git diff -- src/` that no `src/` file changed, and via runtime sys.modules check that importing `app.logic` loads no pygame.

## 2026-10-24 - Live View renderer: full-size desk, bigger glyphs, parallel-jaw gripper

### Changes
- Refined the Live View headless renderer so the scene fills the frame, item glyphs fill their boxes, pens no longer look like arrows, and the gripper reads as a parallel-jaw claw. Presentation only — no simulation behavior or geometry changed; no `src/` file touched; pytest stays at 108.

### Files Modified
- `app/logic.py` — reworked `draw_frame` figure sizing/axes, enlarged the item glyphs, redesigned the pen glyph, and rewrote `_draw_gripper`.

### Technical Details
- `draw_frame` now defaults to 640x480 (the desk's 4:3 aspect) at dpi=150, sets axes to exactly the desk bounds (xlim 0-640, ylim 480-0) with `set_aspect("equal")` and a full-bleed `add_axes([0,0,1,1])`, so there is no empty margin shrinking the scene. The returned array remains exactly (height_px, width_px, 3) — verified 480x640 default and 240x320 for the custom-size test.
- Glyph scales were enlarged so each glyph roughly fills its real bounding box (pen body ~0.9 of the long axis; key bow/blade larger; mug body ~0.86; trash blob ~0.92 of the half-extent).
- Pen glyph redesigned: replaced the large nib triangle (which read as an arrowhead) with a small pointed tip plus a short clip line, so it reads as a pen barrel.
- `_draw_gripper` rewritten as a top-down parallel-jaw claw: a rectangular hub with two thick parallel fingers and a visible gap, scaled to a typical item width (`_GRIPPER_SPAN=40px`), with a dark outline and higher zorder so it stands out over items. The finger gap closes on a grasp; grasp/fail/push/idle keep the existing green/red/amber/pale colours. Item colours and P/K/M/T labels are preserved.

### Testing
- Ran `.venv/bin/python -m pytest -q` (Python 3.12.15, pytest 9.1.1): 108 passed, 2 warnings — unchanged, confirming no behavior changed. The 2 warnings are pre-existing imageio deprecations in `tests/test_render.py`.
- Rendered and visually inspected preview frames (written under out/, gitignored): a sparse desk shows the two-finger claw clearly and pens reading as pens; frame shapes confirmed 640x480 (default) and 240x320 (custom).
- Confirmed via `git diff --stat`/`git status -- src/` that no `src/` file changed, and via runtime sys.modules check that importing `app.logic` loads no pygame.

## 2026-10-24 - SPIKE (experiment-shapes branch): rotated-rectangle item footprints

### Changes
- Feasibility spike (not a finished feature) proving items can be modeled as rotated rectangles instead of axis-aligned boxes while keeping overlap/occlusion/coverage/crowding coherent. All changes are on the throwaway `experiment-shapes` branch. The default (axis-aligned) path is behavior-preserving: no existing sim result changes.

### Files Modified
- `src/scene.py` — added `Item.angle` + `Item.polygon()`; rewrote `_overlap_area` as convex polygon intersection; made `_edge_gap` polygon-aware; added opt-in `make_scene(rotate=False)`.
- `src/trace.py` — `ItemState` now carries `angle` so the renderer can see rotation.
- `app/logic.py` — `ItemBox` carries `angle`; `_draw_item_glyph` draws the true rotated footprint for rotated items.

### Technical Details
- Geometry is encapsulated in scene.py behind `_overlap_area`, `covered_fraction`, `_edge_gap`/`crowding_neighbors`, `occluders`. robot.py and planner.py call only the named functions, so swapping internals to polygons required ZERO planner changes — the key feasibility finding.
- `_overlap_area` uses Sutherland-Hodgman convex clipping + shoelace area with an AABB broad-phase reject; at angle 0 it equals the old AABB overlap exactly (verified 400, 1600, 0 on the hand-built cases). `_edge_gap` keeps the exact old AABB gap when both items are axis-aligned and falls back to a segment-to-segment polygon gap when either is rotated.
- `make_scene` gained `rotate=False`; when False, NO extra rng draw is made, so the default draw order and every existing scene/sim result are byte-for-byte unchanged. Angle is drawn only when `rotate=True`.
- Known limitations (spike): no committed test drives the rotated path (all 108 tests are angle=0); `_clip_convex` can emit a duplicate vertex when a vertex lies exactly on a clip edge (correct area in cases checked, not production-robust); `replay_trial`/Live View can't show rotation through the normal path because `build_trace` calls `make_scene` without `rotate=True`; rotated items draw as plain rectangles (no detailed glyphs); arbitrary non-rectangle polygons untested.

### Testing
- Ran `.venv/bin/python -m pytest -q`: 108 passed, 2 warnings (pre-existing imageio deprecations). No box-based test needed updating because angle=0 is behavior-preserving. Suite is slower (~6.5s vs ~2s) due to polygon clipping replacing AABB math.
- Ran `.venv/bin/python -m src.run_eval --quick`: completed end-to-end (all five planners, both sweeps).
- Verified ad-hoc (not in the committed suite): all five planners run end-to-end and deterministically on `rotate=True` scenes and clear all items; polygon overlap varies correctly with angle (concentric 40x40 squares 1600 -> ~1325 at 45deg); Live View renders rotated footprints (preview written under out/, gitignored).
- Confirmed current branch is `experiment-shapes`; changes are uncommitted.

## 2026-10-24 - SPIKE (experiment-shapes): fix convex-clip degeneracy + rotated-rectangle tests

### Changes
- On the `experiment-shapes` branch, fixed the `_clip_convex` on-edge degeneracy in the rotated-footprint geometry and added committed tests for the rotated-rectangle path. Axis-aligned (angle=0) behavior is unchanged: the 108-test baseline still passes.

### Files Modified
- `src/scene.py` — rewrote `_clip_convex` Sutherland-Hodgman emission logic and added `_dedupe_consecutive`.
- `tests/test_scene_rotated.py` — new test file (12 tests) for the rotated path.

### Technical Details
- The clip previously used a `>= 0` inside test, which emitted a redundant intersection vertex when a polygon vertex lay exactly on a clip edge (duplicate-vertex degeneracy). It now inserts an intersection point only on a STRICT crossing (one endpoint strictly inside, the other strictly outside, beyond `_ON_EDGE_EPS=1e-9`); on-edge endpoints are treated as inside without spawning a crossing point. `_dedupe_consecutive` collapses coincident and wrap-around vertices as a final guard.
- Verified the previously-degenerate case (axis-aligned box + 45-degree diamond sharing a corner) now clips to a clean 3-vertex triangle with the correct 400 px^2 area, and angle=0 parity is preserved (400, 1600, 0; symmetric).
- New tests cover: overlap correctness/symmetry for known rotated cases including the on-edge case; monotonic overlap decrease as two identical concentric squares rotate 0->45 deg; overlap never exceeds item area; covered_fraction stays in [0,1] (and caps at 1.0 with multiple rotated coverers); rotated-scene determinism for a fixed seed (angles, items, occluders stable); rotate=True assigns nonzero angles while rotate=False leaves them at 0; and occluders lists only higher-z items on rotated scenes.

### Testing
- Ran `.venv/bin/python -m pytest -q`: 120 passed, 2 warnings. Breakdown confirmed separately: the new file alone is 12 passed, and everything except the new file is the unchanged 108-passed baseline. The 2 warnings are pre-existing imageio deprecations in `tests/test_render.py`.

## 2026-10-24 - SPIKE (experiment-shapes): wire rotated footprints through app + eval

### Changes
- On the `experiment-shapes` branch, threaded the rotated-rectangle path through the Live View/replay flow and the evaluation harness so rotation is actually used and visible. Rotation is an explicit, default-False parameter everywhere, so the box-based path and the 120-test baseline are unchanged.

### Files Modified
- `src/trace.py` — `build_trace(..., rotate=False)` forwards to `make_scene`.
- `app/logic.py` — `TrialConfig`/`ExperimentConfig` gained `rotate` (default False); `build_scene`, `run_trial`, `run_experiment`, and `replay_trial` forward it; `replay_trial` passes the same `rotate` to both the trace and the paired `run_episode`.
- `app/streamlit_app.py` — added a "Rotated footprints (experimental)" sidebar checkbox (default on) and included `rotate` in the Live View replay cache key.
- `src/run_eval.py` — added a `ROTATE` module flag + `--rotate` CLI option applied in `evaluate` (so sweeps/sensitivity/ablation all honor it); `out/results.md` now states its geometry model.

### Technical Details
- Rotation is threaded as an opt-in parameter rather than a global flip, so every existing axis-aligned test still exercises the box path and passes. The Live View shows rotated scenes because the experiment runs with `rotate=True` and each trial's config flows into `replay_trial`.
- `replay_trial` uses the same `rotate` for the trace and the paired aggregate run, preserving trace-matches-sim on rotated scenes.
- `out/results.md` is regenerated with a "Geometry model: ROTATED-RECTANGLE footprints (experimental)" header so rotated numbers cannot be mistaken for the box numbers. `README.md` was deliberately NOT touched (it still cites box numbers); `src/compare.py` still uses box scenes (only run_eval was wired, as requested).

### Testing
- Ran `.venv/bin/python -m pytest -q`: 120 passed, 2 warnings (pre-existing imageio deprecations). Default-False threading kept the box baseline green.
- Verified via the app layer: a rotated `TrialConfig` yields replays with nonzero item angles; the rotated trace's final counters match its paired `EpisodeResult`; identical rotated configs produce identical frames and results (determinism); all five planners run end-to-end on rotated scenes via `run_experiment` and clear all items.
- Ran `.venv/bin/python src/run_eval.py --rotate` (30 seeds): completed end-to-end; regenerated `out/results.md` and `out/plots/`. Rendered a rotated Live View frame (15/15 items rotated) and visually confirmed tilted footprints.
- Rotated-world headline (clutter 0.9, 25 items): nearest 123.9 +/- 16.5 ticks / 24.10 failed; occlusion 68.7 +/- 5.3 / 10.27; leastcov 62.5 +/- 5.6 / 8.77. Sensitivity sweep shifted to 8 ties + 1 occlusion win (vs box 2/2/5). Push ablation still a tie.

## 2026-10-24 - SPIKE (experiment-shapes): Object shape model control, default axis-aligned

### Changes
- On the `experiment-shapes` branch, added an "Object shape model" sidebar selectbox with "Axis-aligned boxes" (default) and "Rotated footprints (experimental)", replacing the earlier rotation checkbox that defaulted to on. The default is now axis-aligned, so doing nothing reproduces the published box behavior and numbers. Added an honest experimental caption near Results, shown only in rotated mode.

### Files Modified
- `app/streamlit_app.py` — replaced the rotation checkbox with the two-option "Object shape model" selectbox (default index 0 = axis-aligned); added the rotated-mode Results caption; help text avoids the "3D" token.

### Technical Details
- The selectbox maps "Rotated footprints (experimental)" -> rotate=True, else False; the value flows into ExperimentConfig.rotate and (unchanged from the prior wiring turn) through to make_scene across the batch runner, Live View replay, and Results.
- Default-axis-aligned verified byte-identical: a default ExperimentConfig has rotate=False, and the axis-aligned batch result equals a direct box run_episode per seed (checked across seeds for occlusion).
- Honest Results caption added (shown only when result.config.rotate is True): "Rotated footprints are experimental; under tilted overlapping shapes the occlusion-aware planner's advantage shrinks and a greedy least-covered rule performs comparably."
- No grasp-probability constants or physics changed: `src/robot.py` is untouched (empty diff). The feature is never labeled "3D"; help text calls it tilted 2D rectangles. The one remaining "3D" string is the pre-existing Live View disclaimer "(no 3D physics or real hardware)", which denies 3D rather than labeling the feature, and was left in place.

### Testing
- Ran `.venv/bin/python -m pytest -q`: 120 passed, 2 warnings (pre-existing imageio deprecations). The 108 box baseline is unchanged.
- Verified default ExperimentConfig.rotate is False and the axis-aligned batch is byte-identical to a direct box run_episode; confirmed a rotated TrialConfig still produces replays with rotated item footprints. Browser launch is a manual step (not run here).
