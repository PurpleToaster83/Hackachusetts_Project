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
