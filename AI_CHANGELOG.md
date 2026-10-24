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
