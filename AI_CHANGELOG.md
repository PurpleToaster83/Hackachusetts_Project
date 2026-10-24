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
