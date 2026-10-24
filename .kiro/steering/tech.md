# Tech stack and workflow

## Stack
- Python 3.10+
- numpy, pandas, pygame, matplotlib, imageio
- Optional stretch only: gymnasium, stable-baselines3
- No other dependencies without asking. No network access at runtime.

## Simulation parameters
- Desk 640 x 480, 15 Hz tick, kinematic gripper with capped speed per tick
- Primitives: move_to(x, y), grasp(), release(), push(dx, dy)
- Occlusion: item A covers B if A.z > B.z and overlap area / B area > 0.25
- Grasp success probability: 1 - 0.8 * covered_fraction, with a small noise term
- Perception noise: about 5% chance of misclassifying an item's kind
- Instruction templates come from bridge-v2 / droid phrasing

## Commands
```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas pygame matplotlib imageio
python src/mine.py            # writes out/ref_stats.json
python src/run_eval.py --seeds 50 --quick   # quick smoke run
python src/run_eval.py --seeds 50           # full evaluation + plot
python src/render.py --seed 3 --out out/demo.mp4
pytest -q
```

## Agent workflow rules
1. Build in this order and stop after each step so I can check it:
   1. scene.py + tests, then a random-planner baseline in sim.py
   2. robot.py grasp and push primitives + tests
   3. NearestPlanner, then OcclusionPlanner
   4. log.py dataset writer
   5. run_eval.py plots
   6. render.py demo
   7. Stretch: RL wrapper
2. After each step, run `pytest -q` and a small `run_eval.py --quick`. Report the numbers.
3. Do not move on if the occlusion-aware planner does not beat nearest-first at 15+ items. Check that grasp success really depends on coverage before changing the planner.
4. Same seed must give identical results. Add a test for this.
5. Do not add features outside product.md scope. Ask first.
6. Keep functions small and typed. Short docstrings that state units (pixels, ticks).
7. Never fabricate results. The README table is filled only from real run_eval.py output.

## Testing
- Unit tests: occlusion fraction math, determinism per seed, grasp probability, dataset columns match the schema
- Smoke test: one full episode runs headless in under a second