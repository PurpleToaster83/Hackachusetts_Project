# Project structure

```
messy_desk/
  data/                  # local copies of bridge-v2, droid, oxe data.csv (read-only)
  src/
    mine.py              # instruction vocabulary + reference stats -> ref_stats.json
    scene.py             # Item, Container, make_scene, occluders
    robot.py             # Gripper, primitives, grasp model
    planner.py           # RandomPlanner, NearestPlanner, OcclusionPlanner
    log.py               # EpisodeLogger, dataset writer
    sim.py               # run_episode(scene, planner, logger) -> result
    run_eval.py          # sweeps and plots
    render.py            # pygame renderer and video export
  tests/                 # one test file per module
  out/                   # generated: datasets, plots, videos (gitignored)
  README.md
  .kiro/steering/
```

## Module contracts (keep these signatures stable)

- `scene.make_scene(seed: int, n_items: int, clutter: float) -> Scene`
- `scene.occluders(scene) -> dict[int, list[int]]`  # item id -> ids covering >25% of it
- `robot.Gripper.step(action) -> None`  # advances one tick at 15 Hz
- Planner interface: `next_action(scene, gripper) -> Primitive`
- `sim.run_episode(scene, planner, logger=None, seed=0) -> EpisodeResult`
- `EpisodeResult`: items_sorted, items_total, steps, failed_grasps, pushes

## Conventions
- One responsibility per module. No module imports from `render.py`.
- The simulation core (scene, robot, planner, sim, log) must run headless, with no pygame import.
- Only `render.py` imports pygame.
- All randomness comes from a seeded `numpy.random.Generator` passed in. No global random state.
- Dataclasses for Item, Container, Scene, EpisodeResult.
- Generated files go in `out/`. Never write into `data/`.

## Dataset layout (matches the kit's shape)
- `out/dataset/meta/episodes.csv`: `episode,task,frames`
- `out/dataset/data/episode_{index:06d}.csv`: `frame_index,timestamp,state_x,state_y,holding,action_type,action_x,action_y,reward,done`