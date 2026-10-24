# Robotics starter kits

Each folder is one dataset: data.csv, README.md, fetch.py, and fetch.js.
Open data.csv in a spreadsheet, or run fetch.py for a fresh response from the direct URL.

- bridges: National Bridge Inventory, Massachusetts
- oxe: Open X-Embodiment dataset index
- boston-map: Boston street grid, 256
- qcew: Massachusetts manufacturing and warehousing employment
- bridge-v2: BridgeData V2 demonstrations
- droid: DROID-100 Franka demonstrations
- boston-streets: Boston street centerlines

## Results

**Main finding:** planners that reason about occlusion clear a cluttered desk far faster than planners that ignore it. At clutter 0.9 with 25 items, nearest-first needs about [164] ticks and [34.5] failed grasps per desk. Our occlusion-aware planner needs about [69] ticks and [10.5] failed grasps. (Mean over 30 seeds per setting; 95% intervals are in `out/results.md`.)

**An honest caveat:** most of that gain comes from simply being aware of coverage. A simple greedy baseline that always grabs the least-covered item performs about as well as our full occlusion planner at these settings. So the contribution is the measurement that occlusion-awareness matters, not a claim that our planner is far better than every alternative.

**Singulation:** [In our ablation, enabling pushes changed ticks to clear by X and the planner pushed Y times per desk. Fill in from the ablation.]

**How robust is it?** We varied the failed-grasp penalty and the crowding penalty. [The ranking held / changed when ...]. All constants were fixed before running the comparison and were not tuned afterwards.

**Limits:** this is a 2D simulation with a hand-written grasp model. It uses no real vision or hardware, and the kit data supplied only instruction phrasing and episode scale. The ranking of planners could change under different grasp physics.

-----------------

30 random desks per setting. Values are mean ± 95% interval; I call two results different only when their intervals don't overlap.

**Occlusion-awareness matters.** At clutter 0.9 with 25 items, the occlusion-aware planner clears the desk in 69 ± 6 ticks with 10.5 failed grasps. Nearest-first needs 165 ± 17 ticks and 34.5 failed grasps, and random needs 113 ± 10 ticks and 15.4 failed grasps. At low clutter and small sizes, planners are indistinguishable.

**A simple rule does almost as well.** A greedy planner that always grabs the least-covered item matches our occlusion-aware planner at the default settings (66 ± 6 vs 69 ± 6 ticks). Our planner's advantage shows up only when crowded items are heavily penalized: in the sensitivity sweep it clearly wins 2 of 9 settings, `LeastCovered` clearly wins 2, and 5 are ties.

**Singulation did not help here.** Allowing the planner to push blockers aside changed ticks to clear from 69.7 ± 5.5 to 69.4 ± 6.0, with 0.33 pushes per desk. With our grasp model, choosing which item to grab matters far more than moving things.

**Limits.** This is a 2D simulation with a hand-written grasp model and penalty constants we chose before running the comparison. Different physics could change the ranking.