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