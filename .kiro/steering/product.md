# Product: Messy Desk

## Purpose
A 2D simulation and planning system for the Hackachusetts "Messy Desk: Dynamic Object Sorting" robotics challenge. A robot must identify, approach and sort overlapping everyday desk items (pens, keys, mugs, trash) into the correct containers. The system also logs every run as a reusable robotics dataset.

## Judged on
1. Handling occlusion (clutter with overlapping items), including singulation.
2. A clear heuristic orchestration policy with measurable results versus baselines.
3. Data serialization and pipeline design.
4. Scene randomization (optional RL as a stretch).
5. A demo that is easy to understand in under two minutes.

## Users
Hackathon judges and teammates. They need to run it in one command and see results fast.

## Scope
In scope:
- Randomized cluttered desk scenes
- Occlusion graph and occlusion-aware planner
- Three planners: random, nearest-first, occlusion-aware
- Dataset logging in the BridgeData V2 / DROID shape
- Evaluation plots and a pygame demo video

Out of scope (do not build unless told):
- Real computer vision, real robot hardware, 3D physics engines
- Downloading any data (all data is local CSV)
- Web UI, databases, accounts

## Honesty rule
This is a simulation. The starter kit supplies instruction phrasing, episode scale and schema only. It has no images or trajectories. Never describe perception as real vision, and keep the README's limitations section accurate.

## Definition of done
`python run_eval.py` produces the comparison plot, `python render.py` produces the demo video, and the README table matches the numbers.