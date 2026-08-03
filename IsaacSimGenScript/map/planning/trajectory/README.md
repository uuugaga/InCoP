# Generated planning trajectories

Trajectory files are generated locally and intentionally excluded from Git.
Run `map/planning/planner.py` and `map/planning/dual.py` to recreate them.

Expected structure:

```text
map/planning/trajectory/
  <scene>/                  # hospital, office, or warehouse
    planner/
      roadmap*.pkl
      *.png
    dual/
      path_case_distance/case_*.pkl
      path_case_shadow/case_*.pkl
      path_case_highoverlap/case_*.pkl
      path_case_random/case_*.pkl
      path_case_*_split.json
      debug_distance/{train,validate,test}/
      debug_shadow/{train,validate,test}/
      debug_highoverlap/{train,validate,test}/
      debug_random/{train,validate,test}/
```

The same `planner/` and `dual/` layout applies to all three scenes. Do not
commit generated roadmaps, case PKLs, split JSONs, or debug images.
