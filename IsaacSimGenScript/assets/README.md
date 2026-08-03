# Local Isaac Sim assets

USD assets are intentionally excluded from Git and GitHub packages. Restore
them locally after cloning or unpacking the project.

Expected structure:

```text
assets/
  map/
    hospital_v1.usd
    office_v1.usd
    full_warehouse_v1.usd
  map_objects/
    *.usd
```

Keep `map/` and `map_objects/` as sibling directories. The scene USD files use
relative references such as `../map_objects/SM_ChairOffice.usd`. Copy all USD
dependencies referenced by the three scene files into `map_objects/` before
running the generator.

Some NVIDIA base environments and props remain HTTPS references supplied by
Isaac Sim 5.1, so network access may still be required.
