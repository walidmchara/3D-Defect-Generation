# 3D data

V1 does not require an external dataset.

It generates synthetic CT-like 3D volumes and injects geometric defects:

- spherical voids,
- ellipsoidal inclusions,
- thin crack-like defects.

Each sample contains:

```text
clean volume
defective volume
binary defect mask
defect type
```

Default volume shape:

```text
32 × 32 × 32 voxels
```

This synthetic generator is intended for code demonstration and model prototyping.
It should not be interpreted as a physically validated CT defect simulator.

Future versions can add real industrial CT/X-ray volumes and more realistic
scanner/noise/reconstruction physics.
