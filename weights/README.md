# weights/

PhysDNet model checkpoints go here. They are **git-ignored** (`*.pth`) — download them separately.

The notebooks default to:

```
weights/best_model_v610_jaguar.pth
```

This is the checkpoint loaded by [`src/inference.py`](../src/inference.py) via the `weight_path`
variable in each notebook's config cell. Other PhysDNet checkpoints (e.g. `*_eagle`, `*_v611_jaguar`)
can live here too — switch by editing `weight_path`.

> The **MINIMA** matcher uses a separate checkpoint, `MINIMA/weights/minima_lightglue.pth`, downloaded
> into the MINIMA clone (not here). See the top-level [README](../README.md) → "Setup".
