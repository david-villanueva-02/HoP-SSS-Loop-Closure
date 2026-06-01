# data/

Raw side-scan sonar surveys in **XTF** (eXtended Triton Format) go here. They are **git-ignored**
(`*.xtf`, `*.png`, `*.tiff`, `*.db`) — provide your own.

Expected by the notebooks' default config:

```
data/2025-09-24_09-25-24_0.xtf
```

Point the `xtf_file` variable (config cell of either notebook) at the file you want to process. The
notebook loads it once with `load_xtf()` and detects straight-line segments; the `TIMESTAMP_PAIRS`
entries are ping indices into this survey.
