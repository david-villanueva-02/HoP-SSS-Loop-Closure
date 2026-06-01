# SSS Feature-Matching Report Pipeline

Automated multi-method comparison of feature matchers (SIFT, LightGlue + SuperPoint, LightGlue + SIFT, MINIMA) on Side-Scan Sonar image pairs, producing the plots and tables needed for the project report.

> Part of [HoP-SSS-Loop-Closure](../README.md) — see the top-level README for setup and quickstart. Paths below are relative to this `docs/` folder.

## Why this exists

The base pipeline ([configurable_backend_pipeline2.ipynb](../notebooks/configurable_backend_pipeline2.ipynb)) runs **one** matcher at a time: changing the method or a parameter requires editing the config cell and re-running. For the report we need to compare several methods × parameters × image sources on the same pair of images, and do that across multiple manually-chosen timestamp pairs that span easy/hard cases. The report pipeline automates everything downstream of choosing the timestamps.

## File layout

| File | Purpose |
|---|---|
| [report_utils.py](../src/report_utils.py) | Pure module: sweep runner, plot functions, IO helpers. Decoupled from the notebook — accepts `make_matcher` and `run_registration` callables. |
| [auto_report.ipynb](../notebooks/auto_report.ipynb) | Batch driver. Edit `TIMESTAMP_PAIRS`, run all cells, get one folder of plots per pair. |
| [configurable_backend_pipeline2.ipynb](../notebooks/configurable_backend_pipeline2.ipynb) | Single-pair interactive pipeline. Last 9 cells ("Automatic Report") run the sweep on the currently loaded pair without batching. |
| [minima_pipeline.py](../src/minima_pipeline.py) | Shared registration core: matching → mask filter → pixel RANSAC → UTM RANSAC. Used by all notebooks. |

`auto_report.ipynb` and the main notebook share two pieces of code verbatim (matcher classes from cell 29; pipeline overrides from cell 24). If you change those cells in the main notebook, mirror the change into `auto_report.ipynb` cells 6 and 8.

## End-to-end workflow

1. **Find timestamps interactively** (see [Picking timestamps](#picking-timestamps)).
2. Open `auto_report.ipynb`, edit `TIMESTAMP_PAIRS` in cell 3.
3. (Optional) edit `SWEEP_SPECS` to narrow or widen the comparison.
4. Run every cell. Each pair drops a folder under `output_simplified/report/{ts1}_{ts2}/`.
5. Use the saved PNGs and `summary.csv` directly in the report.

## Picking timestamps

The interesting axes for the report are **overlap fraction** (easy vs hard) and **geometry** (parallel, anti-parallel, X-crossing). For two pings `i`, `j`:

| Metric | Formula | Meaning |
|---|---|---|
| Center distance | `d = ‖trajectory[i] - trajectory[j]‖` | How far apart the sonar was |
| Swath span | `w = ‖swaths[i, -1] - swaths[i, 0]‖` | Sonar footprint width in metres |
| Normalised overlap | `o = clip(1 - d / w, 0, 1)` | 1.0 = pings stacked, 0.0 = footprints just touch |
| Heading delta | `Δψ = wrap(yaw[i] - yaw[j])` | 0° = same direction, ±180° = opposite, ±90° = X-crossing |

Suggested buckets: **easy** `o > 0.7`, **medium** `0.4 < o < 0.7`, **hard** `0.1 < o < 0.4`. Plus one of `parallel` (`|Δψ| < 30°`), `anti-parallel` (`|Δψ| > 150°`), or `X-cross` (`60° < |Δψ| < 120°`).

A 4-pair set covering the axes (e.g. `easy/anti-parallel`, `easy/parallel`, `hard/anti-parallel`, `medium/X-cross`) is usually enough for the report. There's a `find_candidate_pairs(group_a, group_b, ...)` snippet documented in the conversation history that ranks candidates from the loaded `trajectory`, `swaths`, `yaw`; paste it after cell 4 of `auto_report.ipynb` if you want quantitative help instead of eyeballing the trajectory plot.

## Configuration reference

The second code cell of both notebooks defines the global defaults below (values shown are
`auto_report.ipynb`'s). Per-spec `matcher_kwargs` / `ransac_kwargs` override them; see
[SweepSpec fields](#sweepspec-fields).

**Paths** — auto-derived from the repo root via `_REPO_ROOT`, so the project is portable:

| Name | Resolves to |
|---|---|
| `xtf_file` | survey filename inside `data/` |
| `xtf_dir` | `data/` (`_REPO_ROOT / "data"`) |
| `output_dir` | `output_simplified/` |
| `weight_path` | `weights/best_model_v610_jaguar.pth` (PhysDNet) |
| `MINIMA_CKPT` | `MINIMA/weights/minima_lightglue.pth` |

**XTF section extraction**

| Name | Default | Meaning |
|---|---|---|
| `segment_size` | 2000 | pings per extracted section window |
| `upper_limit` | 2\*\*15 | clip value for normalising sonar intensity |

**Masking / matching**

| Name | Default | Meaning |
|---|---|---|
| `TERRAIN_MASK_K` | 1.0 | scale on the terrain-height threshold |
| `MASK_COMPONENTS` | `["shadow","blind"]` | masks merged into the rejection mask (`z`/`shadow`/`blind`) |
| `FLIP_LEFT_FINAL_MASK` | False | flip the final LEFT mask before matching |
| `MATCH_MODE` | `"both"` | channels matched: `both`/`left-left`/`right-right`/`left-right`/`right-left` |
| `FLIP_LEFT_INPUT_IMAGE` | False | flip every LEFT input image before matching |
| `APPLY_MASK` | True | enable the mask-filter stage |
| `APPLY_UTM_DISTANCE_FILTER` | True | enable the UTM-distance gross-outlier stage |
| `UTM_DISTANCE_THRESHOLD_METERS` | 10.0 | reject matches whose paired UTM coords differ by more (m) |

**LightGlue defaults**

| Name | Default | Meaning |
|---|---|---|
| `LIGHTGLUE_MAX_NUM_KEYPOINTS` | 4096 | extractor keypoint budget |
| `LIGHTGLUE_RESIZE` | None | optional resize before extraction |
| `LIGHTGLUE_DEPTH_CONFIDENCE` | 0.95 | early-stop confidence (lower = faster; -1 disables) |
| `LIGHTGLUE_WIDTH_CONFIDENCE` | 0.99 | point-pruning confidence (lower = faster; -1 disables) |
| `LIGHTGLUE_FILTER_THRESHOLD` | 0.1 | match cutoff (higher = fewer, stronger) |
| `LIGHTGLUE_MP` | False | mixed precision |
| `LIGHTGLUE_FLASH` | True | use FlashAttention if available |
| `LIGHTGLUE_COMPILE` | False | `torch.compile` the matcher |

**SIFT defaults**

| Name | Default | Meaning |
|---|---|---|
| `SIFT_NFEATURES` | 8000 | max features |
| `SIFT_N_OCTAVE_LAYERS` | 3 | octave layers |
| `SIFT_CONTRAST_THRESHOLD` | 0.01 | contrast threshold |
| `SIFT_EDGE_THRESHOLD` | 10 | edge threshold |
| `SIFT_SIGMA` | 1.6 | Gaussian sigma at octave 0 |
| `SIFT_MATCHER` | `"flann"` | descriptor matcher: `flann` or `bf` |
| `SIFT_RATIO_TEST` | 0.75 | Lowe ratio-test cutoff |
| `SIFT_CROSS_CHECK` | False | BF cross-check (only when `bf`) |
| `SIFT_FLANN_TREES` | 5 | FLANN KD-tree count |
| `SIFT_FLANN_CHECKS` | 64 | FLANN search checks |
| `SIFT_ENFORCE_UNIQUE_MATCHES` | True | keep one-to-one best matches |
| `SIFT_MAX_MATCHES` | None | cap on kept matches (None = all) |

**RANSAC defaults** — override per spec via `ransac_kwargs`:

| Name | Default | Meaning |
|---|---|---|
| `RANSAC_REPROJ_THRESHOLD_PIXEL` | 10.0 | inlier threshold, pixel homography (px) |
| `RANSAC_REPROJ_THRESHOLD_UTM` | 5.0 | inlier threshold, UTM homography (m) |
| `RANSAC_MAX_ITERS` | 50000 | max iterations |
| `RANSAC_CONFIDENCE` | 0.995 | confidence |
| `RANSAC_OUTLIER_PERCENT` | 0.6 | assumed outlier fraction |

## The default sweep

`auto_report.ipynb` cell 3 sets `SWEEP_SPECS = default_sweep_specs_expanded()`. That helper, defined in [report_utils.py](../src/report_utils.py), produces (per `image_source ∈ {raw, rho_gray}`):

- **SIFT** × `ratio_test ∈ {0.7, 0.75, 0.8}` × `nfeatures ∈ {4000, 8000}` → 6 specs
- **LightGlue + SuperPoint** × `filter_threshold ∈ {0.05, 0.1, 0.2}` × `max_num_keypoints ∈ {2048, 4096}` → 6 specs
- **LightGlue + SIFT** × `filter_threshold ∈ {0.1, 0.2}` → 2 specs
- **MINIMA (sp_lg)** × default → 1 spec

Plus a RANSAC sweep on `rho_gray` only: per backend, vary `ransac_reproj_threshold_utm ∈ {2.0, 5.0}` × `ransac_outlier_percent ∈ {0.5, 0.7}` → 12 specs (3 backends × 4 ransac configs).

Total ≈ `(6 + 6 + 2 + 1) × 2 + 12 = 42` runs per pair. ~12-15 minutes per pair on GPU thanks to matcher caching. Swap to `report_utils.minimal_sweep_specs()` for a 6-run smoke test, or `report_utils.default_sweep_specs()` for the previous 30-run grid.

### SweepSpec fields

```python
SweepSpec(
    name="...",                  # short id, used as folder name
    backend="...",               # see backends above
    matcher_kwargs={...},        # backend-specific args
    image_source="raw"|"rho_gray",
    method_label="...",          # legend label
    variant_label="...",         # short variant tag
    ransac_kwargs={...},         # NEW: per-spec RANSAC overrides
)
```

`ransac_kwargs` keys (each falls back to the corresponding `RANSAC_*` global from cell 2 when missing):

| Key | Global it overrides |
|---|---|
| `ransac_reproj_threshold_pixel` | `RANSAC_REPROJ_THRESHOLD_PIXEL` |
| `ransac_reproj_threshold_utm` | `RANSAC_REPROJ_THRESHOLD_UTM` |
| `ransac_max_iters` | `RANSAC_MAX_ITERS` |
| `ransac_confidence` | `RANSAC_CONFIDENCE` |
| `ransac_outlier_percent` | `RANSAC_OUTLIER_PERCENT` |
| `apply_utm_distance_filter` | `APPLY_UTM_DISTANCE_FILTER` |
| `utm_distance_threshold` | `UTM_DISTANCE_THRESHOLD_METERS` |

## Pipeline stages and the funnel

Each registration goes through four filters; the funnel plot tracks the surviving match count at each stage.

| Stage | Removed by | Field on `RegistrationResult` |
|---|---|---|
| `raw` | nothing | `num_matches` |
| `after_mask` | shadow / blind / terrain masks (`filter_matches_with_masks`) | `num_matches_after_mask` |
| `after_utm_distance` | matches whose paired UTM coords differ by more than `UTM_DISTANCE_THRESHOLD_METERS` (gross-outlier pre-filter) | `num_matches_after_utm_distance_filter` |
| `ransac_inliers` | 3-DOF Euclidean RANSAC in UTM (`estimate_utm_homography_ransac`). Funnel + summary `n_inliers` counts UTM inliers only and is 0 when UTM RANSAC fails. Pixel-RANSAC inliers are reported separately as `n_pixel_inliers`. | `ransac_inliers_utm` |

The RANSAC output gives `H_utm` — a 3×3 Euclidean transform in metric UTM frame. Recovered parameters `(tx_utm, ty_utm, theta_utm_deg)` are extracted via `extract_euclidean_homography_params`.

## Output structure

Each pair gets one folder under `output_simplified/report/`. The folder name embeds the optional label so cases are identifiable on disk:

```
output_simplified/
└── report/
    ├── all_pairs_summary.csv              # one row per (pair, spec); cross-pair
    ├── cross_pair_residual_raw.png        # median residual vs pair, one line per method
    ├── cross_pair_residual_rho_gray.png
    ├── cross_pair_residual_combined.png   # pooled across image sources
    ├── cross_pair_inliers_raw.png         # inlier count vs pair
    ├── cross_pair_inliers_rho_gray.png
    ├── cross_pair_inliers_combined.png
    │
    └── {label}_{ts1}_{ts2}/               # one folder per pair (or {ts1}_{ts2} if no label)
        ├── funnel_<method>_raw.png        # match-count evolution, one figure per model with one line per variant
        ├── funnel_<method>_rho_gray.png
        ├── utm_residuals_raw.png          # per-match residual histogram, per method
        ├── utm_residuals_rho_gray.png
        ├── homography_params.png          # tx/ty/θ bar chart across all runs
        ├── inference_times.png            # matcher.match_images runtime histogram per method
        ├── bundled_matches_raw.png        # cross-method matches at each stage, one panel per run
        ├── bundled_matches_after_mask.png
        ├── bundled_matches_after_utm.png
        ├── bundled_matches_ransac.png
        ├── summary.csv                    # per-run table (one row per spec, this pair)
        ├── summary.json                   # same data, JSON
        ├── summary_table.png              # rendered table
        ├── runs/
        │   └── <spec_name>/
        │       ├── matches_raw.png        # cv2.drawMatches at each stage
        │       ├── matches_after_mask.png
        │       ├── matches_after_utm.png
        │       ├── matches_ransac.png
        │       └── matches_stages.png     # 4-panel combined version of the above
        └── intermediates/                 # everything needed to rerun this pair
            ├── section_01/                # XTF dump, PhysDNet output, masks
            └── section_02/
```

One pair = one self-contained folder. Easy to share, archive, or delete.

`TIMESTAMP_PAIRS` entries can be either `(ts1, ts2)` or `(ts1, ts2, "label")`. Labels become the x-axis ticks on the cross-pair plots, so make them short and descriptive (`easy_parallel`, `hard_anti`, `cross_overlap`). The order of the list is preserved on the cross-pair plots — put pairs in difficulty order (easy → hard) so the plot reads left to right.

## Plot reference

### Filtering funnel (`funnel_<method>_<source>.png`)
- One figure per `(method, image_source)` so variants don't crowd each other.
- X-axis: pipeline stages.
- Y-axis: number of matches; linear when counts fit in one decade, log otherwise. Never below 0.
- One line per variant inside the figure, with distinct colours from `tab10`/`tab20` so different parameter settings stand out.
- **Read this for**: how much each stage hurts this method, and how sensitive the method is to its own parameters. On easy pairs all variants stay high; on hard pairs you'll see one collapse early (often at the UTM-distance stage if it's matching noise).

### UTM residuals (`utm_residuals_<source>.png`)
- Overlaid histograms of per-inlier `‖H_utm·p0 − p1‖` in metres.
- Method legend shows median + p95.
- **Read this for**: quality of the recovered transform. Lower median = tighter fit. A method with many inliers but a large median residual is likely over-fitting noise.

### Homography parameters (`homography_params.png`)
- Bar chart of `(tx_utm, ty_utm, θ_utm_deg)` per run, coloured by method.
- **Read this for**: agreement across methods. If three methods recover `tx ≈ 12 m` and one says `tx ≈ −3 m`, the outlier is wrong (or matching across opposite directions).

### Per-run matches at every stage (`runs/<spec>/matches_<stage>.png`, `matches_stages.png`)
- `cv2.drawMatches` overlay of the surviving matches at each pipeline stage: `raw`, `after_mask`, `after_utm`, `ransac`.
- 4 individual PNGs plus a combined `matches_stages.png` with all 4 stacked.
- **Read this for**: per-run sanity check that each filtering stage is doing what it should — a near-empty `ransac` panel after a populated `after_utm` means UTM RANSAC failed to find a consistent transform.

### Bundled matches across methods (`bundled_matches_<stage>.png`)
- One figure per stage, with one panel per `(method, variant, image_source)` run stacked vertically.
- **Read this for**: side-by-side comparison of what each method finds at the same filtering point — usually the most informative visual for the report.

### Inference-time histogram (`inference_times.png`)
- Overlaid histograms (one colour per method) of `matcher.match_images` runtime in seconds.
- Vertical reference lines at 30 FPS (33 ms) and 10 FPS (100 ms) for real-time-viability context.
- Legend reports median, p95, and sample count per method.
- **Read this for**: which method is fast enough to be used online. `runtime_s` in the summary CSV is the *full* pipeline including RANSAC; `matcher_time_s` (also in the CSV) is what this plot uses.

### Summary table (`summary.csv`, `summary_table.png`)
- One row per spec. Columns: counts at each stage, residual stats (mean/median/p95/RMSE), recovered transform, `matcher_time_s` (just `matcher.match_images`), `runtime_s` (full pipeline), `h_utm_ok` flag, error string.
- **Use this directly in the report.**

### Cross-pair plots (`cross_pair_*.png`, `all_pairs_summary.csv`)
- Generated automatically by the last cell of `auto_report.ipynb` (calls `generate_cross_pair_report`).
- X-axis: pair label (or `ts1_ts2` if unlabelled), in the order pairs were processed.
- Y-axis: median (across that method's parameter variants) of `residual_median_m` or `n_inliers`.
- One line per method. Separate figure per image source plus a combined "all sources" version. The legend names every variant that was pooled into that line so the aggregate isn't opaque (e.g. `SIFT  (median over 6 variants: ratio=0.7 n=4000, ratio=0.7 n=8000, ...)`).
- **Read this for**: how each method degrades from easy to hard pairs. The "report-quality" figure for the comparison argument — you can't easily produce this from the per-pair PNGs alone.
- `all_pairs_summary.csv` is the flat join of every per-pair `summary.csv` with `pair_folder`, `pair_label`, `ts1`, `ts2` columns prepended; load it with pandas/spreadsheet for ad-hoc analysis.

## Adding / modifying specs

`SweepSpec` is:

```python
SweepSpec(
    name="lgsp_ft0.1_rho",          # short unique id, used as folder name
    backend="lightglue_superpoint", # "sift" | "lightglue_superpoint" | "lightglue_sift" | "minima_sp_lg"
    matcher_kwargs={"filter_threshold": 0.1, "max_num_keypoints": 4096},
    image_source="rho_gray",        # "raw" | "rho_gray"
    method_label="LightGlue+SuperPoint",  # label on plot legends
    variant_label="ft=0.1",         # short variant tag
    ransac_kwargs={"ransac_reproj_threshold_utm": 2.0},  # optional, overrides globals
)
```

Any kwarg not in `matcher_kwargs` falls back to the corresponding `*_FILTER_THRESHOLD` / `SIFT_*` / `LIGHTGLUE_*` global in cell 2. Any key not in `ransac_kwargs` falls back to the matching `RANSAC_*` global. Matcher instances are cached by `(backend, frozenset(kwargs))` — note this means specs that share `(backend, matcher_kwargs)` but differ in `ransac_kwargs` reuse the matcher (intentional, since RANSAC happens after matching).

## Adding a new backend

1. Implement a class with `match_images(img0, img1) -> (mkpts0, mkpts1, mconf)` returning `(N, 2)` float32 arrays in pixel coordinates.
2. Add it to `make_matcher_for_spec` in `auto_report.ipynb` cell 10 (and the parallel function in the main notebook's "Automatic Report" section if you want it there too).
3. Add `SweepSpec` entries for it in cell 3.

No changes to `report_utils.py` are needed — it works on whatever `RegistrationResult` comes back.

## Known quirks

- **MINIMA imports**: `MINIMA/` has no `__init__.py` or `setup.py`. `minima_pipeline.py` patches `sys.path` at import time with both the repo root (so `from MINIMA.load_model import load_model` resolves) and `MINIMA/` itself (so the `from third_party.LightGlue ...` lines inside `load_model` resolve at runtime). Works regardless of the notebook's cwd.
- **`display_figure` is a no-op in `auto_report.ipynb`** (cell 8, top of the pipeline-helpers block). The main notebook's version shows mask figures interactively; for batch runs that would spam dozens of windows. Mask PNGs are still written to disk by `build_final_masks` regardless. Swap the no-op for the commented-out plotting body if you want them.
- **`run_parametrized_minima_registration` is duplicated**. The base version in `minima_pipeline.py` is shadowed by the override in cell 24 of the main notebook (and cell 8 of `auto_report.ipynb`), which adds `flip_left_image`, `apply_utm_distance_filter`, and populates the `utm_distance_*` fields on `RegistrationResult`. The override is the one in active use.
- **Failed pairs don't abort the batch**. The loop catches per-pair exceptions; check the summary printed at the end. A pair whose timestamp lands in a turning segment is *skipped* (not failed), and `process_pair` returns `None`.
- **MINIMA `filter_threshold` is ignored**. The `MinimaMatcher` constructor in `minima_pipeline.py` doesn't accept matcher-side kwargs, so any sweep variants would produce identical results. Keep one MINIMA spec per image source.

## Verifying a run

Quick sanity checks after the batch finishes:

1. Funnel monotone: `raw ≥ after_mask ≥ after_utm_distance ≥ inliers` for every method. If not, the filtering stages are out of order or a mask is inverted.
2. Residual median ≪ `RANSAC_REPROJ_THRESHOLD_UTM` (2 m default) for any method with `h_utm_ok = yes`.
3. Cross-check one spec against the single-run path in `configurable_backend_pipeline2.ipynb` with the same kwargs — `num_matches`, `num_matches_after_mask`, `num_ransac_inliers` should match exactly.
4. Visual: the `matches_ransac.png` panels should show parallel lines across the image pair (consistent transform); diverging or crossed lines mean RANSAC kept inconsistent matches.
