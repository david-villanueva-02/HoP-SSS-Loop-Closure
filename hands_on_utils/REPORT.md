# SSS Feature-Matching Report Pipeline

Automated multi-method comparison of feature matchers (SIFT, LightGlue + SuperPoint, MINIMA) on Side-Scan Sonar image pairs, producing the plots and tables needed for the project report.

## Why this exists

The base pipeline ([configurable_backend_pipeline2.ipynb](configurable_backend_pipeline2.ipynb)) runs **one** matcher at a time: changing the method or a parameter requires editing the config cell and re-running. For the report we need to compare several methods × parameters × image sources on the same pair of images, and do that across multiple manually-chosen timestamp pairs that span easy/hard cases. The report pipeline automates everything downstream of choosing the timestamps.

## File layout

| File | Purpose |
|---|---|
| [report_utils.py](report_utils.py) | Pure module: sweep runner, plot functions, IO helpers. Decoupled from the notebook — accepts `make_matcher` and `run_registration` callables. |
| [auto_report.ipynb](auto_report.ipynb) | Batch driver. Edit `TIMESTAMP_PAIRS`, run all cells, get one folder of plots per pair. |
| [configurable_backend_pipeline2.ipynb](configurable_backend_pipeline2.ipynb) | Single-pair interactive pipeline. Last 9 cells ("Automatic Report") run the sweep on the currently loaded pair without batching. |
| [minima_pipeline.py](minima_pipeline.py) | Shared registration core: matching → mask filter → pixel RANSAC → UTM RANSAC. Used by all notebooks. |

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

## The default sweep (14 specs)

Defined inline in cell 3 of `auto_report.ipynb`. For each `image_source ∈ {raw, rho_gray}`:

- **SIFT** × `ratio_test ∈ {0.7, 0.8}` at `nfeatures = 8000` → 2 specs
- **LightGlue + SuperPoint** × `filter_threshold ∈ {0.05, 0.1, 0.2}` at `max_num_keypoints = 4096` → 3 specs
- **MINIMA (sp_lg)** × default → 1 spec (the matcher class here doesn't accept `filter_threshold`, so additional variants would be redundant)

Total: `(2 + 3 + 1) × 2 = 12` runs per pair. ~5 minutes per pair on GPU. Use `report_utils.minimal_sweep_specs()` for a 6-run smoke test, or `report_utils.default_sweep_specs()` for the full 30-run grid.

## Pipeline stages and the funnel

Each registration goes through four filters; the funnel plot tracks the surviving match count at each stage.

| Stage | Removed by | Field on `RegistrationResult` |
|---|---|---|
| `raw` | nothing | `num_matches` |
| `after_mask` | shadow / blind / terrain masks (`filter_matches_with_masks`) | `num_matches_after_mask` |
| `after_utm_distance` | matches whose paired UTM coords differ by more than `UTM_DISTANCE_THRESHOLD_METERS` (gross-outlier pre-filter) | `num_matches_after_utm_distance_filter` |
| `ransac_inliers` | 3-DOF Euclidean RANSAC in UTM (`estimate_utm_homography_ransac`); falls back to pixel-domain RANSAC if UTM fails | `num_ransac_inliers` |

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
        ├── funnel_raw.png                 # match-count evolution, one line per method
        ├── funnel_rho_gray.png
        ├── utm_residuals_raw.png          # per-match residual histogram, per method
        ├── utm_residuals_rho_gray.png
        ├── homography_params.png          # tx/ty/θ bar chart across all runs
        ├── summary.csv                    # per-run table (one row per spec, this pair)
        ├── summary.json                   # same data, JSON
        ├── summary_table.png              # rendered table
        ├── runs/
        │   └── <spec_name>/
        │       └── warp_diff.png          # img1, warp(img0, H_pixel), abs-diff
        └── intermediates/                 # everything needed to rerun this pair
            ├── section_01/                # XTF dump, PhysDNet output, masks
            └── section_02/
```

One pair = one self-contained folder. Easy to share, archive, or delete.

`TIMESTAMP_PAIRS` entries can be either `(ts1, ts2)` or `(ts1, ts2, "label")`. Labels become the x-axis ticks on the cross-pair plots, so make them short and descriptive (`easy_parallel`, `hard_anti`, `cross_overlap`). The order of the list is preserved on the cross-pair plots — put pairs in difficulty order (easy → hard) so the plot reads left to right.

## Plot reference

### Filtering funnel (`funnel_<source>.png`)
- X-axis: pipeline stages.
- Y-axis: number of matches (symlog).
- One line per method, with min/max envelope across that method's parameter variants.
- **Read this for**: how much each stage hurts each method. On easy pairs all methods stay high; on hard pairs you'll see one method collapse early (often at the UTM-distance stage if it's matching noise).

### UTM residuals (`utm_residuals_<source>.png`)
- Overlaid histograms of per-inlier `‖H_utm·p0 − p1‖` in metres.
- Method legend shows median + p95.
- **Read this for**: quality of the recovered transform. Lower median = tighter fit. A method with many inliers but a large median residual is likely over-fitting noise.

### Homography parameters (`homography_params.png`)
- Bar chart of `(tx_utm, ty_utm, θ_utm_deg)` per run, coloured by method.
- **Read this for**: agreement across methods. If three methods recover `tx ≈ 12 m` and one says `tx ≈ −3 m`, the outlier is wrong (or matching across opposite directions).

### Warp diff (`runs/<spec>/warp_diff.png`)
- Three panels: img1, warp(img0, H_pixel), absolute pixel difference in the overlap region.
- Title includes mean abs diff, RMSE, overlap fraction.
- One per (method, image_source) best run, selected by lowest median UTM residual.
- **Read this for**: a visual sanity check that the numeric residuals correspond to a believable alignment.

### Summary table (`summary.csv`, `summary_table.png`)
- One row per spec. Columns: counts at each stage, residual stats (mean/median/p95/RMSE), recovered transform, runtime, `h_utm_ok` flag, error string.
- **Use this directly in the report.**

### Cross-pair plots (`cross_pair_*.png`, `all_pairs_summary.csv`)
- Generated automatically by the last cell of `auto_report.ipynb` (calls `generate_cross_pair_report`).
- X-axis: pair label (or `ts1_ts2` if unlabelled), in the order pairs were processed.
- Y-axis: median (across that method's parameter variants) of `residual_median_m` or `n_inliers`.
- One line per method. Separate figure per image source plus a combined "all sources" version.
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
)
```

Any kwarg not in `matcher_kwargs` falls back to the corresponding `*_FILTER_THRESHOLD` / `SIFT_*` / `LIGHTGLUE_*` global in cell 2. Matcher instances are cached by `(backend, frozenset(kwargs))` so duplicate specs across image sources don't re-build the model.

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
4. Visual: the warp-diff "diff" panel should show structure (residual seabed texture) but not large-scale misregistration shadows.
