# HoP-SSS-Loop-Closure

Compare and evaluate feature-matching methods for **loop closure on side-scan sonar (SSS)** imagery.
Given a pair of pings (timestamps) from an XTF survey, the pipeline extracts the two sonar sections,
runs PhysDNet inference and masking, then matches them with several backends and reports how well each
recovers the relative transform.

**Matchers compared:** SIFT (OpenCV), LightGlue + SuperPoint, LightGlue + SIFT, and
MINIMA (SuperPoint + LightGlue).

The deliverable is the batch driver [`notebooks/auto_report.ipynb`](notebooks/auto_report.ipynb): you
list timestamp pairs, run all cells, and get one folder of comparison plots and tables per pair, plus
cross-pair summaries. [`notebooks/configurable_backend_pipeline2.ipynb`](notebooks/configurable_backend_pipeline2.ipynb)
is the interactive single-pair companion for exploring one pair at a time.

## Repository layout

```
README.md                  This file
requirements.txt           Python dependencies (LightGlue/MINIMA installed separately)
docs/PIPELINE.md           Full pipeline reference: parameters, sweep, outputs, plots
notebooks/
  auto_report.ipynb                     Batch driver (main deliverable)
  configurable_backend_pipeline2.ipynb  Interactive single-pair pipeline
src/                       Importable modules (kept on sys.path by the notebooks)
  prep.py              XTF -> sonar sections
  inference.py         PhysDNet inference
  train.py             PhysDNet model definition (used by inference.py)
  masks.py             Terrain / shadow / blind-zone masks
  xtf_utils.py         XTF parsing, swath geometry (UTM)
  minima_pipeline.py   Matching + mask filter + pixel/UTM RANSAC registration
  report_utils.py      Sweep runner + plotting / report generation
  RANSAC/              Custom homography RANSAC solvers
data/                      Put your .xtf survey files here (git-ignored; see data/README.md)
weights/                   Put PhysDNet .pth weights here (git-ignored; see weights/README.md)
LightGlue/  MINIMA/        External repos you clone yourself (git-ignored; see Setup)
output_simplified/         Generated reports (git-ignored)
```

## Setup

1. **Python environment** (Python 3.10+ recommended):
   ```bash
   python -m venv .sss_venv && source .sss_venv/bin/activate
   pip install -r requirements.txt
   ```
   Install the PyTorch build that matches your CUDA version — see <https://pytorch.org/get-started/>.

2. **LightGlue** (matcher backend) — clone at the repo root and install:
   ```bash
   git clone https://github.com/cvg/LightGlue.git
   pip install -e LightGlue
   ```

3. **MINIMA** (matcher backend) — clone at the repo root and fetch its weights:
   ```bash
   git clone https://github.com/LSXI7/MINIMA.git
   # then download minima_lightglue.pth into MINIMA/weights/ (see MINIMA's README)
   ```
   MINIMA isn't pip-installable; [`src/minima_pipeline.py`](src/minima_pipeline.py) adds it to
   `sys.path` automatically at import time.

4. **PhysDNet weights** — place the checkpoint in `weights/`:
   ```
   weights/best_model_v610_jaguar.pth
   ```
   See [weights/README.md](weights/README.md).

5. **Data** — place your XTF survey(s) in `data/`:
   ```
   data/2025-09-24_09-25-24_0.xtf
   ```
   See [data/README.md](data/README.md).

## Quickstart

1. Open [`notebooks/auto_report.ipynb`](notebooks/auto_report.ipynb).
2. Set `xtf_file` (config cell) to your survey filename inside `data/`.
3. Edit `TIMESTAMP_PAIRS` — a list of `(ts1, ts2, "label")` ping-index pairs to compare.
4. (Optional) set `SWEEP_SPECS = report_utils.minimal_sweep_specs()` for a fast smoke test.
5. **Run all cells.** Each pair produces a folder under
   `output_simplified/report/<label>_<ts1>_<ts2>/` with funnel / residual / homography plots, per-spec
   match visualizations, and `summary.csv`; the final cell writes cross-pair comparison plots and
   `all_pairs_summary.csv`.

See [`docs/PIPELINE.md`](docs/PIPELINE.md) for the full parameter reference, the default sweep, the
filtering stages, and how to read every plot.

## Choosing timestamps

`TIMESTAMP_PAIRS` are ping indices into the loaded survey. The notebook keeps only straight-line
segments (turns are dropped), so a timestamp that lands inside a turn is skipped. See
[`docs/PIPELINE.md`](docs/PIPELINE.md) → "Picking timestamps" for the overlap / heading metrics used to
pick interesting easy and hard pairs.

> **Future work:** a helper to discover good timestamp pairs automatically (instead of choosing them by
> hand) is not yet included.
