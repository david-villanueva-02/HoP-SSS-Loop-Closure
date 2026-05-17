"""
Automatic report-plot generation for the SSS feature-matching pipeline.

Module is decoupled from the notebook: it accepts ``make_matcher`` and
``run_registration`` callables built by the notebook so it works with whatever
matcher classes / registration variant the notebook currently defines.

Typical use from a notebook:

    from report_utils import (
        default_sweep_specs, run_sweep, make_report_dir,
        plot_filtering_funnel, plot_utm_residuals, plot_warp_diff,
        plot_homography_params, plot_summary_table,
    )

    specs = default_sweep_specs()

    def make_matcher(spec):
        # notebook closure that knows about LightGlueFeatureMatcher, SIFTMatcher, MinimaMatcher
        ...

    def run_registration(matcher, image_source):
        return run_parametrized_minima_registration(
            matcher=matcher, mode=MATCH_MODE,
            inference0=inference1, inference1=inference2,
            terrain0=terrain1, terrain1=terrain2,
            pings0=pings1, pings1=pings2,
            section0=section1, section1=section2,
            image_source=image_source,
            apply_mask=APPLY_MASK,
            ransac_reproj_threshold_pixel=RANSAC_REPROJ_THRESHOLD_PIXEL,
            ransac_reproj_threshold_utm=RANSAC_REPROJ_THRESHOLD_UTM,
            ransac_max_iters=RANSAC_MAX_ITERS,
            ransac_confidence=RANSAC_CONFIDENCE,
            ransac_outlier_percent=RANSAC_OUTLIER_PERCENT,
            flip_left_image=FLIP_LEFT_INPUT_IMAGE,
            apply_utm_distance_filter=APPLY_UTM_DISTANCE_FILTER,
            utm_distance_threshold=UTM_DISTANCE_THRESHOLD_METERS,
        )

    runs = run_sweep(specs, make_matcher, run_registration)
    out_dir = make_report_dir(output_dir, timestamp1, timestamp2)
    plot_filtering_funnel(runs, out_dir)
    plot_utm_residuals(runs, out_dir)
    plot_homography_params(runs, out_dir)
    plot_summary_table(runs, out_dir)
    for run in select_top_runs(runs):
        plot_warp_diff(run, out_dir)
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Sweep specification
# ---------------------------------------------------------------------------

@dataclass
class SweepSpec:
    name: str                       # short unique id, e.g. "sift_ratio0.75_n8000_raw"
    backend: str                    # "sift" | "lightglue_superpoint" | "minima_sp_lg" | "lightglue_sift"
    matcher_kwargs: dict            # backend-specific kwargs passed to make_matcher
    image_source: str               # "raw" | "rho_gray"
    method_label: str               # display label, e.g. "SIFT"
    variant_label: str = ""         # short label for varied params, e.g. "ratio=0.75"


@dataclass
class RunRecord:
    spec: SweepSpec
    result: Any                     # RegistrationResult
    utm_residuals: np.ndarray       # per-inlier residual in metres (empty if H_utm is None)
    residual_stats: dict            # mean, median, p95, rmse, n_inliers
    runtime_s: float
    error: Optional[str] = None     # populated if the run failed


# ---------------------------------------------------------------------------
# Default sweep specifications
# ---------------------------------------------------------------------------

def default_sweep_specs() -> list[SweepSpec]:
    """
    3 methods x small per-method param grid x {raw, rho_gray}.
    """
    specs: list[SweepSpec] = []

    for image_source in ("raw", "rho_gray"):
        src_short = "raw" if image_source == "raw" else "rho"

        # SIFT: ratio_test x nfeatures
        for ratio in (0.7, 0.75, 0.8):
            for nfeat in (4000, 8000):
                specs.append(SweepSpec(
                    name=f"sift_r{ratio}_n{nfeat}_{src_short}",
                    backend="sift",
                    matcher_kwargs={"ratio_test": ratio, "nfeatures": nfeat},
                    image_source=image_source,
                    method_label="SIFT",
                    variant_label=f"ratio={ratio}, n={nfeat}",
                ))

        # LightGlue + SuperPoint: filter_threshold x max_num_keypoints
        for ft in (0.05, 0.1, 0.2):
            for kp in (2048, 4096):
                specs.append(SweepSpec(
                    name=f"lgsp_ft{ft}_kp{kp}_{src_short}",
                    backend="lightglue_superpoint",
                    matcher_kwargs={"filter_threshold": ft, "max_num_keypoints": kp},
                    image_source=image_source,
                    method_label="LightGlue+SuperPoint",
                    variant_label=f"ft={ft}, kp={kp}",
                ))

        # MINIMA: filter_threshold (passed through if matcher accepts it; otherwise ignored)
        for ft in (0.05, 0.1, 0.2):
            specs.append(SweepSpec(
                name=f"minima_ft{ft}_{src_short}",
                backend="minima_sp_lg",
                matcher_kwargs={"filter_threshold": ft},
                image_source=image_source,
                method_label="MINIMA(sp_lg)",
                variant_label=f"ft={ft}",
            ))

    return specs


def minimal_sweep_specs() -> list[SweepSpec]:
    """
    Smaller sweep for quick smoke tests: 3 methods, default params, both sources.
    """
    out = []
    for image_source in ("raw", "rho_gray"):
        src_short = "raw" if image_source == "raw" else "rho"
        out.append(SweepSpec(f"sift_default_{src_short}", "sift", {},
                             image_source, "SIFT", "default"))
        out.append(SweepSpec(f"lgsp_default_{src_short}", "lightglue_superpoint", {},
                             image_source, "LightGlue+SuperPoint", "default"))
        out.append(SweepSpec(f"minima_default_{src_short}", "minima_sp_lg", {},
                             image_source, "MINIMA(sp_lg)", "default"))
    return out


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    specs: list[SweepSpec],
    make_matcher: Callable[[SweepSpec], Any],
    run_registration: Callable[[Any, str], Any],
    cache_matcher: bool = True,
    verbose: bool = True,
) -> list[RunRecord]:
    """
    Execute each spec end-to-end.

    Matcher reuse: matchers built for one spec are reused for every later spec
    that shares the same ``(backend, frozenset(matcher_kwargs))`` key, since
    LightGlue / MINIMA construction is slow.
    """
    matcher_cache: dict[tuple, Any] = {}
    runs: list[RunRecord] = []

    for i, spec in enumerate(specs):
        if verbose:
            print(f"\n[{i + 1}/{len(specs)}] {spec.name}  ({spec.method_label} | {spec.variant_label} | {spec.image_source})")

        cache_key = (spec.backend, tuple(sorted(spec.matcher_kwargs.items())))

        try:
            if cache_matcher and cache_key in matcher_cache:
                matcher = matcher_cache[cache_key]
            else:
                matcher = make_matcher(spec)
                if cache_matcher:
                    matcher_cache[cache_key] = matcher

            t0 = time.perf_counter()
            result = run_registration(matcher, spec.image_source)
            runtime = time.perf_counter() - t0

            residuals = compute_utm_residuals(result)
            stats = residual_stats(residuals)

            runs.append(RunRecord(
                spec=spec,
                result=result,
                utm_residuals=residuals,
                residual_stats=stats,
                runtime_s=runtime,
            ))

            if verbose:
                print(f"   raw={result.num_matches}  mask={result.num_matches_after_mask}  "
                      f"utm_dist={getattr(result, 'num_matches_after_utm_distance_filter', -1)}  "
                      f"inliers={result.num_ransac_inliers}  "
                      f"residual_median={stats['median']:.3f}m  runtime={runtime:.2f}s")

        except Exception as exc:  # noqa: BLE001 - report-time errors should not abort the sweep
            if verbose:
                print(f"   FAILED: {type(exc).__name__}: {exc}")
            runs.append(RunRecord(
                spec=spec,
                result=None,
                utm_residuals=np.empty(0),
                residual_stats=residual_stats(np.empty(0)),
                runtime_s=0.0,
                error=f"{type(exc).__name__}: {exc}",
            ))

    return runs


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

def compute_utm_residuals(result) -> np.ndarray:
    """
    Per-match UTM residual (metres) for the final RANSAC inliers, computed by
    applying H_utm to p0 and measuring distance to p1.
    """
    if result is None or result.H_utm is None:
        return np.empty(0, dtype=np.float64)

    p0 = np.asarray(result.mkpts0_utm_ransac, dtype=np.float64)
    p1 = np.asarray(result.mkpts1_utm_ransac, dtype=np.float64)

    if len(p0) == 0:
        return np.empty(0, dtype=np.float64)

    # ransac_inliers_utm is full-length (over mkpts_kept). Project to ransac input length.
    if result.ransac_inliers_utm is not None and result.utm_distance_keep_mask is not None:
        inliers_ransac = np.asarray(result.ransac_inliers_utm, dtype=bool)[result.utm_distance_keep_mask]
    else:
        inliers_ransac = np.ones(len(p0), dtype=bool)

    p0 = p0[inliers_ransac]
    p1 = p1[inliers_ransac]

    if len(p0) == 0:
        return np.empty(0, dtype=np.float64)

    p0_h = np.hstack([p0, np.ones((len(p0), 1))])
    pred1 = (result.H_utm @ p0_h.T).T
    w = pred1[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1.0, w)
    pred1 = pred1[:, :2] / w

    return np.linalg.norm(p1 - pred1, axis=1)


def residual_stats(residuals: np.ndarray) -> dict:
    if residuals.size == 0:
        return {"mean": np.nan, "median": np.nan, "p95": np.nan, "rmse": np.nan, "n_inliers": 0}
    return {
        "mean": float(np.mean(residuals)),
        "median": float(np.median(residuals)),
        "p95": float(np.percentile(residuals, 95)),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "n_inliers": int(len(residuals)),
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def make_report_dir(output_dir: str | Path, ts1: int, ts2: int,
                    label: Optional[str] = None) -> Path:
    name = f"{label}_{ts1}_{ts2}" if label else f"{ts1}_{ts2}"
    out = Path(output_dir) / "report" / name
    (out / "runs").mkdir(parents=True, exist_ok=True)
    return out


def _parse_pair_folder_name(name: str) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Parse a pair-report folder name into (ts1, ts2, label).

    Accepted formats:
        "{ts1}_{ts2}"               -> (ts1, ts2, None)
        "{label}_{ts1}_{ts2}"       -> (ts1, ts2, label)  (label may contain underscores)

    Returns (None, None, name) if the trailing two parts aren't integers.
    """
    parts = name.split("_")
    if len(parts) >= 2:
        try:
            ts2 = int(parts[-1])
            ts1 = int(parts[-2])
            label = "_".join(parts[:-2]) or None
            return ts1, ts2, label
        except ValueError:
            pass
    return None, None, name


@dataclass
class PairSummary:
    """One row per pair, loaded from the per-pair summary.json."""
    folder_name: str
    label: Optional[str]
    ts1: Optional[int]
    ts2: Optional[int]
    rows: list[dict]            # the list of run dicts as written by plot_summary_table
    folder: Path


def load_pair_summaries(report_root: str | Path,
                        pair_order: Optional[list[str]] = None) -> list[PairSummary]:
    """
    Scan ``report_root`` for per-pair folders containing ``summary.json`` and
    return them as ``PairSummary`` objects.

    If ``pair_order`` is given, only those folder names are loaded, in the order
    given. Otherwise folders are loaded in alphabetical order.
    """
    report_root = Path(report_root)
    if not report_root.is_dir():
        return []

    if pair_order is None:
        candidates = sorted(p for p in report_root.iterdir() if p.is_dir())
    else:
        candidates = [report_root / name for name in pair_order]

    out = []
    for folder in candidates:
        sj = folder / "summary.json"
        if not sj.exists():
            continue
        rows = json.loads(sj.read_text())
        ts1, ts2, label = _parse_pair_folder_name(folder.name)
        out.append(PairSummary(folder_name=folder.name, label=label,
                               ts1=ts1, ts2=ts2, rows=rows, folder=folder))
    return out


def write_cross_pair_summary_csv(summaries: list[PairSummary],
                                 out_path: str | Path) -> Path:
    """Flatten every per-run dict into one CSV with pair columns prepended."""
    out_path = Path(out_path)
    flat = []
    for s in summaries:
        for r in s.rows:
            flat.append({
                "pair_folder": s.folder_name,
                "pair_label": s.label or "",
                "ts1": s.ts1, "ts2": s.ts2,
                **r,
            })

    fields = (["pair_folder", "pair_label", "ts1", "ts2"] + _SUMMARY_FIELDS)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in flat:
            writer.writerow(row)
    return out_path


def _aggregate(values: list[float], how: str = "median") -> float:
    cleaned = [v for v in values
               if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not cleaned:
        return float("nan")
    arr = np.asarray(cleaned, dtype=float)
    return float(np.median(arr) if how == "median" else np.mean(arr))


def plot_cross_pair_metric(summaries: list[PairSummary],
                           metric_key: str,
                           ylabel: str,
                           out_path: str | Path,
                           agg: str = "median",
                           source_filter: Optional[str] = None,
                           lower_is_better: bool = True) -> Optional[Path]:
    """
    One line per method across pairs.

    X-axis: pair labels (or "ts1_ts2" when no label), preserving the order
    summaries were loaded in (so the caller controls easy→hard ordering).
    Y-axis: ``agg`` of ``metric_key`` across that method's parameter variants.
    """
    if not summaries:
        return None

    methods = sorted({r["method"] for s in summaries for r in s.rows})
    colors = _color_cycle(len(methods))
    pair_labels = [s.label or f"{s.ts1}_{s.ts2}" for s in summaries]
    x = np.arange(len(summaries))

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(summaries)), 5))
    for color, method in zip(colors, methods):
        ys = []
        for s in summaries:
            vals = [r.get(metric_key) for r in s.rows
                    if r.get("method") == method
                    and (source_filter is None or r.get("image_source") == source_filter)]
            ys.append(_aggregate(vals, agg))
        ax.plot(x, ys, "-o", color=color, label=method, linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    direction = "lower is better" if lower_is_better else "higher is better"
    src_str = source_filter if source_filter else "all sources"
    ax.set_title(f"{ylabel} across pairs  ({agg} over variants, {direction}, {src_str})")
    fig.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close(fig)
    return out_path


def generate_cross_pair_report(report_root: str | Path,
                               out_dir: Optional[str | Path] = None,
                               pair_order: Optional[list[str]] = None,
                               image_sources: tuple[str, ...] = ("raw", "rho_gray"),
                               ) -> dict[str, Any]:
    """
    Build the cross-pair artefacts (one CSV + a few line plots) from every
    per-pair ``summary.json`` under ``report_root``.

    Returns the dict of saved paths. Pass ``pair_order`` to control left→right
    pair ordering on the plots (e.g. easy → hard).
    """
    report_root = Path(report_root)
    out_dir = Path(out_dir) if out_dir else report_root
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_pair_summaries(report_root, pair_order=pair_order)
    if not summaries:
        print(f"No per-pair summaries found under {report_root}")
        return {}

    print(f"Aggregating {len(summaries)} pairs:")
    for s in summaries:
        print(f"  - {s.folder_name}  ({len(s.rows)} runs)")

    paths: dict[str, Any] = {
        "csv": write_cross_pair_summary_csv(summaries, out_dir / "all_pairs_summary.csv"),
    }

    for src in image_sources:
        paths[f"residual_{src}"] = plot_cross_pair_metric(
            summaries, "residual_median_m", "Median UTM residual (m)",
            out_dir / f"cross_pair_residual_{src}.png",
            source_filter=src, lower_is_better=True,
        )
        paths[f"inliers_{src}"] = plot_cross_pair_metric(
            summaries, "n_inliers", "RANSAC inliers",
            out_dir / f"cross_pair_inliers_{src}.png",
            source_filter=src, agg="median", lower_is_better=False,
        )

    paths["residual_combined"] = plot_cross_pair_metric(
        summaries, "residual_median_m", "Median UTM residual (m)",
        out_dir / "cross_pair_residual_combined.png",
        source_filter=None, lower_is_better=True,
    )
    paths["inliers_combined"] = plot_cross_pair_metric(
        summaries, "n_inliers", "RANSAC inliers",
        out_dir / "cross_pair_inliers_combined.png",
        source_filter=None, lower_is_better=False,
    )

    return paths


def select_top_runs(runs: list[RunRecord], by: str = "median") -> list[RunRecord]:
    """
    Return the best run per (method_label, image_source) by lowest residual `by`.
    Used to pick a small number of runs for warp-diff visualisation.
    """
    grouped: dict[tuple, RunRecord] = {}
    for run in runs:
        if run.error or run.result is None or run.result.H_utm is None:
            continue
        key = (run.spec.method_label, run.spec.image_source)
        score = run.residual_stats.get(by, np.inf)
        if np.isnan(score):
            score = np.inf
        prev = grouped.get(key)
        if prev is None or score < prev.residual_stats.get(by, np.inf):
            grouped[key] = run
    return list(grouped.values())


# ---------------------------------------------------------------------------
# Plot: filtering funnel
# ---------------------------------------------------------------------------

_STAGES = ("raw", "after_mask", "after_utm_distance", "ransac_inliers")
_STAGE_LABELS = ("Raw", "After mask", "After UTM dist.", "RANSAC inliers")


def _funnel_counts(result) -> list[int]:
    if result is None:
        return [0, 0, 0, 0]
    return [
        int(result.num_matches),
        int(result.num_matches_after_mask),
        int(getattr(result, "num_matches_after_utm_distance_filter", result.num_matches_after_mask)),
        int(result.num_ransac_inliers),
    ]


def _color_cycle(n: int) -> list:
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def plot_filtering_funnel(runs: list[RunRecord], out_dir: Path,
                          group_by_image_source: bool = True) -> list[Path]:
    """
    Histogram-style plot of the number of matches at each filtering stage,
    one line per method (averaged across that method's parameter variants) and
    one figure per image_source.

    Saves PNGs to ``out_dir/funnel_<image_source>.png``.
    """
    out_dir = Path(out_dir)
    saved: list[Path] = []

    if group_by_image_source:
        sources = sorted({r.spec.image_source for r in runs if r.result is not None})
    else:
        sources = [None]

    for src in sources:
        runs_src = [r for r in runs if r.result is not None and (src is None or r.spec.image_source == src)]
        if not runs_src:
            continue

        methods = sorted({r.spec.method_label for r in runs_src})
        colors = _color_cycle(len(methods))

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(_STAGES))

        for color, method in zip(colors, methods):
            method_runs = [r for r in runs_src if r.spec.method_label == method]
            counts = np.array([_funnel_counts(r.result) for r in method_runs], dtype=float)
            mean = counts.mean(axis=0)
            lo = counts.min(axis=0)
            hi = counts.max(axis=0)

            ax.plot(x, mean, "-o", color=color, label=f"{method} (n={len(method_runs)})", linewidth=2)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15)

        ax.set_xticks(x)
        ax.set_xticklabels(_STAGE_LABELS)
        ax.set_ylabel("Number of matches")
        title_src = f"image_source = {src}" if src else "all image sources"
        ax.set_title(f"Match-count evolution through the pipeline ({title_src})")
        ax.set_yscale("symlog", linthresh=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        suffix = f"_{src}" if src else ""
        path = out_dir / f"funnel{suffix}.png"
        fig.savefig(path, dpi=200)
        plt.show()
        plt.close(fig)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Plot: UTM residual histograms
# ---------------------------------------------------------------------------

def plot_utm_residuals(runs: list[RunRecord], out_dir: Path,
                       group_by_image_source: bool = True) -> list[Path]:
    """
    Overlaid histograms of per-match UTM residuals (metres) for the final
    RANSAC inlier set. One figure per image_source.
    """
    out_dir = Path(out_dir)
    saved: list[Path] = []

    if group_by_image_source:
        sources = sorted({r.spec.image_source for r in runs if r.utm_residuals.size > 0})
    else:
        sources = [None]

    for src in sources:
        runs_src = [r for r in runs
                    if r.utm_residuals.size > 0 and (src is None or r.spec.image_source == src)]
        if not runs_src:
            continue

        methods = sorted({r.spec.method_label for r in runs_src})
        colors = _color_cycle(len(methods))

        all_res = np.concatenate([r.utm_residuals for r in runs_src])
        upper = max(np.percentile(all_res, 99), 1e-3) if all_res.size else 1.0
        bins = np.linspace(0, upper, 40)

        fig, ax = plt.subplots(figsize=(10, 6))

        for color, method in zip(colors, methods):
            method_runs = [r for r in runs_src if r.spec.method_label == method]
            pooled = np.concatenate([r.utm_residuals for r in method_runs])
            if pooled.size == 0:
                continue
            stats = residual_stats(pooled)
            label = (f"{method}  median={stats['median']:.3f} m  "
                     f"p95={stats['p95']:.3f} m  n={stats['n_inliers']}")
            ax.hist(pooled, bins=bins, color=color, alpha=0.45, label=label, edgecolor="none")

        ax.set_xlabel("UTM residual after H_utm (metres)")
        ax.set_ylabel("Inlier count")
        title_src = f"image_source = {src}" if src else "all image sources"
        ax.set_title(f"Per-match UTM residuals ({title_src})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        suffix = f"_{src}" if src else ""
        path = out_dir / f"utm_residuals{suffix}.png"
        fig.savefig(path, dpi=200)
        plt.show()
        plt.close(fig)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Plot: warp difference
# ---------------------------------------------------------------------------

def plot_warp_diff(run: RunRecord, out_dir: Path) -> Optional[Path]:
    """
    For one run: img1 vs warp(img0, H_pixel) and their absolute difference
    in the overlap region. One PNG per run, saved under
    ``out_dir/runs/<spec_name>/warp_diff.png``.
    """
    if run.error or run.result is None or run.result.H_pixel is None:
        return None

    out_dir = Path(out_dir)
    run_dir = out_dir / "runs" / run.spec.name
    run_dir.mkdir(parents=True, exist_ok=True)

    img0 = _to_gray_u8(run.result.img0)
    img1 = _to_gray_u8(run.result.img1)
    H = run.result.H_pixel

    h, w = img1.shape[:2]
    warped = cv2.warpPerspective(img0, H, (w, h))

    overlap = (warped > 0) & (img1 > 0)
    diff = np.zeros_like(img1, dtype=np.uint8)
    if overlap.any():
        diff_full = cv2.absdiff(warped, img1)
        diff[overlap] = diff_full[overlap]

    mean_abs = float(diff[overlap].mean()) if overlap.any() else float("nan")
    rmse = float(np.sqrt((diff[overlap].astype(np.float32) ** 2).mean())) if overlap.any() else float("nan")
    overlap_frac = float(overlap.mean())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title("Image 1")
    axes[0].axis("off")

    axes[1].imshow(warped, cmap="gray")
    axes[1].set_title("Warp(Image 0, H_pixel)")
    axes[1].axis("off")

    im = axes[2].imshow(diff, cmap="magma", vmin=0, vmax=255)
    axes[2].set_title(f"|diff| in overlap  mean={mean_abs:.1f}  rmse={rmse:.1f}  overlap={overlap_frac:.2%}")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"{run.spec.name}  ({run.spec.method_label} | {run.spec.variant_label} | {run.spec.image_source})")
    fig.tight_layout()

    path = run_dir / "warp_diff.png"
    fig.savefig(path, dpi=200)
    plt.show()
    plt.close(fig)
    return path


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        a_min, a_max = float(a.min()), float(a.max())
        if a_max - a_min > 1e-12:
            a = (a - a_min) / (a_max - a_min) * 255.0
        else:
            a = np.zeros_like(a)
        arr = a.astype(np.uint8)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr


# ---------------------------------------------------------------------------
# Plot: homography parameters
# ---------------------------------------------------------------------------

def plot_homography_params(runs: list[RunRecord], out_dir: Path) -> Optional[Path]:
    """
    Bar chart of recovered (tx_utm, ty_utm, theta_utm_deg) across all successful
    runs. Useful to spot methods that disagree on the recovered transform.
    """
    successful = [r for r in runs if r.result is not None and r.result.H_utm is not None]
    if not successful:
        return None

    out_dir = Path(out_dir)
    labels = [r.spec.name for r in successful]
    tx = [r.result.tx_utm for r in successful]
    ty = [r.result.ty_utm for r in successful]
    th = [r.result.theta_utm_deg for r in successful]

    method_color = {m: c for m, c in zip(
        sorted({r.spec.method_label for r in successful}),
        _color_cycle(len({r.spec.method_label for r in successful})),
    )}
    colors = [method_color[r.spec.method_label] for r in successful]

    fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.35 * len(labels)), 10), sharex=True)
    for ax, vals, name in zip(axes, (tx, ty, th), ("tx_utm [m]", "ty_utm [m]", "theta_utm [deg]")):
        ax.bar(range(len(labels)), vals, color=colors)
        ax.set_ylabel(name)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    axes[-1].set_xticks(range(len(labels)))
    axes[-1].set_xticklabels(labels, rotation=75, ha="right", fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in method_color.values()]
    fig.legend(handles, list(method_color.keys()), loc="upper right", fontsize=9)
    fig.suptitle("Recovered UTM homography parameters by run")
    fig.tight_layout()

    path = out_dir / "homography_params.png"
    fig.savefig(path, dpi=200)
    plt.show()
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Summary table (CSV / JSON / PNG)
# ---------------------------------------------------------------------------

_SUMMARY_FIELDS = [
    "name", "method", "variant", "image_source", "backend",
    "n_raw", "n_after_mask", "n_after_utm_dist", "n_inliers",
    "residual_mean_m", "residual_median_m", "residual_p95_m", "residual_rmse_m",
    "tx_utm_m", "ty_utm_m", "theta_utm_deg",
    "runtime_s", "h_utm_ok", "error",
]


def _row_for(run: RunRecord) -> dict:
    r = run.result
    row = {
        "name": run.spec.name,
        "method": run.spec.method_label,
        "variant": run.spec.variant_label,
        "image_source": run.spec.image_source,
        "backend": run.spec.backend,
        "n_raw": r.num_matches if r is not None else 0,
        "n_after_mask": r.num_matches_after_mask if r is not None else 0,
        "n_after_utm_dist": (getattr(r, "num_matches_after_utm_distance_filter", 0) if r is not None else 0),
        "n_inliers": r.num_ransac_inliers if r is not None else 0,
        "residual_mean_m": run.residual_stats["mean"],
        "residual_median_m": run.residual_stats["median"],
        "residual_p95_m": run.residual_stats["p95"],
        "residual_rmse_m": run.residual_stats["rmse"],
        "tx_utm_m": (r.tx_utm if r is not None else None),
        "ty_utm_m": (r.ty_utm if r is not None else None),
        "theta_utm_deg": (r.theta_utm_deg if r is not None else None),
        "runtime_s": run.runtime_s,
        "h_utm_ok": bool(r is not None and r.H_utm is not None),
        "error": run.error or "",
    }
    return row


def plot_summary_table(runs: list[RunRecord], out_dir: Path) -> dict[str, Path]:
    """
    Save the per-run summary as ``summary.csv``, ``summary.json``, and a PNG
    table rendering. Returns a dict of the saved paths.
    """
    out_dir = Path(out_dir)
    rows = [_row_for(r) for r in runs]

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = out_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2, default=_json_default)

    fig_rows = [[_fmt_cell(row[k]) for k in _SUMMARY_FIELDS] for row in rows]
    fig_width = min(28, 1.0 + 0.9 * len(_SUMMARY_FIELDS))
    fig_height = min(40, 0.4 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=fig_rows,
        colLabels=_SUMMARY_FIELDS,
        cellLoc="center",
        loc="upper left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.2)
    ax.set_title("Sweep summary", pad=12)
    fig.tight_layout()

    png_path = out_dir / "summary_table.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return {"csv": csv_path, "json": json_path, "png": png_path}


def _fmt_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return f"{v:.4g}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if np.isnan(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON-serialisable: {type(o)}")


# ---------------------------------------------------------------------------
# One-shot driver
# ---------------------------------------------------------------------------

def generate_full_report(
    runs: list[RunRecord],
    out_dir: Path,
) -> dict[str, Any]:
    """
    Convenience wrapper that calls every plot function and returns the saved
    file paths.
    """
    out_dir = Path(out_dir)
    out: dict[str, Any] = {}
    out["funnel"] = plot_filtering_funnel(runs, out_dir)
    out["utm_residuals"] = plot_utm_residuals(runs, out_dir)
    out["homography_params"] = plot_homography_params(runs, out_dir)
    out["summary"] = plot_summary_table(runs, out_dir)
    out["warp_diff"] = [plot_warp_diff(r, out_dir) for r in select_top_runs(runs)]
    return out
