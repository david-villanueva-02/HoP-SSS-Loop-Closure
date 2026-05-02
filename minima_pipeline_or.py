from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import pyxtf 
import cv2
import numpy as np
from physdnet.xtf_utils import calculate_swath_positions

from hands_on_utils.MINIMA.load_model import load_model

@dataclass
class MatchResult:
    mkpts0: np.ndarray         # (N, 2)
    mkpts1: np.ndarray         # (N, 2)
    mconf: np.ndarray          # (N,)
    keep_mask: np.ndarray      # (N,) bool after your mask filtering
    H: Optional[np.ndarray]    # (3, 3) or None
    ransac_inliers: Optional[np.ndarray]  # (K,) bool or None after homography
    mkpts0_kept: np.ndarray    # filtered by your masks, before RANSAC
    mkpts1_kept: np.ndarray
    mconf_kept: np.ndarray


def build_minima_args(
    method: str = "sp_lg",
    ckpt: Optional[str] = None,
    save_dir: str = "./outputs",
):
    """
    Build a minimal args object compatible with MINIMA's load_model().
    Extend this if you want loftr/xoftr/roma.
    """
    args = SimpleNamespace()
    args.method = method
    args.save_dir = save_dir
    args.exp_name = "custom_pipeline"

    if method == "sp_lg":
        args.ckpt = ckpt or "./weights/minima_lightglue.pth"

    elif method == "loftr":
        args.ckpt = ckpt or "./weights/minima_loftr.ckpt"
        args.thr = 0.2

    elif method == "xoftr":
        args.ckpt = ckpt or "./weights/weights_xoftr_640.ckpt"
        args.match_threshold = 0.3
        args.fine_threshold = 0.1

    elif method == "roma":
        args.ckpt = ckpt or "./weights/minima_roma.pth"
        args.ckpt2 = "large"

    else:
        raise ValueError(f"Unsupported method: {method}")

    return args


class MinimaMatcher:
    """
    Wrapper around MINIMA.

    - use_path=True  -> matcher expects image paths
    - use_path=False -> matcher expects OpenCV images in memory
    """
    def __init__(
        self,
        method: str = "sp_lg",
        ckpt: Optional[str] = None,
        use_path: bool = False,
    ) -> None:
        self.method = method
        self.args = build_minima_args(method=method, ckpt=ckpt)
        self.matcher = load_model(method, self.args, use_path=use_path)

    def match_paths(self, img0_path: str, img1_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For use_path=True only.
        Returns matched points and confidence scores.
        """
        result = self.matcher(img0_path, img1_path)
        return self._normalize_output(result)

    def match_images(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For use_path=False only.
        img0 and img1 are OpenCV images (BGR uint8) or grayscale uint8 arrays.
        """
        result = self.matcher(img0, img1)
        return self._normalize_output(result)

    @staticmethod
    def _normalize_output(result: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MINIMA wrappers can expose either:
        - keypoints0 / keypoints1 / matching_scores
        or
        - mkpts0 / mkpts1 / mconf
        depending on the wrapper path.
        """
        if "keypoints0" in result:
            mkpts0 = result["keypoints0"]
            mkpts1 = result["keypoints1"]
            mconf = result["matching_scores"]
        else:
            mkpts0 = result["mkpts0"]
            mkpts1 = result["mkpts1"]
            mconf = result["mconf"]

        mkpts0 = np.asarray(mkpts0, dtype=np.float32)
        mkpts1 = np.asarray(mkpts1, dtype=np.float32)
        mconf = np.asarray(mconf, dtype=np.float32).reshape(-1)

        return mkpts0, mkpts1, mconf


def ensure_gray(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def point_mask_test(points_xy: np.ndarray, mask: np.ndarray, threshold: int = 0) -> np.ndarray:
    """
    Returns a boolean mask telling whether each point falls inside a valid region.

    points_xy: (N, 2) in pixel coordinates [x, y]
    mask: 2D uint8 mask where valid pixels are > threshold
    """
    mask = ensure_gray(mask)
    h, w = mask.shape[:2]

    x = np.round(points_xy[:, 0]).astype(int)
    y = np.round(points_xy[:, 1]).astype(int)

    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)

    valid = np.zeros(len(points_xy), dtype=bool)
    valid[inside] = mask[y[inside], x[inside]] == threshold
    return valid


def filter_matches_with_masks(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    mconf: np.ndarray,
    mask0: Optional[np.ndarray] = None,
    mask1: Optional[np.ndarray] = None,
    threshold0: int = 0,
    threshold1: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep matches that survive both masks.
    """
    keep = np.ones(len(mkpts0), dtype=bool)

    if mask0 is not None:
        keep &= point_mask_test(mkpts0, mask0, threshold=threshold0)

    if mask1 is not None:
        keep &= point_mask_test(mkpts1, mask1, threshold=threshold1)

    return mkpts0[keep], mkpts1[keep], mconf[keep], keep


def estimate_homography_ransac(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    ransac_reproj_threshold: float = 3.0,
    max_iters: int = 10_000,   # 10k iterations according to the paper
    confidence: float = 0.995, 
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate homography with OpenCV RANSAC.
    Returns:
      H: (3, 3) or None
      inliers: (N,) bool or None
    """
    if len(mkpts0) < 4 or len(mkpts1) < 4:
        return None, None

    H, inliers = cv2.findHomography(
        mkpts0,
        mkpts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        maxIters=max_iters,
        confidence=confidence,
    )

    if inliers is None:
        return H, None

    return H, inliers.reshape(-1).astype(bool)


def run_pipeline_on_images(
    matcher: MinimaMatcher,
    img0: np.ndarray,
    img1: np.ndarray,
    pings0: list[pyxtf.XTFHeaderType] = None, 
    pings1: list[pyxtf.XTFHeaderType] = None,
    mask0: Optional[np.ndarray] = None,
    mask1: Optional[np.ndarray] = None,
    ransac_reproj_threshold: float = 8.0,
    ransac_max_iters: int = 10_000,
    ransac_confidence: float = 0.995,
) -> MatchResult:
    """
    Full pipeline for in-memory images:
      image pair -> MINIMA matching -> mask filtering -> homography with RANSAC
    """
    # Get matches from SuperPoint + LightGlue + MINIMA
    mkpts0, mkpts1, mconf = matcher.match_images(img0, img1)

    # Filter matches with masks
    mkpts0_kept, mkpts1_kept, mconf_kept, keep_mask = filter_matches_with_masks(
        mkpts0, mkpts1, mconf, mask0=mask0, mask1=mask1
    )

    # Convert matches into UTM coordinates
    if pings0 is not None and pings1 is not None: 
        
        pass
    

    # Estimate homography
    H, ransac_inliers = estimate_homography_ransac(
        mkpts0_kept, mkpts1_kept, ransac_reproj_threshold=ransac_reproj_threshold, max_iters=ransac_max_iters, confidence=ransac_confidence
    )

    return MatchResult(
        mkpts0=mkpts0,
        mkpts1=mkpts1,
        mconf=mconf,
        keep_mask=keep_mask,
        H=H,
        ransac_inliers=ransac_inliers,
        mkpts0_kept=mkpts0_kept,
        mkpts1_kept=mkpts1_kept,
        mconf_kept=mconf_kept,
    )


def run_pipeline_on_paths(
    matcher: MinimaMatcher,
    img0_path: str,
    img1_path: str,
    mask0: Optional[np.ndarray] = None,
    mask1: Optional[np.ndarray] = None,
    ransac_reproj_threshold: float = 3.0,
) -> MatchResult:
    """
    Same pipeline, but with path-based loading through MINIMA.
    """
    mkpts0, mkpts1, mconf = matcher.match_paths(img0_path, img1_path)

    mkpts0_kept, mkpts1_kept, mconf_kept, keep_mask = filter_matches_with_masks(
        mkpts0, mkpts1, mconf, mask0=mask0, mask1=mask1
    )

    H, ransac_inliers = estimate_homography_ransac(
        mkpts0_kept, mkpts1_kept, ransac_reproj_threshold=ransac_reproj_threshold
    )

    return MatchResult(
        mkpts0=mkpts0,
        mkpts1=mkpts1,
        mconf=mconf,
        keep_mask=keep_mask,
        H=H,
        ransac_inliers=ransac_inliers,
        mkpts0_kept=mkpts0_kept,
        mkpts1_kept=mkpts1_kept,
        mconf_kept=mconf_kept,
    )


def draw_matches(
    img0: np.ndarray,
    img1: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    max_draw: int = 300,
) -> np.ndarray:
    """
    Simple OpenCV visualization.
    """
    if inlier_mask is not None:
        mkpts0 = mkpts0[inlier_mask]
        mkpts1 = mkpts1[inlier_mask]

    if len(mkpts0) > max_draw:
        idx = np.random.choice(len(mkpts0), size=max_draw, replace=False)
        mkpts0 = mkpts0[idx]
        mkpts1 = mkpts1[idx]

    kpts0 = [cv2.KeyPoint(float(x), float(y), 4) for x, y in mkpts0]
    kpts1 = [cv2.KeyPoint(float(x), float(y), 4) for x, y in mkpts1]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(kpts0))]

    vis = cv2.drawMatches(
        img0, kpts0,
        img1, kpts1,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis