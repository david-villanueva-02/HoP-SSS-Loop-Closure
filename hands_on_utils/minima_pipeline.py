from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import pyxtf
import matplotlib.pyplot as plt

from MINIMA.load_model import load_model

from RANSAC.compute_homography_ransac import compute_homography_ransac
from RANSAC.projection_error import projection_error
from xtf_utils import calculate_swath_positions


from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class RegistrationResult:
    mode: str
    image_source: str

    img0: np.ndarray
    img1: np.ndarray
    mask0: Optional[np.ndarray]
    mask1: Optional[np.ndarray]

    mkpts0: np.ndarray
    mkpts1: np.ndarray
    mconf: np.ndarray
    keep_mask: np.ndarray

    mkpts0_kept: np.ndarray
    mkpts1_kept: np.ndarray
    mconf_kept: np.ndarray

    # UTM distance pre-filter information.
    # mkpts*_utm contain all matches after image/mask filtering.
    # mkpts*_utm_ransac contain the subset passed to UTM RANSAC after
    # optional distance filtering.
    apply_utm_distance_filter: bool
    utm_distance_threshold: Optional[float]
    utm_match_distances: Optional[np.ndarray]
    utm_distance_keep_mask: Optional[np.ndarray]
    mkpts0_utm_ransac: Optional[np.ndarray]
    mkpts1_utm_ransac: Optional[np.ndarray]

    mkpts0_utm: Optional[np.ndarray]
    mkpts1_utm: Optional[np.ndarray]
    mkpts0_utm_local: Optional[np.ndarray]
    mkpts1_utm_local: Optional[np.ndarray]

    H: Optional[np.ndarray]
    H_pixel: Optional[np.ndarray]
    H_utm: Optional[np.ndarray]
    H_utm_local: Optional[np.ndarray]

    tx_utm: Optional[float]
    ty_utm: Optional[float]
    theta_utm: Optional[float]
    theta_utm_deg: Optional[float]

    tx_utm_local: Optional[float]
    ty_utm_local: Optional[float]
    theta_utm_local: Optional[float]
    theta_utm_local_deg: Optional[float]

    ransac_inliers: Optional[np.ndarray]
    ransac_inliers_pixel: Optional[np.ndarray]
    ransac_inliers_utm: Optional[np.ndarray]

    utm_offset0: Optional[np.ndarray]
    utm_offset1: Optional[np.ndarray]

    num_matches: int
    num_matches_after_mask: int
    num_matches_after_utm_distance_filter: int
    num_utm_distance_rejected: int
    num_ransac_inliers: int


def build_minima_args(
    method: str = "sp_lg",
    ckpt: Optional[str] = None,
    save_dir: str = "./outputs",
):
    """
    Build a minimal args object compatible with MINIMA's load_model().
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
        self.use_path = use_path
        self.args = build_minima_args(method=method, ckpt=ckpt)
        self.matcher = load_model(method, self.args, use_path=use_path)

    def match_images(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        img0 and img1 are OpenCV images or grayscale uint8 arrays.
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


def get_key_by_side(data_dict: dict, side: str) -> str:
    """
    Finds the dictionary key containing 'left' or 'right'.
    """
    side = side.lower()

    for key in data_dict.keys():
        if side in key.lower():
            return key

    raise KeyError(f"Could not find side='{side}' in keys: {list(data_dict.keys())}")


def ensure_gray(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def ensure_uint8_image(img: np.ndarray) -> np.ndarray:
    """
    Converts an image into uint8 if needed.

    This is useful when using raw sonar images, which may be uint16 or float,
    while MINIMA/SuperPoint usually expects uint8-like image data.
    """
    img = np.asarray(img)

    if img.dtype == np.uint8:
        return img

    img_float = img.astype(np.float32)

    finite = np.isfinite(img_float)
    if not np.any(finite):
        return np.zeros(img.shape, dtype=np.uint8)

    min_val = float(np.min(img_float[finite]))
    max_val = float(np.max(img_float[finite]))

    if max_val - min_val < 1e-12:
        return np.zeros(img.shape, dtype=np.uint8)

    img_norm = (img_float - min_val) / (max_val - min_val)
    img_norm = np.clip(img_norm, 0.0, 1.0)

    return (255.0 * img_norm).astype(np.uint8)


def resize_mask_to_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Ensures the mask has the same pixel size as the image used for matching.
    """
    mask = ensure_gray(mask)

    img_h, img_w = image.shape[:2]
    mask_h, mask_w = mask.shape[:2]

    if (mask_h, mask_w) != (img_h, img_w):
        mask = cv2.resize(
            mask,
            (img_w, img_h),
            interpolation=cv2.INTER_NEAREST,
        )

    return mask


def get_image_dict_from_source(
    image_source: str,
    inference_obj=None,
    section_obj=None,
) -> dict:
    """
    Selects the image dictionary used for matching.

    image_source:
        'rho_gray' -> use PhysDNet rho_gray output
        'raw'      -> use raw sonar images from section.images
    """
    image_source = image_source.lower().strip()

    if image_source in {"rho_gray", "rho", "physdnet"}:
        if inference_obj is None:
            raise ValueError("inference_obj is required when image_source='rho_gray'")
        return inference_obj.rho_gray

    if image_source in {"raw", "original", "section"}:
        if section_obj is None:
            raise ValueError("section_obj is required when image_source='raw'")
        return section_obj.images

    raise ValueError(
        f"Invalid image_source='{image_source}'. Use 'rho_gray' or 'raw'."
    )


def prepare_side_image_and_mask(
    inference_obj,
    terrain_obj,
    side: str,
    image_source: str = "rho_gray",
    section_obj=None,
    flip_left_image: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns one side image and its corresponding mask.

    image_source:
        'rho_gray' -> use inference_obj.rho_gray
        'raw'      -> use section_obj.images

    The left image is flipped horizontally when flip_left_image=True.

    terrain_obj.final_mask is expected to already be in the same orientation as
    the image used for matching. In the standard pipeline, xtf_utils.combine_masks()
    flips the final left mask once during mask construction.

    The right image and right mask are kept unchanged.
    """
    side = side.lower()

    image_dict = get_image_dict_from_source(
        image_source=image_source,
        inference_obj=inference_obj,
        section_obj=section_obj,
    )

    img_key = get_key_by_side(image_dict, side)
    mask_key = get_key_by_side(terrain_obj.final_mask, side)

    img = image_dict[img_key]
    mask = terrain_obj.final_mask[mask_key]

    if side == "left" and flip_left_image:
        img = np.fliplr(img)

    img = ensure_uint8_image(img)

    img = np.ascontiguousarray(img)
    mask = np.ascontiguousarray(mask)

    mask = resize_mask_to_image(mask, img)

    return img, mask


def build_matching_inputs(
    mode: str,
    inference0,
    inference1,
    terrain0,
    terrain1,
    image_source: str = "rho_gray",
    section0=None,
    section1=None,
    flip_left_image: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    """
    Builds the input images and masks for the selected matching mode.

    image_source:
        'rho_gray' -> use PhysDNet rho_gray images
        'raw'      -> use raw sonar images from section.images

    Returns:
        img0, img1, mask0, mask1, swath_side0, swath_side1

    Supported modes:
        'left-left'
        'right-right'
        'left-right'
        'right-left'
        'both'
    """
    mode = mode.lower().strip()

    if mode == "both":
        left0_img, left0_mask = prepare_side_image_and_mask(
            inference0,
            terrain0,
            "left",
            image_source=image_source,
            section_obj=section0,
            flip_left_image=flip_left_image,
        )
        right0_img, right0_mask = prepare_side_image_and_mask(
            inference0,
            terrain0,
            "right",
            image_source=image_source,
            section_obj=section0,
            flip_left_image=flip_left_image,
        )

        left1_img, left1_mask = prepare_side_image_and_mask(
            inference1,
            terrain1,
            "left",
            image_source=image_source,
            section_obj=section1,
            flip_left_image=flip_left_image,
        )
        right1_img, right1_mask = prepare_side_image_and_mask(
            inference1,
            terrain1,
            "right",
            image_source=image_source,
            section_obj=section1,
            flip_left_image=flip_left_image,
        )

        img0 = np.hstack([left0_img, right0_img])
        img1 = np.hstack([left1_img, right1_img])

        mask0 = np.hstack([left0_mask, right0_mask])
        mask1 = np.hstack([left1_mask, right1_mask])

        img0 = np.ascontiguousarray(img0)
        img1 = np.ascontiguousarray(img1)
        mask0 = np.ascontiguousarray(mask0)
        mask1 = np.ascontiguousarray(mask1)

        return img0, img1, mask0, mask1, "both", "both"

    valid_modes = {"left-left", "right-right", "left-right", "right-left"}

    if mode not in valid_modes:
        raise ValueError(
            f"Invalid MATCH_MODE='{mode}'. Use one of: {sorted(valid_modes | {'both'})}"
        )

    side0, side1 = mode.split("-")

    img0, mask0 = prepare_side_image_and_mask(
        inference0,
        terrain0,
        side0,
        image_source=image_source,
        section_obj=section0,
        flip_left_image=flip_left_image,
    )

    img1, mask1 = prepare_side_image_and_mask(
        inference1,
        terrain1,
        side1,
        image_source=image_source,
        section_obj=section1,
        flip_left_image=flip_left_image,
    )

    return img0, img1, mask0, mask1, side0, side1


def extract_swaths_from_calculate_output(output) -> np.ndarray:
    """
    Accepts either:
      swaths
    or:
      swaths, trajectory, altitude, roll, pitch, yaw
    or any tuple/list whose first element is swaths.
    """
    if isinstance(output, tuple) or isinstance(output, list):
        return output[0]

    return output


def get_swaths_for_side(
    pings: list[pyxtf.XTFPingHeader],
    side: str,
    flip_left_side: bool = True,
) -> np.ndarray:
    """
    Calculates the swath grid and selects the part corresponding to the image.

    side='left'  -> first half of the swath grid
    side='right' -> second half of the swath grid
    side='both'  -> full swath grid

    When flip_left_side=True, the left swath grid is horizontally flipped to
    stay consistent with the left image orientation used for matching. This is
    required so pixels from a flipped left image are converted to the correct
    UTM coordinates.
    """
    swaths = extract_swaths_from_calculate_output(calculate_swath_positions(pings))

    side = side.lower()
    mid = swaths.shape[1] // 2

    left_swaths = swaths[:, :mid, :]
    right_swaths = swaths[:, mid:, :]

    if flip_left_side:
        left_swaths = np.ascontiguousarray(np.flip(left_swaths, axis=1))

    if side == "left":
        return left_swaths

    if side == "right":
        return right_swaths

    if side == "both":
        return np.ascontiguousarray(np.concatenate([left_swaths, right_swaths], axis=1))

    raise ValueError(f"Invalid swath side: {side}")


def point_mask_test(points_xy: np.ndarray, mask: np.ndarray, threshold: int = 0) -> np.ndarray:
    """
    Returns a boolean mask telling whether each point falls inside a valid region.

    points_xy: (N, 2) in pixel coordinates [x, y]
    mask: 2D uint8 mask where white pixels are rejected and black pixels are kept
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
    Keeps matches that survive both masks.
    """
    keep = np.ones(len(mkpts0), dtype=bool)

    if mask0 is not None:
        keep &= point_mask_test(mkpts0, mask0, threshold=threshold0)

    if mask1 is not None:
        keep &= point_mask_test(mkpts1, mask1, threshold=threshold1)

    return mkpts0[keep], mkpts1[keep], mconf[keep], keep


def pixels_to_utm(
    points_xy: np.ndarray,
    swaths: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> np.ndarray:
    """
    Converts image pixel coordinates into UTM coordinates.

    For 512x512 PhysDNet images, coordinates are scaled into the original swath
    grid before bilinear interpolation.

    For raw sonar images, the same mapping is used, but the image size is usually
    closer to the original swath-grid size.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)

    img_h, img_w = image_shape[:2]
    num_pings, num_samples = swaths.shape[:2]

    if img_w <= 1 or img_h <= 1:
        raise ValueError("Invalid image shape for pixel to UTM conversion")

    # Map image coordinates to swath-grid coordinates.
    x = points_xy[:, 0] * (num_samples - 1) / (img_w - 1)
    y = points_xy[:, 1] * (num_pings - 1) / (img_h - 1)

    x = np.clip(x, 0, num_samples - 1)
    y = np.clip(y, 0, num_pings - 1)

    # Bilinear interpolation.
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)

    x1 = np.clip(x0 + 1, 0, num_samples - 1)
    y1 = np.clip(y0 + 1, 0, num_pings - 1)

    wx = (x - x0).reshape(-1, 1)
    wy = (y - y0).reshape(-1, 1)

    p00 = swaths[y0, x0]
    p10 = swaths[y0, x1]
    p01 = swaths[y1, x0]
    p11 = swaths[y1, x1]

    utm = (1 - wx) * (1 - wy) * p00 + \
          wx * (1 - wy) * p10 + \
          (1 - wx) * wy * p01 + \
          wx * wy * p11

    return utm


def extract_euclidean_homography_params(
    H: Optional[np.ndarray],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Extracts tx, ty, and theta from a 3-DOF Euclidean homography.

    Expected structure:
        H = [[cos(theta), -sin(theta), tx],
             [sin(theta),  cos(theta), ty],
             [0,           0,          1 ]]

    Returns:
        tx, ty, theta_rad, theta_deg
    """
    if H is None:
        return None, None, None, None

    H = np.asarray(H, dtype=float)

    if H.shape != (3, 3):
        return None, None, None, None

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    tx = float(H[0, 2])
    ty = float(H[1, 2])

    theta = float(np.arctan2(H[1, 0], H[0, 0]))
    theta_deg = float(np.degrees(theta))

    return tx, ty, theta, theta_deg


def get_ransac_inlier_mask_from_homography(
    H: Optional[np.ndarray],
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    threshold: float,
) -> Optional[np.ndarray]:
    """
    Reconstructs a boolean RANSAC inlier mask using your projection_error() function.
    """
    if H is None or len(src_pts) == 0:
        return None

    errors = projection_error(H, src_pts, dst_pts)
    return errors < threshold


def estimate_homography_ransac_custom(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    ransac_reproj_threshold: float,
    max_iters: int,
    confidence: float,
    model: str = "Euclidean",
    outlier_percent: float = 0.5,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calls your custom RANSAC implementation and reconstructs a boolean inlier mask.
    """
    min_points = {
        "Translation": 1,
        "Euclidean": 2,
        "Similarity": 2,
        "Affine": 3,
        "Projective": 4,
    }

    if model not in min_points:
        raise ValueError(f"Unsupported model: {model}")

    if len(mkpts0) < min_points[model] or len(mkpts1) < min_points[model]:
        return None, None

    ransac_result = compute_homography_ransac(
        mkpts0,
        mkpts1,
        model=model,
        num_iterations=max_iters,
        outlier_percent=outlier_percent,
        p=confidence,
        t=ransac_reproj_threshold,
    )

    if ransac_result is None:
        return None, None

    H, _, _ = ransac_result

    inliers = get_ransac_inlier_mask_from_homography(
        H,
        mkpts0,
        mkpts1,
        threshold=ransac_reproj_threshold,
    )

    if inliers is not None:
        inliers = inliers.reshape(-1).astype(bool)

    return H, inliers


def estimate_utm_homography_ransac(
    mkpts0_utm: np.ndarray,
    mkpts1_utm: np.ndarray,
    ransac_reproj_threshold: float,
    max_iters: int,
    confidence: float,
    outlier_percent: float,
) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Estimates the 3-DOF Euclidean homography in UTM coordinates.

    The estimation is performed in a local UTM frame to avoid large coordinate
    magnitudes. The final H_utm is converted back to the original UTM frame.
    """
    if len(mkpts0_utm) < 2 or len(mkpts1_utm) < 2:
        return None, None, None, None, None, None, None

    utm_offset0 = np.mean(mkpts0_utm, axis=0)
    utm_offset1 = np.mean(mkpts1_utm, axis=0)

    mkpts0_utm_local = mkpts0_utm - utm_offset0
    mkpts1_utm_local = mkpts1_utm - utm_offset1

    H_utm_local, ransac_inliers_utm = estimate_homography_ransac_custom(
        mkpts0_utm_local,
        mkpts1_utm_local,
        ransac_reproj_threshold=ransac_reproj_threshold,
        max_iters=max_iters,
        confidence=confidence,
        model="Euclidean",
        outlier_percent=outlier_percent,
    )

    if H_utm_local is None:
        return None, None, mkpts0_utm_local, mkpts1_utm_local, utm_offset0, utm_offset1, ransac_inliers_utm

    T0 = np.array([
        [1.0, 0.0, -utm_offset0[0]],
        [0.0, 1.0, -utm_offset0[1]],
        [0.0, 0.0, 1.0],
    ])

    T1_inv = np.array([
        [1.0, 0.0, utm_offset1[0]],
        [0.0, 1.0, utm_offset1[1]],
        [0.0, 0.0, 1.0],
    ])

    H_utm = T1_inv @ H_utm_local @ T0

    if abs(H_utm[2, 2]) > 1e-12:
        H_utm = H_utm / H_utm[2, 2]

    return H_utm, H_utm_local, mkpts0_utm_local, mkpts1_utm_local, utm_offset0, utm_offset1, ransac_inliers_utm


def run_parametrized_minima_registration(
    matcher: MinimaMatcher,
    mode: str,
    inference0,
    inference1,
    terrain0,
    terrain1,
    pings0: list[pyxtf.XTFPingHeader],
    pings1: list[pyxtf.XTFPingHeader],
    image_source: str = "rho_gray",
    section0=None,
    section1=None,
    apply_mask: bool = True,
    ransac_reproj_threshold_pixel: float = 10.0,
    ransac_reproj_threshold_utm: float = 1.0,
    ransac_max_iters: int = 10_000,
    ransac_confidence: float = 0.995,
    ransac_outlier_percent: float = 0.5,
) -> RegistrationResult:
    """
    Runs the complete MINIMA + optional mask filtering + pixel/UTM RANSAC pipeline once.
    """
    if pings0 is None or pings1 is None:
        raise ValueError("pings0 and pings1 must always be passed for UTM conversion")

    image_source = image_source.lower().strip()

    img0, img1, mask0, mask1, swath_side0, swath_side1 = build_matching_inputs(
        mode,
        inference0,
        inference1,
        terrain0,
        terrain1,
        image_source=image_source,
        section0=section0,
        section1=section1,
        flip_left_image=True,
    )

    swaths0 = get_swaths_for_side(pings0, swath_side0)
    swaths1 = get_swaths_for_side(pings1, swath_side1)

    print(f"Matching mode: {mode}")
    print(f"Image source: {image_source}")
    print(f"Apply mask: {apply_mask}")
    print(f"Image 0 shape: {img0.shape} | swath side: {swath_side0} | swath grid: {swaths0.shape}")
    print(f"Image 1 shape: {img1.shape} | swath side: {swath_side1} | swath grid: {swaths1.shape}")
    print(f"Mask 0 shape:  {mask0.shape}")
    print(f"Mask 1 shape:  {mask1.shape}")

    mkpts0, mkpts1, mconf = matcher.match_images(img0, img1)

    if apply_mask:
        mkpts0_kept, mkpts1_kept, mconf_kept, keep_mask = filter_matches_with_masks(
            mkpts0,
            mkpts1,
            mconf,
            mask0=mask0,
            mask1=mask1,
        )
    else:
        keep_mask = np.ones(len(mkpts0), dtype=bool)
        mkpts0_kept = mkpts0
        mkpts1_kept = mkpts1
        mconf_kept = mconf

    H_pixel, ransac_inliers_pixel = estimate_homography_ransac_custom(
        mkpts0_kept,
        mkpts1_kept,
        ransac_reproj_threshold=ransac_reproj_threshold_pixel,
        max_iters=ransac_max_iters,
        confidence=ransac_confidence,
        model="Euclidean",
        outlier_percent=ransac_outlier_percent,
    )

    mkpts0_utm = pixels_to_utm(mkpts0_kept, swaths0, img0.shape)
    mkpts1_utm = pixels_to_utm(mkpts1_kept, swaths1, img1.shape)

    H_utm, H_utm_local, mkpts0_utm_local, mkpts1_utm_local, utm_offset0, utm_offset1, ransac_inliers_utm = estimate_utm_homography_ransac(
        mkpts0_utm,
        mkpts1_utm,
        ransac_reproj_threshold=ransac_reproj_threshold_utm,
        max_iters=ransac_max_iters,
        confidence=ransac_confidence,
        outlier_percent=ransac_outlier_percent,
    )

    if H_utm is not None:
        H = H_utm
        ransac_inliers = ransac_inliers_utm
    else:
        H = H_pixel
        ransac_inliers = ransac_inliers_pixel

    tx_utm, ty_utm, theta_utm, theta_utm_deg = extract_euclidean_homography_params(H_utm)

    tx_utm_local, ty_utm_local, theta_utm_local, theta_utm_local_deg = extract_euclidean_homography_params(H_utm_local)

    num_ransac_inliers = 0 if ransac_inliers is None else int(np.sum(ransac_inliers))

    return RegistrationResult(
        mode=mode,
        image_source=image_source,

        img0=img0,
        img1=img1,
        mask0=mask0,
        mask1=mask1,

        mkpts0=mkpts0,
        mkpts1=mkpts1,
        mconf=mconf,
        keep_mask=keep_mask,

        mkpts0_kept=mkpts0_kept,
        mkpts1_kept=mkpts1_kept,
        mconf_kept=mconf_kept,

        mkpts0_utm=mkpts0_utm,
        mkpts1_utm=mkpts1_utm,
        mkpts0_utm_local=mkpts0_utm_local,
        mkpts1_utm_local=mkpts1_utm_local,

        H=H,
        H_pixel=H_pixel,
        H_utm=H_utm,
        H_utm_local=H_utm_local,

        tx_utm=tx_utm,
        ty_utm=ty_utm,
        theta_utm=theta_utm,
        theta_utm_deg=theta_utm_deg,

        tx_utm_local=tx_utm_local,
        ty_utm_local=ty_utm_local,
        theta_utm_local=theta_utm_local,
        theta_utm_local_deg=theta_utm_local_deg,

        ransac_inliers=ransac_inliers,
        ransac_inliers_pixel=ransac_inliers_pixel,
        ransac_inliers_utm=ransac_inliers_utm,

        utm_offset0=utm_offset0,
        utm_offset1=utm_offset1,

        num_matches=len(mkpts0),
        num_matches_after_mask=len(mkpts0_kept),
        num_ransac_inliers=num_ransac_inliers,
    )



def build_debug_self_rotation_input(
    mode: str,
    inference_obj,
    terrain_obj,
    image_source: str = "rho_gray",
    section_obj=None,
    flip_left_image: bool = True,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Builds a single image and mask for the synthetic 180-degree self-rotation test.

    The selected image is the same as the image that would be used as image 0 in
    run_parametrized_minima_registration().

    For example:
        mode='both'       -> use [flipped left | right]
        mode='left-left'  -> use left side
        mode='left-right' -> use left side
        mode='right-left' -> use right side
        mode='right-right'-> use right side

    Returns:
        img, mask, swath_side
    """
    mode = mode.lower().strip()

    if mode == "both":
        left_img, left_mask = prepare_side_image_and_mask(
            inference_obj,
            terrain_obj,
            "left",
            image_source=image_source,
            section_obj=section_obj,
            flip_left_image=flip_left_image,
        )
        right_img, right_mask = prepare_side_image_and_mask(
            inference_obj,
            terrain_obj,
            "right",
            image_source=image_source,
            section_obj=section_obj,
            flip_left_image=flip_left_image,
        )

        img = np.hstack([left_img, right_img])
        mask = np.hstack([left_mask, right_mask])

        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)

        return img, mask, "both"

    valid_modes = {"left-left", "right-right", "left-right", "right-left"}

    if mode not in valid_modes:
        raise ValueError(
            f"Invalid MATCH_MODE='{mode}'. Use one of: {sorted(valid_modes | {'both'})}"
        )

    side0, _ = mode.split("-")

    img, mask = prepare_side_image_and_mask(
        inference_obj,
        terrain_obj,
        side0,
        image_source=image_source,
        section_obj=section_obj,
        flip_left_image=flip_left_image,
    )

    return img, mask, side0


def run_debug_self_rotation_registration(
    matcher: MinimaMatcher,
    mode: str,
    inference_obj,
    terrain_obj,
    pings: list[pyxtf.XTFPingHeader],
    image_source: str = "rho_gray",
    section_obj=None,
    apply_mask: bool = True,
    ransac_reproj_threshold_pixel: float = 10.0,
    ransac_reproj_threshold_utm: float = 1.0,
    ransac_max_iters: int = 10_000,
    ransac_confidence: float = 0.995,
    ransac_outlier_percent: float = 0.5,
) -> RegistrationResult:
    """
    Debug registration test.

    This function builds one image and its mask, rotates both by 180 degrees,
    then matches the original image against the rotated version.

    RANSAC is not told that this is a synthetic self-match.

    Pixel-domain expectation:
        H_pixel should be approximately a 180-degree rotation:
            [[-1,  0, width - 1],
             [ 0, -1, height - 1],
             [ 0,  0, 1]]

    UTM-domain expectation:
        The swath grid of the rotated image is also rotated by 180 degrees.
        Therefore, correct matches should map approximately to the same physical
        UTM points, and H_utm should be close to identity.
    """
    if pings is None:
        raise ValueError("pings must be passed for UTM conversion")

    image_source = image_source.lower().strip()

    img0, mask0, swath_side = build_debug_self_rotation_input(
        mode=mode,
        inference_obj=inference_obj,
        terrain_obj=terrain_obj,
        image_source=image_source,
        section_obj=section_obj,
        flip_left_image=True,
    )

    img1 = np.ascontiguousarray(np.rot90(img0, 2))
    mask1 = np.ascontiguousarray(np.rot90(mask0, 2))

    swaths0 = get_swaths_for_side(pings, swath_side)
    swaths1 = np.ascontiguousarray(np.rot90(swaths0, 2, axes=(0, 1)))

    print(f"Debug mode: self-rotation 180 degrees")
    print(f"Original matching mode used to select image: {mode}")
    print(f"Image source: {image_source}")
    print(f"Apply mask: {apply_mask}")
    print(f"Image 0 shape: {img0.shape} | swath side: {swath_side} | swath grid: {swaths0.shape}")
    print(f"Image 1 shape: {img1.shape} | rotated swath grid: {swaths1.shape}")
    print(f"Mask 0 shape:  {mask0.shape}")
    print(f"Mask 1 shape:  {mask1.shape}")

    mkpts0, mkpts1, mconf = matcher.match_images(img0, img1)

    if apply_mask:
        mkpts0_kept, mkpts1_kept, mconf_kept, keep_mask = filter_matches_with_masks(
            mkpts0,
            mkpts1,
            mconf,
            mask0=mask0,
            mask1=mask1,
        )
    else:
        keep_mask = np.ones(len(mkpts0), dtype=bool)
        mkpts0_kept = mkpts0
        mkpts1_kept = mkpts1
        mconf_kept = mconf

    H_pixel, ransac_inliers_pixel = estimate_homography_ransac_custom(
        mkpts0_kept,
        mkpts1_kept,
        ransac_reproj_threshold=ransac_reproj_threshold_pixel,
        max_iters=ransac_max_iters,
        confidence=ransac_confidence,
        model="Euclidean",
        outlier_percent=ransac_outlier_percent,
    )

    mkpts0_utm = pixels_to_utm(mkpts0_kept, swaths0, img0.shape)
    mkpts1_utm = pixels_to_utm(mkpts1_kept, swaths1, img1.shape)

    H_utm, H_utm_local, mkpts0_utm_local, mkpts1_utm_local, utm_offset0, utm_offset1, ransac_inliers_utm = estimate_utm_homography_ransac(
        mkpts0_utm,
        mkpts1_utm,
        ransac_reproj_threshold=ransac_reproj_threshold_utm,
        max_iters=ransac_max_iters,
        confidence=ransac_confidence,
        outlier_percent=ransac_outlier_percent,
    )

    if H_utm is not None:
        H = H_utm
        ransac_inliers = ransac_inliers_utm
    else:
        H = H_pixel
        ransac_inliers = ransac_inliers_pixel

    tx_utm, ty_utm, theta_utm, theta_utm_deg = extract_euclidean_homography_params(H_utm)

    tx_utm_local, ty_utm_local, theta_utm_local, theta_utm_local_deg = extract_euclidean_homography_params(H_utm_local)

    num_ransac_inliers = 0 if ransac_inliers is None else int(np.sum(ransac_inliers))

    return RegistrationResult(
        mode=f"debug-rotate180-{mode}",
        image_source=image_source,

        img0=img0,
        img1=img1,
        mask0=mask0,
        mask1=mask1,

        mkpts0=mkpts0,
        mkpts1=mkpts1,
        mconf=mconf,
        keep_mask=keep_mask,

        mkpts0_kept=mkpts0_kept,
        mkpts1_kept=mkpts1_kept,
        mconf_kept=mconf_kept,

        mkpts0_utm=mkpts0_utm,
        mkpts1_utm=mkpts1_utm,
        mkpts0_utm_local=mkpts0_utm_local,
        mkpts1_utm_local=mkpts1_utm_local,

        H=H,
        H_pixel=H_pixel,
        H_utm=H_utm,
        H_utm_local=H_utm_local,

        tx_utm=tx_utm,
        ty_utm=ty_utm,
        theta_utm=theta_utm,
        theta_utm_deg=theta_utm_deg,

        tx_utm_local=tx_utm_local,
        ty_utm_local=ty_utm_local,
        theta_utm_local=theta_utm_local,
        theta_utm_local_deg=theta_utm_local_deg,

        ransac_inliers=ransac_inliers,
        ransac_inliers_pixel=ransac_inliers_pixel,
        ransac_inliers_utm=ransac_inliers_utm,

        utm_offset0=utm_offset0,
        utm_offset1=utm_offset1,

        num_matches=len(mkpts0),
        num_matches_after_mask=len(mkpts0_kept),
        num_ransac_inliers=num_ransac_inliers,
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
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return vis


def warp_and_overlay(
    img0: np.ndarray,
    img1: np.ndarray,
    H_pixel: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Warps img0 into img1's pixel frame and overlays them.

    Use H_pixel here, not H_utm.
    """
    if H_pixel is None:
        raise ValueError("H_pixel is None. Cannot create overlay.")

    h, w = img1.shape[:2]

    warped = cv2.warpPerspective(img0, H_pixel, (w, h))

    if img1.ndim == 2:
        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_bgr = img1.copy()

    if warped.ndim == 2:
        warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    else:
        warped_bgr = warped.copy()

    overlay = cv2.addWeighted(img1_bgr, 1.0 - alpha, warped_bgr, alpha, 0.0)

    return overlay


def show_registration_result(result: RegistrationResult, max_draw: int = 300):
    """
    Displays inputs, masks, matches, and pixel-domain overlay.
    """
    print("Mode:                    ", result.mode)
    print("Image source:            ", result.image_source)
    print("Raw matches:             ", result.num_matches)
    print("After mask filtering:    ", result.num_matches_after_mask)
    print("RANSAC inliers:          ", result.num_ransac_inliers)

    print("\nPixel homography H_pixel:")
    print(result.H_pixel)

    print("\nUTM homography H_utm:")
    print(result.H_utm)

    print("\nLocal UTM homography H_utm_local:")
    print(result.H_utm_local)

    print("\nUTM homography parameters:")
    print(f"tx_utm:        {result.tx_utm}")
    print(f"ty_utm:        {result.ty_utm}")
    print(f"theta_utm:     {result.theta_utm} rad")
    print(f"theta_utm_deg: {result.theta_utm_deg} deg")

    print("\nLocal UTM homography parameters:")
    print(f"tx_utm_local:        {result.tx_utm_local}")
    print(f"ty_utm_local:        {result.ty_utm_local}")
    print(f"theta_utm_local:     {result.theta_utm_local} rad")
    print(f"theta_utm_local_deg: {result.theta_utm_local_deg} deg")

    print("\nUTM offset 0:", result.utm_offset0)
    print("UTM offset 1:", result.utm_offset1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(result.img0, cmap="gray")
    axes[0, 0].set_title("Image 0")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(result.img1, cmap="gray")
    axes[0, 1].set_title("Image 1")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(result.mask0, cmap="gray")
    axes[1, 0].set_title("Mask 0: black=keep, white=reject")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(result.mask1, cmap="gray")
    axes[1, 1].set_title("Mask 1: black=keep, white=reject")
    axes[1, 1].axis("off")

    plt.suptitle(f"Inputs for MATCH_MODE='{result.mode}', IMAGE_SOURCE='{result.image_source}'")
    plt.tight_layout()
    plt.show()

    vis = draw_matches(
        result.img0,
        result.img1,
        result.mkpts0_kept,
        result.mkpts1_kept,
        result.ransac_inliers,
        max_draw=max_draw,
    )

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 8))
    plt.imshow(vis_rgb)
    plt.axis("off")
    plt.title("Matches after mask filtering and 3-DOF RANSAC")
    plt.show()

    if result.H_pixel is not None:
        overlay = warp_and_overlay(result.img0, result.img1, result.H_pixel, alpha=0.5)

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Overlay using H_pixel")
        plt.show()
    else:
        print("Pixel homography was not estimated, so overlay was skipped.")