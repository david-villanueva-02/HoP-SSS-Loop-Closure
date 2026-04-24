import numpy as np
import cv2
from dataclasses import dataclass, field
from tqdm import tqdm
from inference import InferenceResult
from prep import SonarSectionData
import os

@dataclass
class TerrainResult:
    """
    Outputs from terrain_mask for a single sonar section.

    Keys match the source image key (e.g. 'survey_001.xtf_left').
    All arrays are uint8 grayscale (H, W).
    """
    height_visual: dict[str, np.ndarray] = field(default_factory=dict)
    # CLAHE-enhanced, contrast-inverted height map masked to valid seafloor area
    
    z_mask:        dict[str, np.ndarray] = field(default_factory=dict)
    # Binary shadow mask derived from height_visual via mean/std thresholding

    final_mask:        dict[str, np.ndarray] = field(default_factory=dict)
    # Binary final masks that will be applied for outlier rejection


def terrain_mask(
    section:   "SonarSectionData",
    inference: "InferenceResult",
    k: float = 1.0,
) -> TerrainResult:
    """
    Computes terrain height visuals and shadow masks from predicted depth maps.

    Processes all keys found in inference.z_gray — side selection is already
    encoded in the keys by prepare_data_range upstream.

    Parameters
    ----------
    section   : SonarSectionData from prepare_data_range
                (provides altitude and range arrays)
    inference : InferenceResult from run_inference
                (provides z_gray predicted height maps)
    k         : std-deviation multiplier for shadow thresholding (default 1.0)

    Returns
    -------
    TerrainResult with height_visual and z_mask dicts keyed by image name
    """
    result   = TerrainResult()
    _1D_SIZE = (1, 512)

    for key, z_gray in tqdm(inference.z_gray.items(), desc="terrain_mask"):

        # ── Resolve matching altitude and range ──────────────────────────────
        if key not in section.altitudes or key not in section.ranges:
            print(f"Warning: no matching altitude/range for '{key}', skipping.")
            continue

        altitude    = section.altitudes[key].astype(np.float32)
        range_array = section.ranges   [key].astype(np.float32)

        # ── Resample altitude and range to 512 ──────────────────────────────
        altitude = cv2.resize(
            altitude.reshape(-1, 1), _1D_SIZE, interpolation=cv2.INTER_LINEAR
        ).flatten()
        range_array = cv2.resize(
            range_array.reshape(-1, 1), _1D_SIZE, interpolation=cv2.INTER_LINEAR
        ).flatten()

        # ── Build terrain mask from altitude/range geometry ─────────────────
        H, W = z_gray.shape
        height_mask = np.zeros((H, W), dtype=np.uint8)

        for i, (val_range, val_altitude) in enumerate(zip(range_array, altitude)):
            if val_range == 0:
                continue
            sub_index = int(val_altitude * W / val_range)
            sub_index = np.clip(sub_index, 0, W - 1)
            height_mask[i, sub_index + 1:] = 1

        # ── Mask height map to valid seafloor area ───────────────────────────
        gradient_masked = z_gray * height_mask

        # ── Filter top 20% gradient values within mask ───────────────────────
        masked_vals = gradient_masked[height_mask == 1]
        if masked_vals.size == 0:
            print(f"Warning: empty mask for '{key}', skipping.")
            continue

        threshold = np.percentile(masked_vals, 80)
        gradient_filtered = gradient_masked.copy()
        gradient_filtered[gradient_masked > threshold] = 0

        # ── CLAHE on contrast-inverted masked image ──────────────────────────
        clahe         = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        height_visual = clahe.apply((255 - gradient_masked).astype(np.uint8))

        # ── Shadow mask via mean/std thresholding ────────────────────────────
        mean_val = np.mean(height_visual)
        std_val  = np.std(height_visual)
        z_mask   = (height_visual < mean_val - k * std_val).astype(np.uint8) * 255

        result.height_visual[key] = height_visual
        result.z_mask       [key] = z_mask

    return result


# ──────────────────────────────────────────────────────────────────────────── #
#  Saving utility                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def save_terrain_result(result: TerrainResult, output_dir: str) -> None:
    """
    Saves all arrays in a TerrainResult as flat PNGs into a single output_dir.
    Filename format: {key}_{category}.png
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + r"/masks", exist_ok=True)

    for key, arr in result.height_visual.items():
        cv2.imwrite(os.path.join(output_dir, f"{key}_height_visual.png"), arr)

    for key, arr in result.z_mask.items():
        cv2.imwrite(os.path.join(output_dir + r"/masks", f"{key}_z_mask.png"), arr)

    print(f"Terrain results saved to: {output_dir}")