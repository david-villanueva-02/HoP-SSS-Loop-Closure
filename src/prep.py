import pyxtf
import numpy as np
import os

try:
    from xtf_utils import calculate_blind_zone_indices
except ImportError as e:
    from xtf_utils import calculate_blind_zone_indices

from dataclasses import dataclass, field
import numpy as np
import cv2
import pyxtf

def generate_shadow(img):
    k = 1
    if img is None:
        print("Unable to read image")
    mean_val = np.mean(img)
    std_val = np.std(img)
    shadow_mask = (img < mean_val - k * std_val).astype(np.uint8) * 255

    return shadow_mask

@dataclass
class SonarSectionData:
    """Structured output from prepare_data_range for a single XTF section."""
    # Core sonar images (uint8 NumPy arrays, grayscale)
    images:     dict[str, np.ndarray] = field(default_factory=dict)
    
    # Shadow/mask images (uint8 NumPy arrays, grayscale)
    shadows:    dict[str, np.ndarray] = field(default_factory=dict)

    # Altitude arrays (float NumPy arrays, one value per ping row)
    altitudes:  dict[str, np.ndarray] = field(default_factory=dict)

    # Slant-range arrays (float NumPy arrays, one value per ping row)
    ranges:     dict[str, np.ndarray] = field(default_factory=dict)

    # Blind-zone masks (uint8 NumPy arrays)
    blind:      dict[str, np.ndarray] = field(default_factory=dict)

    # Combined left+right image (uint8 NumPy array, or None if side != "both")
    combined:   np.ndarray | None = None

    # Raw sonar packets for downstream use
    sonar_packets: list = field(default_factory=list)


def prepare_data_range(
    input_dir: str,
    xtf_file: str,
    segment_size: int,          # kept for blind-zone mask shape — not used for windowing
    upper_limit: int,
    side: str,
    data_lower_limit: int,
    data_upper_limit: int,
) -> SonarSectionData:
    """
    Loads one XTF section (data_lower_limit : data_upper_limit) and returns
    all generated images/arrays as a SonarSectionData object.

    Parameters
    ----------
    input_dir         : directory containing the XTF file
    xtf_file          : filename of the XTF file
    segment_size      : used only for blind-zone mask shape (rows dimension)
    upper_limit       : clip + log-normalisation ceiling
    side              : "left", "right", or "both"
    data_lower_limit  : first ping index to include
    data_upper_limit  : last ping index (exclusive)

    Returns
    -------
    SonarSectionData  : all images, shadows, altitudes, ranges, masks, combined
    """
    result = SonarSectionData()

    file_path = os.path.join(input_dir, xtf_file)
    base = xtf_file  # used as key prefix — mirrors the original filename stem

    # ------------------------------------------------------------------ #
    #  1. Read XTF and slice the requested ping range                      #
    # ------------------------------------------------------------------ #
    print(f"Processing file: {xtf_file}  [{data_lower_limit}:{data_upper_limit}]")
    try:
        _file_header, packets = pyxtf.xtf_read(file_path)
        sonar_packets: list[pyxtf.XTFPingHeader] = packets[pyxtf.XTFHeaderType.sonar]
        sonar_packets = sonar_packets[data_lower_limit:data_upper_limit]
        result.sonar_packets = sonar_packets
    except Exception as e:
        print(f"Error reading {xtf_file}: {e}")
        return result

    # ------------------------------------------------------------------ #
    #  2. Extract per-ping data                                            #
    # ------------------------------------------------------------------ #
    left_channel_data  = []
    right_channel_data = []
    altitude_list      = []
    range_left_list    = []
    range_right_list   = []

    left_idx, right_idx = calculate_blind_zone_indices(sonar_packets)

    mask_left_inv  = None
    mask_right_inv = None

    for packet in sonar_packets:
        altitude_list.append(packet.SensorPrimaryAltitude)
        range_left_list.append(packet.ping_chan_headers[0].SlantRange)
        range_right_list.append(packet.ping_chan_headers[1].SlantRange)

        ping_data = getattr(packet, "data", None)
        if isinstance(ping_data, list) and len(ping_data) >= 2:

            # Build blind-zone masks once (shape depends on channel width)
            if mask_left_inv is None:
                n_cols = ping_data[0].shape[0]
                mask_right = np.zeros((segment_size, n_cols), dtype=np.uint8)
                mask_right[:, left_idx] = 255          # blind zone columns
                mask_left  = np.flip(mask_right, axis=1)

                # mask_left_inv  = np.where(mask_left  == 0, 255, 0).astype(np.uint8)
                # mask_right_inv = np.where(mask_right == 0, 255, 0).astype(np.uint8)
                mask_right_inv  = np.where(mask_left  == 0, 255, 0).astype(np.uint8)
                mask_left_inv = np.where(mask_right == 0, 255, 0).astype(np.uint8)
                print(f"Blind-zone masks built — shape: {mask_left_inv.shape}")

            left_channel_data.append(ping_data[0])
            right_channel_data.append(ping_data[1])

    # ------------------------------------------------------------------ #
    #  3. Validate                                                         #
    # ------------------------------------------------------------------ #
    left_channel_data  = np.array(left_channel_data)
    right_channel_data = np.array(right_channel_data)
    altitude_arr       = np.array(altitude_list)
    range_left_arr     = np.array(range_left_list)
    range_right_arr    = np.array(range_right_list)

    print(f"Left channel shape: {left_channel_data.shape}")

    if left_channel_data.size == 0 or right_channel_data.size == 0:
        print(f"Warning: {xtf_file} has no valid channel data in the requested range.")
        return result

    # ------------------------------------------------------------------ #
    #  4. Log-scale + normalise to uint8                                   #
    # ------------------------------------------------------------------ #
    vmax = np.log10(upper_limit)

    def _to_uint8(data: np.ndarray) -> np.ndarray:
        clipped   = np.clip(data, 0, upper_limit - 1)
        log_data  = np.log10(clipped + 1e-4)
        image     = np.vstack(log_data)
        return ((image / vmax) * 255).astype(np.uint8)

    left_norm  = _to_uint8(left_channel_data)
    right_norm = _to_uint8(right_channel_data)

    # ------------------------------------------------------------------ #
    #  5. Build outputs for requested side(s)                              #
    # ------------------------------------------------------------------ #
    process_left  = side in ("left",  "both")
    process_right = side in ("right", "both")

    if process_left:
        # flipped_left = np.fliplr(left_norm)
        left_shadow  = generate_shadow(left_norm)

        result.images   [f"{base}_left"]    = left_norm
        result.shadows  [f"{base}_left"]    = left_shadow
        result.altitudes[f"{base}_left"]    = altitude_arr
        result.ranges   [f"{base}_left"]    = range_left_arr

    if process_right:
        right_shadow = generate_shadow(right_norm)

        result.images   [f"{base}_right"]   = right_norm
        result.shadows  [f"{base}_right"]   = right_shadow
        result.altitudes[f"{base}_right"]   = altitude_arr
        result.ranges   [f"{base}_right"]   = range_right_arr

    # Combined image (only meaningful when both sides processed)
    if process_left and process_right:
        result.combined = np.hstack((left_norm, right_norm))

    # Blind-zone masks
    if mask_left_inv is not None:
        if process_left:
            result.blind[f"{base}_mask_left"]  = mask_left_inv
        if process_right:
            result.blind[f"{base}_mask_right"] = mask_right_inv

    n_rows = left_norm.shape[0] if process_left else right_norm.shape[0]
    print(f"{xtf_file}: section rows={n_rows}, side={side}")

    return result

def save_sonar_section(data: SonarSectionData, output_dir: str, xtf_file: str = "") -> None:
    """
    Saves all images/arrays from a SonarSectionData object to output_dir.
    Call this only when you explicitly want files on disk.
    """
    # Folder for images
    os.makedirs(output_dir, exist_ok=True)

    # Folder for NPY files
    os.makedirs(output_dir + r"/npy", exist_ok=True)

    # Folder for masks images
    os.makedirs(output_dir + r"/masks", exist_ok=True)

    for key, img in data.images.items():
        cv2.imwrite(os.path.join(output_dir, f"{key}.png"), img)

    for key, img in data.shadows.items():
        cv2.imwrite(os.path.join(output_dir + r"/masks", f"{key}_shadow.png"), img)

    for key, img in data.blind.items():
        cv2.imwrite(os.path.join(output_dir + r"/masks", f"{key}.png"), img)

    for key, arr in data.altitudes.items():
        np.save(os.path.join(output_dir + r"/npy", f"{key}_altitude.npy"), arr)

    for key, arr in data.ranges.items():
        np.save(os.path.join(output_dir + r"/npy", f"{key}_range.npy"), arr)

    if data.combined is not None:
        fname = f"{xtf_file}_combined.png" if xtf_file else "combined.png"
        cv2.imwrite(os.path.join(output_dir, fname), data.combined)

    print(f"Saved section data to: {output_dir}")