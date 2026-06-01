import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass, field
from train import PhysDNet
from prep import SonarSectionData
import os

# ──────────────────────────────────────────────────────────────────────────── #
#  Output dataclass                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class InferenceResult:
    """
    All per-image outputs from one inference pass.
    Keys in each dict match the source image name (without extension).

    Arrays are uint8 grayscale (H, W) unless noted.
    'theta', 'z', 'path', 'rho' colourmap images are uint8 RGB (H, W, 3).
    """
    z:        dict[str, np.ndarray] = field(default_factory=dict)  # height – colourmap RGB
    z_gray:   dict[str, np.ndarray] = field(default_factory=dict)  # height – grayscale
    rho:      dict[str, np.ndarray] = field(default_factory=dict)  # reflectivity – colourmap RGB
    rho_gray: dict[str, np.ndarray] = field(default_factory=dict)  # reflectivity – grayscale
    path:     dict[str, np.ndarray] = field(default_factory=dict)  # path loss – colourmap RGB
    theta:    dict[str, np.ndarray] = field(default_factory=dict)  # incidence factor – colourmap RGB
    shadow:   dict[str, np.ndarray] = field(default_factory=dict)  # shadow mask – binary uint8


# ──────────────────────────────────────────────────────────────────────────── #
#  In-memory dataset  (replaces TestSonarImageDataset)                         #
# ──────────────────────────────────────────────────────────────────────────── #

class InMemorySonarDataset(Dataset):
    """
    Wraps the arrays that come directly out of SonarSectionData so the
    DataLoader pipeline is unchanged — no disk I/O required.

    Parameters
    ----------
    image    : (H, W) uint8 grayscale array  — section.images[key]
    altitude : (H,)   float array            — section.altitudes[key]
    range_   : (H,)   float array            — section.ranges[key]
    name     : label string used as the dict key in InferenceResult
    """

    _TARGET_SIZE = 1024
    _RESIZE = transforms.Resize((_TARGET_SIZE, _TARGET_SIZE))
    _1D_SIZE = (1, _TARGET_SIZE)

    def __init__(
        self,
        image:    np.ndarray,
        altitude: np.ndarray,
        range_:   np.ndarray,
        name:     str,
    ):
        # A section is one contiguous block — we treat every row-pair as one
        # "sample" only if the model expects batches.  For a single section we
        # expose it as a single item so the DataLoader loop still works.
        self._image    = image
        self._altitude = altitude
        self._range    = range_
        self._name     = name

    def __len__(self) -> int:
        return 1   # one section = one sample

    def __getitem__(self, idx: int):
        # ── Image: float32 in [0, 1], shape (1, H, W) ──────────────────────
        img_f32 = self._image.astype(np.float32) / 255.0
        tensor  = torch.tensor(img_f32).unsqueeze(0)   # (1, H, W)
        tensor  = self._RESIZE(tensor)                 # (1, 1024, 1024)

        # ── Range: resample to length 1024 ─────────────────────────────────
        rng = self._range.astype(np.float32)
        rng_resized = cv2.resize(
            rng.reshape(-1, 1), self._1D_SIZE, interpolation=cv2.INTER_LINEAR
        ).flatten()

        # ── Altitude: resample to length 1024 ──────────────────────────────
        alt = self._altitude.astype(np.float32)
        alt_resized = cv2.resize(
            alt.reshape(-1, 1), self._1D_SIZE, interpolation=cv2.INTER_LINEAR
        ).flatten()

        return tensor, rng_resized, alt_resized, self._name


# ──────────────────────────────────────────────────────────────────────────── #
#  Pure helper: tensor → uint8 colourmap image (no saving)                     #
# ──────────────────────────────────────────────────────────────────────────── #

def compute_real_dis(A: torch.Tensor, M_list: torch.Tensor) -> torch.Tensor:
    M, N = A.shape

    if M_list.shape[0] != M:
        raise ValueError("The length of M_list must be equal to the number of matrix rows M")

    device = A.device
    col_indices = torch.arange(N, device=device, dtype=A.dtype)

    D = (M_list[:, None] * col_indices[None, :]) / float(N)

    return D


def compute_real_dis_horizontal(A: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    if A.ndim != 2 or H.ndim != 1:
        raise ValueError("A must be two-dimensional and H must be one-dimensional")
    if A.shape[0] != H.shape[0]:
        raise ValueError("The length of H must match the number of rows in A")

    H = H.to(device=A.device, dtype=A.dtype)
    H_squared = H.unsqueeze(1) ** 2  # shape = (M, 1)
    # D[i, j] = sqrt( A[i, j]^2 - H[i]^2 )
    D = torch.sqrt((A ** 2 - H_squared).clamp(min=0))  # 防止负值

    return D


def compute_theta_new(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if not (x.shape == z.shape):
        raise ValueError("A, B, C must have the same shape")
    M, N = x.shape
    D = torch.full((M, N), 0, dtype=z.dtype, device=x.device)

    x_inner = x[1:, 1:]
    z_inner = z[1:, 1:]

    z_up = z[:-1, 1:]
    z_left = z[1:, :-1]

    x_left = x[1:, :-1]

    numerator = x_inner * z_left - x_left * z_inner
    denom = (torch.sqrt((x_left - x_inner) ** 2 + (z_left - z_inner) ** 2 + (z_up - z_inner) ** 2 + 1e-8) *
             torch.sqrt(x_inner ** 2 + z_inner ** 2))

    # Safe calculation, avoid division by 0
    D_inner = torch.where(x_inner == 0, torch.full_like(numerator, -1), numerator / denom)

    D[1:, 1:] = D_inner
    return D


def _tensor_to_colormap(
    tensor: torch.Tensor,
    cmap: str = "seismic",
    symmetric: bool = False,
) -> np.ndarray:
    """
    Convert a 2-D tensor to a uint8 RGB array using a Matplotlib colourmap.
    Replaces visualize_matrix_auto — returns array instead of saving.
    """
    arr = tensor.detach().cpu().numpy()
    arr = np.nan_to_num(arr)

    mn, mx = arr.min(), arr.max()
    if symmetric:
        abs_max = max(abs(mn), abs(mx))
        mn, mx  = -abs_max, abs_max

    norm = (arr - mn) / (mx - mn + 1e-8)           # [0, 1]
    rgba = (cm.get_cmap(cmap)(norm) * 255).astype(np.uint8)   # (H, W, 4)
    return rgba[:, :, :3]                           # drop alpha → (H, W, 3)


def _tensor_to_gray(tensor: torch.Tensor) -> np.ndarray:
    """Normalise a 2-D tensor to a uint8 grayscale array."""
    arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
    arr = np.nan_to_num(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    return arr.astype(np.uint8)


def _tensor_to_binary(tensor: torch.Tensor) -> np.ndarray:
    """Map non-zero values → 255, zero → 0 (shadow mask)."""
    arr = tensor.detach().cpu().numpy()
    arr = np.nan_to_num(arr)
    return ((arr > 0).astype(np.uint8) * 255)


# ──────────────────────────────────────────────────────────────────────────── #
#  Core inference loop (no saving)                                             #
# ──────────────────────────────────────────────────────────────────────────── #

def _run_model(
    triple_unet: torch.nn.Module,
    loader:      DataLoader,
    device:      torch.device,
) -> InferenceResult:
    """
    Runs the forward pass over all batches and collects outputs.
    No file I/O.  Returns an InferenceResult.
    """
    result = InferenceResult()

    triple_unet.eval()
    with torch.no_grad():
        for x, sss_range, sss_altitude, filenames in loader:
            x            = x.to(device)
            sss_range    = sss_range.to(device)
            sss_altitude = sss_altitude.to(device)

            z, rho, path = triple_unet(x, x, x)

            for b in range(z.shape[0]):
                # name = os.path.splitext(filenames[b])[0]
                name = filenames[b]

                z_b        = z[b, 0].to(device)
                rho_b      = rho[b, 0].to(device)
                path_b     = path[b, 0].to(device)
                x_b        = x[b, 0].to(device)
                range_b    = sss_range[b].to(device)
                altitude_b = sss_altitude[b].to(device)

                # ── Derived geometry ────────────────────────────────────────
                slant     = compute_real_dis(x_b, range_b)
                hori      = compute_real_dis_horizontal(slant, altitude_b)
                cos_theta = compute_theta_new(z_b, hori)

                # ── Shadow (differentiable soft version) ────────────────────
                delta, temperature = 0.1, 0.05
                tan_phi       = hori / z_b.clamp(min=1e-3)
                phi_deg       = torch.rad2deg(torch.atan(tan_phi))
                cummax, _     = torch.cummax(phi_deg, dim=1)
                shadow_logits = (cummax - phi_deg - delta) / temperature
                shadow_b      = (
                    (torch.sigmoid(shadow_logits) > 0.5).to(torch.uint8)
                    | (tan_phi <= 0)
                )

                # ── Store as arrays ─────────────────────────────────────────
                result.z       [name] = _tensor_to_colormap(cos_theta, cmap="seismic")
                result.z_gray  [name] = _tensor_to_gray(z_b)
                result.rho     [name] = _tensor_to_colormap(rho_b,     cmap="seismic")
                result.rho_gray[name] = _tensor_to_gray(rho_b)
                result.path    [name] = _tensor_to_colormap(path_b,    cmap="seismic")
                result.theta   [name] = _tensor_to_colormap(cos_theta, cmap="seismic")
                result.shadow  [name] = _tensor_to_binary(shadow_b)

    return result


# ──────────────────────────────────────────────────────────────────────────── #
#  Public entry point                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

def run_inference(
    section:     "SonarSectionData",
    weight_path: str,
    device:      torch.device,
    batch_size:  int = 5,
) -> InferenceResult:
    """
    Run neural-network inference on a single sonar section.
    Processes all images found in section.images regardless of side —
    side selection is handled upstream by prepare_data_range.

    Parameters
    ----------
    section      : SonarSectionData returned by prepare_data_range
    weight_path  : path to the pretrained .pth weights file
    device       : torch device
    batch_size   : DataLoader batch size (default 5)

    Returns
    -------
    InferenceResult — all outputs as numpy arrays, keyed by source image name
    """
    triple_unet = PhysDNet().to(device)
    triple_unet.load_state_dict(torch.load(weight_path, map_location="cpu"))

    combined = InferenceResult()

    for img_key in section.images:
        # Resolve matching altitude and range by the same key
        if img_key not in section.altitudes or img_key not in section.ranges:
            print(f"Warning: no matching altitude/range for '{img_key}', skipping.")
            continue

        dataset = InMemorySonarDataset(
            image    = section.images   [img_key],
            altitude = section.altitudes[img_key],
            range_   = section.ranges   [img_key],
            name     = img_key,
        )

        loader = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = 0,
            pin_memory  = device.type == "cuda",
        )

        side_result = _run_model(triple_unet, loader, device)

        for field_name in InferenceResult.__dataclass_fields__:
            getattr(combined, field_name).update(
                getattr(side_result, field_name)
            )
        print(f"Processed {img_key}")

    return combined


# ──────────────────────────────────────────────────────────────────────────── #
#  Saving utility (call only when you want files on disk)                      #
# ──────────────────────────────────────────────────────────────────────────── #

def save_inference_result(result: InferenceResult, output_dir: str) -> None:
    """
    Saves all arrays in an InferenceResult as flat PNGs into a single output_dir.
    Filename format: {name}_{category}.png
    Example: survey_001.xtf_left_rho_gray.png
    """
    os.makedirs(output_dir, exist_ok=True)

    for field_name in InferenceResult.__dataclass_fields__:
        arrays: dict = getattr(result, field_name)
        for name, arr in arrays.items():
            filename = f"{name}_{field_name}.png"
            img_to_save = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
            cv2.imwrite(os.path.join(output_dir, filename), img_to_save)

    print(f"Inference results saved to: {output_dir}")