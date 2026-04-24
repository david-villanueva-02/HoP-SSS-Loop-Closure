import os
from typing import List, Any 
import pyxtf 
import cv2 
import torch
from pathlib import Path

LEN_RANGE = 2000

def load_xtf(xtf_file_path: str) -> List[Any]:
    """Extract XTF ping data from XTF file 

    Args:
        xtf_file_path (str): XTF file path 

    Raises:
        FileNotFoundError: file not found on disk 
        TypeError: file not correct file type 

    Returns:
        List[Any]: XTF ping data 
    """

    if not os.path.exists(xtf_file_path):
        raise FileNotFoundError("Invalid Path", xtf_file_path)

    if not xtf_file_path.endswith('.xtf'):
        raise TypeError("Invalid File", xtf_file_path)
    
    (fh, packet) = pyxtf.xtf_read(xtf_file_path)   # Returns file header and packets
    xtf_pings = packet[pyxtf.XTFHeaderType.sonar]  # Extract sonar pings from packets
    print("---------------------")
    print(fh)
    print("XTF File loaded!")
    print("---------------------")
    return xtf_pings


from typing import Tuple 
import numpy as np 

def calculate_blind_zone_indices(xtf_pings: List[pyxtf.XTFPingHeader]) -> Tuple[slice, slice]:
    """Calculates indices of waterfall's non-blind zone

    Args:
        xtf_pings (List[pyxtf.XTFPingHeader]): Input XTF ping data 

    Returns:
        Tuple[slice, slice]: Waterfall column indices corresponding to pixels that do not fall in blind zone 
    """
    '''
    xtf_pings is a list with pings, each one with differen attributes
    xtf_pints[0] gets the first element
    ping_chan_headers [0] or [1] gets the info from either channel of the SSS
    '''
    num_samples = xtf_pings[0].ping_chan_headers[0].NumSamples * 2
    slant_res = xtf_pings[0].ping_chan_headers[0].SlantRange * 2 / num_samples
    altitude = np.max([ping.SensorPrimaryAltitude for ping in xtf_pings]) # Gets tha max altitude among all pings, which is the one that determines the blind zone size

    # Gets the numbre of bins inside and outside the blind zone, which is the same for both channels
    num_bins_blind = int(round(altitude / slant_res))            
    num_bins_ground = int(num_samples / 2 - num_bins_blind)   
    
    # Gets the indices of the non-blind zone for both channels
    port_idx = slice(0, num_bins_ground, 1)
    stbd_idx = slice(num_bins_blind, -1, 1)

    print(f"\tNumber of blind bins: {num_bins_blind*2}")
    return port_idx, stbd_idx

from pyproj import Proj, CRS

def calculate_swath_positions(xtf_pings: List[pyxtf.XTFPingHeader]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates geographical locations of the pixels of the waterfall, and the trajectory of the side-scan-sonar

    Args:
        xtf_pings (List[pyxtf.XTFPingHeader]): Input XTF ping data 

    Returns:
        Tuple[np.ndarray, np.ndarray]: Geographical locations of the pixels of the waterfall, and the trajectory of the side-scan-sonar
    """
    # Define projection for coordinate transformation (from longitude-latitude to East-North)
    lonlat_to_EN = Proj(CRS.from_epsg(25831), preserve_units=False)
    ping_info = xtf_pings[0].ping_chan_headers[0]

    # Fetch data dimensions
    num_pings = len(xtf_pings) 
    num_samples = ping_info.NumSamples * 2 # For each channel ig
    print(f"Number of pings: {num_pings}\nNumber of samples: {num_samples}")

    # Compute swath resolution
    slant_range = ping_info.SlantRange         
    slant_res = slant_range * 2 / num_samples  # "Thickness" of the swath
    print(f"Slant range: {slant_range} [metres]\nSlant resolution: {slant_res} [metres/bin]")

    # Fetch navigation parameters
    longitude, latitude, altitude, roll, pitch, yaw = zip(*[(ping.SensorXcoordinate, ping.SensorYcoordinate,
                                                                ping.SensorPrimaryAltitude, ping.SensorRoll,
                                                                ping.SensorPitch, ping.SensorHeading)
                                                                for ping in xtf_pings])

    # Convert to East-North coordinates
    east, north = lonlat_to_EN(longitude, latitude)
    altitude = np.asarray(altitude).reshape(num_pings, 1)

    # Convert to radians
    roll, pitch, yaw = np.radians(roll), np.radians(pitch), np.radians(yaw)

    bin_central = (num_samples - 1) / 2
    bins = np.arange(num_samples).reshape(1, num_samples)          
    bins_from_center = bins - bin_central                      

    # Number of bins corresponding to the Blind Zone per side
    n_bins_blind = np.round(altitude / slant_res)    

    # Number of bins corresponding to Ground Range per side
    n_bins_ground = (num_samples / 2 - n_bins_blind)            

    # Bins inside the blind zone, those whose slant range is smaller than the altitude of the sonar, and thus do not correspond to any point on the ground
    blind_idx = (n_bins_ground <= bins) & \
                (bins < n_bins_ground + 2 * n_bins_blind)      

    # Increments along the x-axis (swath width) from the altitude of the sonar and the slant range
    X = np.zeros((num_pings, num_samples))
    np.sqrt(np.square(slant_res * bins_from_center) -
            np.square(altitude).reshape(num_pings, 1),
            where=~blind_idx, out=X) # Only for not blind bins, the rest are set to 0
    X *= np.sign(bins_from_center) 

    # Increment along y-axis 
    Y = np.repeat(altitude * np.tan(pitch[:,None]), num_samples, 1)                          

    # Ping coordinates (swath center)
    T = np.vstack((east, north)).T          

    # Rotation of horizontal axis (swath), displaced along forward axis (by pitch), about the z-axis (heading)
    Rx = np.vstack((np.cos(yaw), -np.sin(yaw))).T 
    Ry = np.vstack((np.sin(yaw), np.cos(yaw))).T           

    X = np.expand_dims(X, axis=2)    
    Y = np.expand_dims(Y, axis=2)                        
    T = np.expand_dims(T, axis=1)                          
    Rx = np.expand_dims(Rx, axis=1)
    Ry = np.expand_dims(Ry, axis=1)                             

    # Compute the transformation
    swaths = T + Rx * X + Ry * Y 
    trajectory = np.vstack([east, north]).T
    print("Swath positions calculated!")  
    print(f"Swath positions shape: {swaths.shape}")   
                            
    return swaths, trajectory, altitude, roll, pitch, yaw


def calculate_waterfall(xtf_pings: List[pyxtf.XTFPingHeader], 
                        channels: List[int] = [0,1]) -> np.ndarray:
    """Calculates waterfall from XTF ping data 

    Args:
        xtf_pings (List[pyxtf.XTFPingHeader]): Input XTF ping data 
        channels (List[int], optional): Channel indices from the XTF file. Defaults to [0,1].

    Returns:
        np.ndarray: Waterfall image 
    """

    num_channels = len(xtf_pings[0].data) # Number of channels, data is a list of arrays, one per channel, therefore len(data) gives the number of channels
    assert len(channels) ==2 and channels[0] < num_channels and channels[1] < num_channels # Verify number of channels consistency
    
    # Load all the intensity information 
    # List of two channels, for each one extract all data, each element is an array of the samples of that ping of that channel, so this is a 3D array (or a list of two 2D arrays)
    sonar_chans = [np.vstack([ping.data[i] for ping in xtf_pings]) for i in channels] 

    # Stack port and starboard data side by side 
    sonar_image = np.hstack( (sonar_chans[channels[0]], sonar_chans[channels[1]]) ) # Put the two lists side by side, this is a 2D array now
    print(f"Waterfall shape: {sonar_image.shape}")

    bit_depth = 2**(16 - 1)
    waterfall_img = np.log1p(sonar_image) / np.log1p(bit_depth)    # Logarithmic normalization to [0,1]
    return waterfall_img, sonar_image

def numerical_derivative_5point(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 5:
        raise ValueError("Need at least 5 points")

    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h):
        raise ValueError("This 5-point formula assumes equally spaced x values")

    dydx = np.empty_like(y)

    # Forward 5-point formula for first two points
    dydx[0] = (-25*y[0] + 48*y[1] - 36*y[2] + 16*y[3] - 3*y[4]) / (12*h)
    dydx[1] = (-3*y[0] - 10*y[1] + 18*y[2] - 6*y[3] + y[4]) / (12*h)

    # Central 5-point formula for interior points
    for i in range(2, len(y) - 2):
        dydx[i] = (y[i-2] - 8*y[i-1] + 8*y[i+1] - y[i+2]) / (12*h)

    # Backward 5-point formula for last two points
    dydx[-2] = (-y[-5] + 6*y[-4] - 18*y[-3] + 10*y[-2] + 3*y[-1]) / (12*h)
    dydx[-1] = (3*y[-5] - 16*y[-4] + 36*y[-3] - 48*y[-2] + 25*y[-1]) / (12*h)

    return dydx

def lowpass_and_derivative_5point(x, y, alpha=0.2):
    """
    Applies a simple low-pass filter first, then computes the numerical
    derivative using a 5-point finite-difference stencil.

    Parameters
    ----------
    x : array-like
        Equally spaced x values.
    y : array-like
        Signal values.
    alpha : float
        Low-pass filter coefficient in (0, 1].
        Smaller -> stronger smoothing.

    Returns
    -------
    y_filt : np.ndarray
        Low-pass filtered signal.
    dydx : np.ndarray
        Numerical derivative of filtered signal.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 5:
        raise ValueError("Need at least 5 points")

    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h):
        raise ValueError("This 5-point formula assumes equally spaced x values")

    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in the range (0, 1]")

    # Low-pass filter: exponential smoothing
    y_filt = np.empty_like(y)
    y_filt[0] = y[0]
    for i in range(1, len(y)):
        y_filt[i] = alpha * y[i] + (1 - alpha) * y_filt[i - 1]

    # 5-point derivative
    dydx = np.empty_like(y_filt)

    # Forward formulas
    dydx[0] = (-25*y_filt[0] + 48*y_filt[1] - 36*y_filt[2] + 16*y_filt[3] - 3*y_filt[4]) / (12*h)
    dydx[1] = (-3*y_filt[0] - 10*y_filt[1] + 18*y_filt[2] - 6*y_filt[3] + y_filt[4]) / (12*h)

    # Central formula
    for i in range(2, len(y_filt) - 2):
        dydx[i] = (y_filt[i - 2] - 8*y_filt[i - 1] + 8*y_filt[i + 1] - y_filt[i + 2]) / (12*h)

    # Backward formulas
    dydx[-2] = (-y_filt[-5] + 6*y_filt[-4] - 18*y_filt[-3] + 10*y_filt[-2] + 3*y_filt[-1]) / (12*h)
    dydx[-1] = (3*y_filt[-5] - 16*y_filt[-4] + 36*y_filt[-3] - 48*y_filt[-2] + 25*y_filt[-1]) / (12*h)

    return y_filt, dydx

def get_range_window(group: np.ndarray, timestamp: int, len_range: int = LEN_RANGE) -> tuple[int, int]:
    lower_ind: int = timestamp - len_range/2 # Ideal value for the timestamp
    upper_ind: int = timestamp + len_range/2

    print(f"Group corresponding to timestamp: {timestamp} has {len(group)} pings.")
    if len(group) < LEN_RANGE:
        # The groups is too short, take it completely
        print(f"Warning, the transact corrsponding to timestamp: {timestamp} is too short, it has only {len(group)} pings. Consider selecting a different timestamp or reducing the segment size. The window will be set from the beginning of the group + 2000 pings.")
        lower_ind = group[0]
        upper_ind = group[0] + LEN_RANGE
    elif len(group) == LEN_RANGE:
        print(f"The window is exactly 2000, so it will be taken as the window for timestamp: {timestamp}.")
        lower_ind = group[0]
        upper_ind = group[-1]

        # From now on, assume the len of the group is larger than 2000
    elif lower_ind not in group: # Lower limit is not in the group
        lower_ind = group[0]
        upper_ind = lower_ind + LEN_RANGE
        print(f"Warning, the ideal lower index {lower_ind} is not in the group, so it will be set to the first index of the group: {group[0]}. The upper index will be set to {upper_ind}.")
        upper_ind = lower_ind + LEN_RANGE
    elif upper_ind not in group: # Upper limit is not in the group
        upper_ind = group[-1]
        lower_ind = upper_ind - LEN_RANGE
        print(f"Warning, the ideal upper index {upper_ind} is not in the group, so it will be set to the last index of the group: {group[-1]}. The lower index will be set to {lower_ind}.")
    else:
        print(f"The ideal window for timestamp: {timestamp} is valid and will be set from {lower_ind} to {upper_ind}.")
    
    return int(lower_ind), int(upper_ind)

def combine_masks(*mask_paths, output_path):
    '''Combines multiple binary masks into one by taking the union of their valid areas.'''
    masks = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in mask_paths]

    shape = masks[0].shape # All masks should have the same shape as the first mask
    for i, mask in enumerate(masks):
        if mask.shape != shape:
            masks[i] = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    combined_mask = (sum(mask > 0 for mask in masks) > 0).astype(np.uint8) * 255

    cv2.imwrite(Path(output_path), combined_mask)

def prepare_mask(mask: torch.Tensor, device):
    mask = mask.to(device)

    if mask.dim() == 3:
        # if RGB, take one channel
        mask = mask[0]

    # binary: 1 invalid, 0 valid
    mask = (mask > 0.5).to(torch.uint8)
    return mask

def warp_and_overlay(
    img_ref: np.ndarray,
    img_to_warp: np.ndarray,
    H: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Warp img_to_warp into the frame of img_ref using H and create a canvas
    large enough to show both images entirely.

    H must map points from img_to_warp -> img_ref.
    """
    h_ref, w_ref = img_ref.shape[:2]
    h_warp, w_warp = img_to_warp.shape[:2]

    # Corners of both images
    corners_ref = np.array([
        [0, 0],
        [w_ref, 0],
        [w_ref, h_ref],
        [0, h_ref]
    ], dtype=np.float32).reshape(-1, 1, 2)

    corners_warp = np.array([
        [0, 0],
        [w_warp, 0],
        [w_warp, h_warp],
        [0, h_warp]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners of img_to_warp into img_ref coordinates
    warped_corners = cv2.perspectiveTransform(corners_warp, H)

    # Combine all corners to find full canvas bounds
    all_corners = np.vstack((corners_ref, warped_corners)).reshape(-1, 2)

    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

    # Translation to shift everything into positive canvas coordinates
    tx = -x_min
    ty = -y_min

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)

    out_w = x_max - x_min
    out_h = y_max - y_min

    # Warp the moving image into the new canvas
    warped = cv2.warpPerspective(img_to_warp, T @ H, (out_w, out_h))

    # Place reference image into the same canvas
    canvas_ref = np.zeros((out_h, out_w, 3), dtype=img_ref.dtype)
    canvas_ref[ty:ty + h_ref, tx:tx + w_ref] = img_ref

    # Overlay
    overlay = cv2.addWeighted(canvas_ref, alpha, warped, 1 - alpha, 0)

    return overlay