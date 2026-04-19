import os
from typing import List, Any 
import pyxtf 

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