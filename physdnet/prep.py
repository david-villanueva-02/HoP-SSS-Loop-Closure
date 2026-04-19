import pyxtf
import numpy as np
import os
import cv2
from rpds import List

try:
    from .xtf_utils import calculate_blind_zone_indices
except ImportError as e:
    from xtf_utils import calculate_blind_zone_indices

def generate_shadow(img):
    k = 1
    if img is None:
        print("Unable to read image")
    mean_val = np.mean(img)
    std_val = np.std(img)
    shadow_mask = (img < mean_val - k * std_val).astype(np.uint8) * 255

    return shadow_mask

def prepare_data(input_dir: str, output_dir: str, segment_size: int, overlap_size: int, upper_limit: int, side: str):
    '''Prepares the dataset by reading XTF files, processing them, and saving the results in a structured format.
    It works for side = left, right, or both.'''

    if side != "both": output_dir += f"_{side}"
    os.makedirs(output_dir, exist_ok=True)

    # Scan all .xtf files
    xtf_files_or = [f for f in os.listdir(input_dir) if f.lower().endswith('.xtf')]
    xtf_files = [xtf_files_or[0]] # Taking only the first file

    if overlap_size >= segment_size:
        raise ValueError("overlap_size 必须小于 segment_size")

    stride = segment_size - overlap_size

    # For all the files in the directory
    for file_idx, xtf_filename in enumerate(xtf_files):
        
        file_path = os.path.join(input_dir, xtf_filename)

        # Initialize data storage
        left_channel_data = []
        right_channel_data = []
        range_left_list = []
        range_right_list = []
        altitude_list = []

        print(f'Processing file: {xtf_filename}')
        try:
            file_header, packets = pyxtf.xtf_read(file_path)
            sonar_packets: list[pyxtf.XTFPingHeader] = packets[pyxtf.XTFHeaderType.sonar]

            # Remove from blind zones
            left_idx, right_idx = calculate_blind_zone_indices(sonar_packets)

            mask_left = None
            mask_right = None

            for packet in sonar_packets:
                altitude = packet.SensorPrimaryAltitude
                range_left = packet.ping_chan_headers[0].SlantRange
                range_right = packet.ping_chan_headers[1].SlantRange

                altitude_list.append(altitude)
                range_left_list.append(range_left)
                range_right_list.append(range_right)

                if hasattr(packet, "data"):
                    # Filter and accumulate data
                    ping_data = getattr(packet, "data", None)


                    if isinstance(ping_data, list) and len(ping_data) >= 2:
                        if mask_left is None or mask_right is None:
                            mask_right = np.zeros((segment_size, ping_data[0].shape[0]))*0
                            mask_right[:, left_idx] = 255
                            mask_left = np.flip(mask_right)
                            
                            # Invert left mask
                            mask_left_inv = mask_left.copy()
                            mask_left_inv[mask_left == 0] = 255
                            mask_left_inv[mask_left == 255] = 0
                            
                            # Invert right mask
                            mask_right_inv = mask_right.copy()
                            mask_right_inv[mask_right == 0] = 255
                            mask_right_inv[mask_right == 255] = 0

                            print("Mask calculated!")
                            print(f"mask_left: {mask_left_inv.shape}, mask_right: {mask_right_inv.shape}")

                        left_channel  = ping_data[0]
                        right_channel = ping_data[1]

                        if isinstance(left_channel, np.ndarray) and isinstance(right_channel, np.ndarray):
                            left_channel_data.append(left_channel)
                            right_channel_data.append(right_channel)

            # Convert back into numpy arrays
            left_channel_data = np.array(left_channel_data)
            right_channel_data = np.array(right_channel_data)
            altitude_list = np.array(altitude_list)
            range_left_list = np.array(range_left_list)
            range_right_list = np.array(range_right_list)

            # DEBUG
            print(f"left channel data shape {left_channel_data.shape}")
            print(f"left channel data shape {left_channel_data.size}")

            if left_channel_data.size > 0 and right_channel_data.size > 0:
                left_channel_data = np.clip(left_channel_data, 0, upper_limit - 1)
                right_channel_data = np.clip(right_channel_data, 0, upper_limit - 1)

                left_channel_data = np.log10(left_channel_data + 1e-4)
                right_channel_data = np.log10(right_channel_data + 1e-4)

                left_image = np.vstack(left_channel_data)
                right_image = np.vstack(right_channel_data)

                vmax = np.log10(upper_limit)
                left_image_normalized = ((left_image - 0) / (vmax - 0) * 255).astype(np.uint8)
                right_image_normalized = ((right_image - 0) / (vmax - 0) * 255).astype(np.uint8)

                combined_image = np.hstack((left_image_normalized, right_image_normalized))
                cv2.imwrite(os.path.join(output_dir, f"{xtf_filename}_combined.png"), combined_image)

                # 统一目录结构
                dirs = ["images", "shadow", "altitude", "range"]
                for d in dirs:
                    os.makedirs(os.path.join(output_dir, d), exist_ok=True)

                # -----------------------------
                # Segment processing with overlap
                # -----------------------------
                total_rows = left_image_normalized.shape[0]

                # 生成所有起点
                start_positions = list(range(0, total_rows - segment_size + 1, stride))

                # 如果最后剩余部分没有被覆盖，可以补一个最后窗口
                if len(start_positions) == 0:
                    print(f"Warning: File {xtf_filename} 行数不足一个 segment，跳过")
                    continue

                last_start = total_rows - segment_size
                if start_positions[-1] != last_start:
                    start_positions.append(last_start)

                for i, start in enumerate(start_positions):
                    end = start + segment_size

                    if side == "left" or side == "both":

                        # ----- left -----
                        image_left_data = left_image_normalized[start:end, :]
                        altitude_left_segment = altitude_list[start:end]
                        range_left_segment = range_left_list[start:end]

                        flipped_image_left = np.fliplr(image_left_data)
                        left_mask = generate_shadow(flipped_image_left)

                        cv2.imwrite(
                            os.path.join(output_dir, "images", f"{xtf_filename}_left_{i:03d}.png"),
                            flipped_image_left
                        )
                        cv2.imwrite(
                            os.path.join(output_dir, "shadow", f"{xtf_filename}_left_{i:03d}.png"),
                            left_mask
                        )
                        np.save(
                            os.path.join(output_dir, "altitude", f"{xtf_filename}_left_{i:03d}.npy"),
                            altitude_left_segment
                        )
                        np.save(
                            os.path.join(output_dir, "range", f"{xtf_filename}_left_{i:03d}.npy"),
                            range_left_segment
                        )
                    
                    if side == "right" or side == "both":

                        # ----- right -----
                        image_right_data = right_image_normalized[start:end, :]
                        altitude_right_segment = altitude_list[start:end]
                        range_right_segment = range_right_list[start:end]

                        right_mask = generate_shadow(image_right_data)

                        cv2.imwrite(
                            os.path.join(output_dir, "images", f"{xtf_filename}_right_{i:03d}.png"),
                            image_right_data
                        )
                        cv2.imwrite(
                            os.path.join(output_dir, "shadow", f"{xtf_filename}_right_{i:03d}.png"),
                            right_mask
                        )
                        np.save(
                            os.path.join(output_dir, "altitude", f"{xtf_filename}_right_{i:03d}.npy"),
                            altitude_right_segment
                        )
                        np.save(
                            os.path.join(output_dir, "range", f"{xtf_filename}_right_{i:03d}.npy"),
                            range_right_segment
                        )

                    cv2.imwrite(
                        os.path.join(output_dir, f"{xtf_filename}_mask_left.png"),
                        mask_left_inv
                    )
                    cv2.imwrite(
                        os.path.join(output_dir, f"{xtf_filename}_mask_right.png"),
                        mask_right_inv
                    )

                print(f"{xtf_filename}: total_rows={total_rows}, segments={len(start_positions)}, stride={stride}")

            else:
                print(f"Warning: File {xtf_filename} has no valid channel data")

        except Exception as e:
            print(f"Error reading or processing {xtf_filename}: {e}")

if __name__ == "__main__":
    input_dir = r'data'  # XTF file folder
    output_dir = r'output'  # Output dataset folder
    segment_size = 2000
    overlap_size = 1000
    upper_limit = 2 ** 15

    prepare_data(input_dir, output_dir, segment_size, overlap_size, upper_limit, side="left")