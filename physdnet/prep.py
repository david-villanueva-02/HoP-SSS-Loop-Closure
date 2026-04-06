import pyxtf
import numpy as np
import os
import cv2

input_dir = r'D:\dataset\2025\sept\0803'  # XTF file folder
output_dir = r'D:\dataset\2025\sept\0803_dataset'

os.makedirs(output_dir, exist_ok=True)

# Scan all .xtf files
xtf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.xtf')]

# The segment_size is the same as the number of sonar samples.
segment_size = 2000
# overlap_size is the number of repeated pings between two samples (used to augment the dataset).
overlap_size = 1000
# upper_limit is the maximum value of the sound intensity collected in the original XTF file.
upper_limit = 2 ** 15

def generate_shadow(img):
    k = 1
    if img is None:
        print("Unable to read image")
    mean_val = np.mean(img)
    std_val = np.std(img)
    shadow_mask = (img < mean_val - k * std_val).astype(np.uint8) * 255

    return shadow_mask

if overlap_size >= segment_size:
    raise ValueError("overlap_size 必须小于 segment_size")

stride = segment_size - overlap_size

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
        sonar_packets = packets[pyxtf.XTFHeaderType.sonar]

        for packet in sonar_packets:
            altitude = packet.SensorPrimaryAltitude
            range_left = packet.ping_chan_headers[0].SlantRange
            range_right = packet.ping_chan_headers[1].SlantRange

            altitude_list.append(altitude)
            range_left_list.append(range_left)
            range_right_list.append(range_right)

            if hasattr(packet, "data"):
                ping_data = getattr(packet, "data", None)
                if isinstance(ping_data, list) and len(ping_data) >= 2:
                    left_channel = ping_data[0]
                    right_channel = ping_data[1]

                    if isinstance(left_channel, np.ndarray) and isinstance(right_channel, np.ndarray):
                        left_channel_data.append(left_channel)
                        right_channel_data.append(right_channel)

        left_channel_data = np.array(left_channel_data)
        right_channel_data = np.array(right_channel_data)
        altitude_list = np.array(altitude_list)
        range_left_list = np.array(range_left_list)
        range_right_list = np.array(range_right_list)

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

            print(f"{xtf_filename}: total_rows={total_rows}, segments={len(start_positions)}, stride={stride}")

        else:
            print(f"Warning: File {xtf_filename} has no valid channel data")

    except Exception as e:
        print(f"Error reading or processing {xtf_filename}: {e}")