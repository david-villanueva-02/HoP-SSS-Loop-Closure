import os
import numpy as np
import cv2
from tqdm import tqdm

altitude_dir = r"D:\dataset\2025\sept\0803_dataset\test\altitude"
range_dir = r"D:\dataset\2025\sept\0803_dataset\test\range"

# Predicted height map
depth_dir = r"D:\dataset\2025\sept\0803_dataset\test\visual_test\z_gray"

# Store the grayscale image of the generated height map and the mask of its understated terrain areas.
height_output_dir = r"D:\dataset\2025\sept\0803_dataset\test\visual_test\height_visual"
output_folder = r"D:\dataset\2025\sept\0803_dataset\test\visual_test\z_mask"

os.makedirs(height_output_dir, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

file_list = sorted(os.listdir(depth_dir))

for filename in tqdm(file_list):
    if not filename.endswith('.png'):
        continue
    base_name = filename.replace('_z.png', '')

    # Build the corresponding .npy file name
    height_map_path = os.path.join(depth_dir, filename)
    altitude_path = os.path.join(altitude_dir, base_name + '.npy')
    range_path = os.path.join(range_dir, base_name + '.npy')

    # Load the corresponding file
    height_map = cv2.imread(height_map_path, cv2.IMREAD_GRAYSCALE)
    altitude = np.load(altitude_path)
    range_array = np.load(range_path)

    oneD_target_size = (1, 512)
    sss_altitude_resized = cv2.resize(altitude.reshape(-1, 1), oneD_target_size, interpolation=cv2.INTER_LINEAR)
    altitude = sss_altitude_resized.flatten()

    sss_range_resized = cv2.resize(range_array.reshape(-1, 1), oneD_target_size, interpolation=cv2.INTER_LINEAR)
    range_array = sss_range_resized.flatten()

    H, W = height_map.shape
    height_mask_map = np.zeros_like(height_map, dtype=np.uint8)

    # Traverse each row and determine the valid seafloor area (only mask out the area where there is real seafloor reflection)
    for i, (val_range, val_altitude) in enumerate(zip(range_array, altitude)):
        if val_range == 0:
            continue
        # Convert seafloor line positions to image column indices
        sub_index = int(val_altitude * W / val_range)
        sub_index = np.clip(sub_index, 0, W - 1)
        height_mask_map[i, sub_index+1:] = 1


    # Calculate the gradient of the mask area in height_map
    grad_x = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Only keep the mask area
    gradient_masked = height_map * height_mask_map

    threshold = np.percentile(gradient_masked[height_mask_map == 1], 80)
    gradient_filtered = gradient_masked.copy()
    gradient_filtered[gradient_masked > threshold] = 0

    def stretch_contrast(image):
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val < 1e-5:
            return np.zeros_like(image, dtype=np.uint8)
        stretched = (image - min_val) / (max_val - min_val) * 255
        return stretched.astype(np.uint8)

    # gradient_masked = stretch_contrast(gradient_masked)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gradient_masked = clahe.apply(255-gradient_masked)

    cv2.imwrite(os.path.join(height_output_dir, filename), gradient_masked)


input_folder = height_output_dir
k = 1
for idx, filename in enumerate(os.listdir(input_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Skip unreadable images
        if img is None:
            print(f"无法读取图像：{img_path}")
            continue

        # Calculate the mean and standard deviation
        mean_val = np.mean(img)
        std_val = np.std(img)
        shadow_mask = (img < mean_val - k * std_val).astype(np.uint8) * 255


        base_name = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(output_folder, f"{base_name}.png"), shadow_mask)

print("Processing complete")