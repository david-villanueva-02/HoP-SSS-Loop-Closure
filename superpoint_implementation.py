import cv2
import torch
import numpy as np

# pseudo-code
img = cv2.imread(rho_path, cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32) / 255.0

tensor = torch.from_numpy(img)[None, None]   # [1,1,H,W]
tensor = tensor.to(device)

with torch.no_grad():
    result = superpoint({'image': tensor})

keypoints = result['keypoints'][0]      # [N,2]
descriptors = result['descriptors'][0]  # [D,N] or [N,D] depending on implementation
scores = result['scores'][0]

shadow = cv2.imread(shadow_path, cv2.IMREAD_GRAYSCALE)
zmask = cv2.imread(zmask_path, cv2.IMREAD_GRAYSCALE)

filtered_kpts = []
filtered_desc = []
filtered_scores = []

for i, (x, y) in enumerate(keypoints.cpu().numpy()):
    xi, yi = int(round(x)), int(round(y))
    if shadow[yi, xi] == 0 and zmask[yi, xi] == 0:
        filtered_kpts.append(keypoints[i])
        filtered_desc.append(descriptors[i])
        filtered_scores.append(scores[i])