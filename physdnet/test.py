import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import cv2
from .train import PhysDNet


def compute_real_dis(A: torch.Tensor, M_list: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    if M_list.shape[0] != M:
        raise ValueError("The length of M_list must be equal to the number of matrix rows M")
    device = A.device
    col_indices = torch.arange(N, device=device)
    D = (M_list[:, None] * col_indices[None, :]) / 512.0

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


def compute_shadow(hori, z_b):
    h = z_b
    tan_fai = hori / h
    fai_rad = torch.atan(tan_fai)
    fai_deg = torch.rad2deg(fai_rad)
    cummax, _ = torch.cummax(fai_deg, dim=1)
    mask = (fai_deg < cummax) | (tan_fai <= 0)

    return mask.to(torch.uint8), fai_rad


class TestSonarImageDataset(Dataset):
    def __init__(self, data_dir, range_dir, altitude_dir, transform=None):
        """
       Args:
        data_dir (str): path to directory containing .png or .jpg images
        transform (callable, optional): image augmentation transform (e.g. ToTensor, Normalize)
        """
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        self.range_dir = range_dir
        self.range_files = [f for f in os.listdir(range_dir) if f.endswith('.npy')]

        self.altitude_dir = altitude_dir
        self.altitude_files = [f for f in os.listdir(altitude_dir) if f.endswith('.npy')]

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        image_path = os.path.join(self.data_dir, fname)

        image = Image.open(image_path).convert('L')
        image_np = np.array(image).astype(np.float32) / 255.0

        # Convert to Tensor and add channel dimension (1, H, W)
        input_tensor = torch.tensor(image_np).unsqueeze(0)

        if self.transform:
            input_tensor = self.transform(input_tensor)

        oneD_target_size = (1, 512)

        fname_range = self.range_files[idx]
        sss_range = np.load(os.path.join(self.range_dir, fname_range)).astype(np.float32)
        # Downsampling a 1D array using linear interpolation
        sss_range_resized = cv2.resize(sss_range.reshape(-1, 1), oneD_target_size, interpolation=cv2.INTER_LINEAR)
        sss_range_resized = sss_range_resized.flatten()

        fname_altitude = self.altitude_files[idx]
        sss_altitude = np.load(os.path.join(self.altitude_dir, fname_altitude)).astype(np.float32)
        sss_altitude_resized = cv2.resize(sss_altitude.reshape(-1, 1), oneD_target_size, interpolation=cv2.INTER_LINEAR)
        sss_altitude_resized = sss_altitude_resized.flatten()

        return input_tensor, sss_range_resized, sss_altitude_resized, fname

def pre_model(triple_unet, val_loader, weight_path, output_dir, device):

    triple_unet.load_state_dict(torch.load(weight_path, map_location='cpu'))  # 加载权重
    triple_unet.eval()

    os.makedirs(os.path.join(output_dir, "z"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "z_gray"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rho"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rho_gray"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "path"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "theta"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "phi"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "shadow"), exist_ok=True)

    def tensor_to_twovalue(tensor):
        arr = tensor.detach().cpu().numpy()
        arr = np.nan_to_num(arr)  # 去除 NaN
        arr = (arr > 0).astype(np.uint8) * 255  # 将非零值映射为255
        return arr
    
    def numpy_to_img(arr):
        if isinstance(arr, torch.Tensor): 
            arr = arr.detach().cpu().numpy()
        arr = np.nan_to_num(arr)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        return arr.astype(np.uint8)

    def visualize_matrix_auto(matrix, save_path='', cmap='seismic', show_colorbar=True,
                              show_axis=True, symmetric=False):
        """
        para:
            matrix (np.ndarray)
            save_path (str)
            cmap (str)
            show_colorbar (bool)
            show_axis (bool)
            symmetric (bool)
        """
        matrix = matrix.detach().cpu().numpy()

        min_val, max_val = matrix.min(), matrix.max()
        if symmetric:
            abs_max = max(abs(min_val), abs(max_val))
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = min_val, max_val

        plt.figure(figsize=(6, 5))
        im = plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)

        if show_colorbar:
            plt.colorbar(im, label='Value')

        if not show_axis:
            plt.axis('off')

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Image saved to: {save_path}")

    with torch.no_grad():  # No need to calculate gradients
        for x, sss_range, sss_altitude, filenames in val_loader:
            x = x.to(device)
            z, rho, path = triple_unet(x, x, x)
            B = z.shape[0]
            for b in range(B):
                filename = filenames[b]
                name_wo_ext = os.path.splitext(filename)[0]

                # predicted depth, reflectivity, path loss
                z_b = z[b, 0].to(device)
                rho_b = rho[b, 0].to(device)
                path_b = path[b, 0].to(device)

                x_b = x[b, 0].to(device)
                range_b = sss_range[b].to(device)
                altitude_b = sss_altitude[b].to(device)

                slant = compute_real_dis(x_b, range_b)
                hori = compute_real_dis_horizontal(slant, altitude_b)
                cos_theta = compute_theta_new(z_b, hori)

                delta = 0.1
                # delta = 0.1,
                temperature = 0.05
                tan_phi = hori / z_b.clamp(min=1e-3)
                phi_rad = torch.atan(tan_phi)
                phi_deg = torch.rad2deg(phi_rad)
                cummax, _ = torch.cummax(phi_deg, dim=1)
                shadow_logits = (cummax - phi_deg - delta) / temperature
                shadow_pre_b = (torch.sigmoid(shadow_logits) > 0.5).to(torch.uint8) | (tan_phi <= 0)

                visualize_matrix_auto(cos_theta, save_path=os.path.join(output_dir, "theta", f"{name_wo_ext}_theta.png"), symmetric=False) # incidence factor (is it theta or cos theta?)
                visualize_matrix_auto(z_b, save_path=os.path.join(output_dir, "z", f"{name_wo_ext}_z.png"), symmetric=False)               # height z
                visualize_matrix_auto(path_b, save_path=os.path.join(output_dir, "path", f"{name_wo_ext}_path.png"), symmetric=False)      # path loss L
                visualize_matrix_auto(rho_b, save_path=os.path.join(output_dir, "rho", f"{name_wo_ext}_rho.png"), symmetric=False)         # reflectivity rho
                cv2.imwrite(os.path.join(output_dir, "rho_gray", f"{name_wo_ext}_rho_gray.png"), numpy_to_img(rho_b))                      # saves reflectivity as grayscale image
                cv2.imwrite(os.path.join(output_dir, "z_gray", f"{name_wo_ext}_z.png"), numpy_to_img(z_b))                                 # saves height as grayscale image
                cv2.imwrite(os.path.join(output_dir, "shadow", f"{name_wo_ext}_shadow.png"),                                                
                            tensor_to_twovalue(shadow_pre_b))
                #cv2.imwrite(os.path.join(output_dir, "rho", f"{name_wo_ext}_rho.png"), numpy_to_img(rho_b)) # why is this commented?

def run_inference(device: torch.device, 
         test_image_path: str, test_range_path: str, test_altitude_path: str, weight_path: str, 
         output_dir: str, side: str, prefix: str = "output"): 
    
    if side == "left": 
        test_image_path = test_image_path.replace(prefix, prefix + "_left")
        test_range_path = test_range_path.replace(prefix, prefix + "_left")
        test_altitude_path = test_altitude_path.replace(prefix, prefix + "_left")
        output_dir = output_dir.replace(prefix, prefix + "_left")
    if side == "right": 
        test_image_path = test_image_path.replace(prefix, prefix + "_right")
        test_range_path = test_range_path.replace(prefix, prefix + "_right")
        test_altitude_path = test_altitude_path.replace(prefix, prefix + "_right")
        output_dir = output_dir.replace(prefix, prefix + "_right")

    triple_unet = PhysDNet().to(device)

    custom_transform = transforms.Compose([
        transforms.Resize((512, 512)),
    ])

    test_dataset = TestSonarImageDataset(test_image_path, test_range_path, test_altitude_path, transform=custom_transform)

    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=2, pin_memory=True)

    pre_model(triple_unet, test_loader, weight_path, output_dir, device)

if __name__ == '__main__':

    # Testing set data path
    test_image_path = r'output/images'
    test_range_path = r'output/range'
    test_altitude_path = r'output/altitude'

    # Trained weight path
    weight_path = r'best_model_v610_eagle.pth'
    # weight_path = r'best_model_v610_jaguar.pth'
    # weight_path = r'best_model_v611_jaguar.pth'
    
    # Result saving path
    output_dir = r'output/visual_test'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    run_inference(device, test_image_path, test_range_path, test_altitude_path, weight_path, output_dir)