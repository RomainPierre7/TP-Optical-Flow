import sys
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append("core")

from utils import flow_viz
from utils.utils import InputPadder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def read_flo_file(filename):
    """Read a .flo optical flow file."""
    with open(filename, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise ValueError(f"Invalid .flo file: incorrect magic number ({magic}).")

        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]

        # Read the flow data
        data = np.fromfile(f, np.float32, count=2 * width * height)
        data = data.reshape((height, width, 2))  # Shape: (H, W, 2)

    # Convert to tensor (shape: H, W, 2)
    flow_tensor = torch.tensor(data).permute(2, 0, 1)  # Shape: (2, H, W)

    target_height = 436
    target_width = 1024
    flow_rescaled = F.interpolate(
        flow_tensor.unsqueeze(0),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    return flow_rescaled.squeeze(0)  # Shape: (2, 440, 1024)


def compute_metrics(img1, img2, flow_estimated, flow_gt):
    """
    Compute the average End Point Error (aEPE),
    average Angular Error (aAE), and
    Mean Square Error (MSE) between the original and compensated images.

    Parameters:
        img1 (torch.Tensor): The first image (1, 3, H, W).
        img2 (torch.Tensor): The second image (1, 3, H, W).
        flow_estimated (torch.Tensor): Estimated optical flow (2, H, W).
        flow_gt (torch.Tensor): Ground-truth optical flow (2, H, W).

    Returns:
        tuple: (aEPE, aAE, mse)
            - aEPE: Average End Point Error.
            - aAE: Average Angular Error.
            - mse: Mean Square Error between img1 and the compensated img2.
    """
    # Remove batch dimension
    img1 = img1[0]  # Shape: (3, H, W)
    img2 = img2[0]  # Shape: (3, H, W)

    # Image dimensions
    _, H, W = img1.shape

    # Meshgrid for pixel coordinates
    x = torch.arange(W, device=flow_estimated.device).view(1, -1).expand(H, W)
    y = torch.arange(H, device=flow_estimated.device).view(-1, 1).expand(H, W)

    x_comp = x + flow_estimated[0]  # Add horizontal flow
    y_comp = y + flow_estimated[1]  # Add vertical flow

    # Interpolate img2 to create the compensated image for each channel
    compensated_img = torch.zeros_like(img1)
    for c in range(3):
        compensated_img[c] = bilinear_interpolation(img2[c], x_comp, y_comp)

    # Mean Square Error (MSE)
    mse = torch.mean((img1 - compensated_img) ** 2).item()

    # End Point Error (EPE)
    epe = torch.sqrt(
        (flow_estimated[0] - flow_gt[0]) ** 2 + (flow_estimated[1] - flow_gt[1]) ** 2
    )
    aEPE = torch.mean(epe).item()

    # Angular Error (AE)
    dot_product = flow_estimated[0] * flow_gt[0] + flow_estimated[1] * flow_gt[1]
    norm_est = torch.sqrt(flow_estimated[0] ** 2 + flow_estimated[1] ** 2)
    norm_gt = torch.sqrt(flow_gt[0] ** 2 + flow_gt[1] ** 2)
    cos_theta = torch.clamp(dot_product / (norm_est * norm_gt + 1e-6), -1.0, 1.0)
    angular_error = torch.acos(cos_theta)
    aAE = torch.mean(angular_error).item()

    return aEPE, aAE, mse


def bilinear_interpolation(img, x, y):
    """
    Perform bilinear interpolation for image sampling.

    Parameters:
        img (torch.Tensor): Input image (H, W).
        x (torch.Tensor): X-coordinates for sampling.
        y (torch.Tensor): Y-coordinates for sampling.

    Returns:
        torch.Tensor: Interpolated image.
    """
    H, W = img.shape

    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def viz(img, flo, flow_gt):
    print(type(img), type(flo), type(flow_gt))
    print(img.shape, flo.shape, flow_gt.shape)
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    flow_gt = flow_gt.permute(1, 2, 0).cpu().numpy()

    # Map flow to RGB image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    img_flo_gt = np.concatenate([img_flo, flow_viz.flow_to_image(flow_gt)], axis=0)

    cv2.imshow("image", img_flo_gt[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()


def compute_optical_flow_lk(images_path, flo_path):
    aEPEs = []
    aAEs = []
    MSEs = []

    images = sorted(
        glob.glob(os.path.join(images_path, "*.png"))
        + glob.glob(os.path.join(images_path, "*.jpg"))
    )

    flows = sorted(glob.glob(os.path.join(flo_path, "*.flo")))

    if len(images) < 2:
        raise ValueError("La séquence doit contenir au moins deux images.")

    optical_flows = []

    prev_img = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    for i in range(1, len(images)):
        next_img = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)

        flow = cv2.calcOpticalFlowFarneback(
            prev_img,
            next_img,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        flow_data = read_flo_file(flows[i - 1])
        flow_gt = flow_data.float().to(DEVICE)

        torch_flow = torch.from_numpy(flow).permute(2, 0, 1).float().to(DEVICE)

        if i == len(images) // 2:
            viz(load_image(images[i - 1]), torch_flow.unsqueeze(0), flow_gt)

        optical_flows.append(flow)

        aEPE, aAE, MSE = compute_metrics(
            load_image(images[i - 1]),
            load_image(images[i]),
            torch_flow,
            flow_gt,
        )
        aEPEs.append(aEPE)
        aAEs.append(aAE)
        MSEs.append(MSE)

        prev_img = next_img

    print("aEPE:", np.mean(aEPEs))
    print("aAE:", np.mean(aAEs))
    print("MSE:", np.mean(MSEs))

    # Ploting
    plt.figure(figsize=(15, 5))

    # Plot aEPE
    plt.subplot(1, 3, 1)
    plt.plot(aEPEs, label="aEPE", color="b")
    plt.legend()
    plt.title("Average End Point Error")

    # Plot aAE
    plt.subplot(1, 3, 2)
    plt.plot(aAEs, label="aAE", color="g")
    plt.legend()
    plt.title("Average Angular Error")

    # Plot MSE
    plt.subplot(1, 3, 3)
    plt.plot(MSEs, label="MSE", color="r")
    plt.legend()
    plt.title("Mean Square Error")

    plt.tight_layout()
    plt.show()

    return optical_flows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the image sequence", required=True)
    parser.add_argument("--path_flo", help="path to .flo files")
    args = parser.parse_args()

    compute_optical_flow_lk(args.path, args.path_flo)
