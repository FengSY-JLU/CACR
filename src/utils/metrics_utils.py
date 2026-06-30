# src/utils/metrics_utils.py

import numpy as np
import torch
import cv2
from scipy.stats import entropy as scipy_entropy


def to_numpy_image(img_tensor: torch.Tensor) -> np.ndarray:
    img_np = img_tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (img_np * 255).astype(np.uint8)


def calculate_entropy(image_tensor: torch.Tensor) -> float:
    if image_tensor.ndim == 3 and image_tensor.shape[0] == 3:
        gray = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
    else:
        gray = image_tensor[0]
    gray_np = (gray.cpu().numpy() * 255).astype(np.uint8)
    hist, _ = np.histogram(gray_np, bins=256, range=(0, 255), density=True)
    return scipy_entropy(hist + 1e-8)


def rgb2lab_n(img: np.ndarray):
    img = img.astype(np.float64) / 255.0 if img.max() > 1.0 else img.astype(np.float64)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    def srgb_to_linear(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)

    x = 0.4124 * r_lin + 0.3576 * g_lin + 0.1805 * b_lin
    y = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    z = 0.0193 * r_lin + 0.1192 * g_lin + 0.9505 * b_lin

    x /= 0.9505
    y /= 1.0
    z /= 1.0891

    def f(t):
        return np.where(t > 0.008856, t ** (1. / 3), 7.787 * t + 16. / 116)

    fx, fy, fz = f(x), f(y), f(z)
    L = np.where(y > 0.008856, 116 * (y ** (1. / 3)) - 16, 903.3 * y)
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b


def calculate_uciqe(img_tensor: torch.Tensor) -> float:
    img = to_numpy_image(img_tensor)
    img = img.astype(np.float64)

    L, a_lab, b_lab = rgb2lab_n(img)
    chroma = np.sqrt(a_lab ** 2 + b_lab ** 2)
    std_chroma = np.std(chroma.reshape(-1))

    hsv = cv2.cvtColor(img.astype(np.float32) / 255., cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    mean_saturation = np.mean(saturation.reshape(-1))

    contrast_luminance = np.max(L) - np.min(L)
    uciqe = 0.4680 * std_chroma + 0.2745 * contrast_luminance + 0.2576 * mean_saturation
    return round(uciqe, 4)


def UICM(img):
    img = img.astype(np.float64)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    RG = R - G
    YB = (R + G) / 2 - B

    RG1 = RG.flatten()
    RG1.sort()
    K = len(RG1)
    alphaL = 0.1
    alphaR = 0.1
    RG1 = RG1[int(alphaL * K):int(K * (1 - alphaR))]
    meanRG = np.mean(RG1)
    deltaRG = np.std(RG1)

    YB1 = YB.flatten()
    YB1.sort()
    YB1 = YB1[int(alphaL * K):int(K * (1 - alphaR))]
    meanYB = np.mean(YB1)
    deltaYB = np.std(YB1)

    return -0.0268 * np.sqrt(meanRG ** 2 + meanYB ** 2) + 0.1586 * np.sqrt(deltaRG ** 2 + deltaYB ** 2)


def UIConM(img, patch_size=5):
    img = img.astype(np.float64)
    h, w, _ = img.shape
    h_pad = patch_size - h % patch_size if h % patch_size != 0 else 0
    w_pad = patch_size - w % patch_size if w % patch_size != 0 else 0
    img = cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REPLICATE)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    def local_contrast(channel):
        h, w = channel.shape
        contrast_sum = 0.0
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = channel[i:i + patch_size, j:j + patch_size]
                patch_max = np.max(patch)
                patch_min = np.min(patch)
                if patch_max != patch_min:
                    contrast = np.log((patch_max - patch_min) / (patch_max + patch_min))
                    contrast_sum += contrast * (patch_max - patch_min) / (patch_max + patch_min)
        return contrast_sum / ((h // patch_size) * (w // patch_size))

    return abs(local_contrast(R)) + abs(local_contrast(G)) + abs(local_contrast(B))


def UISM(img, patch_size=5):
    img = img.astype(np.float64)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    sobel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    def eme(channel):
        edge_x = cv2.filter2D(channel, -1, sobel_x)
        edge_y = cv2.filter2D(channel, -1, sobel_y)
        sobel = np.abs(edge_x + edge_y)
        h, w = sobel.shape
        h_pad = patch_size - h % patch_size if h % patch_size != 0 else 0
        w_pad = patch_size - w % patch_size if w % patch_size != 0 else 0
        sobel = cv2.copyMakeBorder(sobel, 0, h_pad, 0, w_pad, cv2.BORDER_REPLICATE)
        h, w = sobel.shape

        eme_val = 0.0
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = sobel[i:i + patch_size, j:j + patch_size]
                patch_max = np.max(patch)
                patch_min = np.min(patch)
                if patch_max > 0 and patch_min > 0:
                    eme_val += np.log(patch_max / patch_min)
        return 2.0 * eme_val / ((h // patch_size) * (w // patch_size))

    lambdaR, lambdaG, lambdaB = 0.299, 0.587, 0.114
    return lambdaR * eme(R) + lambdaG * eme(G) + lambdaB * eme(B)


def calculate_uiqm(img_tensor: torch.Tensor) -> float:
    img = to_numpy_image(img_tensor).astype(np.float64)
    return round(0.0282 * UICM(img) + 0.2953 * UISM(img) + 3.5753 * UIConM(img), 4)
