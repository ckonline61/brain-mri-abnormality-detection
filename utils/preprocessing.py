"""
Preprocessing utilities for Brain MRI images.
Handles MRI image loading, normalization, skull stripping, reconstruction error,
and visualization support for binary Normal/Abnormal detection.
"""

import numpy as np
import cv2
from PIL import Image
import os

TARGET_SIZE = (224, 224)

# ─── Image Loading ─────────────────────────────────────────────────────────────

def load_image(path, target_size=TARGET_SIZE, grayscale=True):
    """Load image from disk and resize/normalize to [0,1]."""
    if path.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        vol = nib.load(path).get_fdata()
        mid_slice = vol[:, :, vol.shape[2] // 2]
        img = mid_slice.astype(np.float32)
    else:
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(path, flag)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")

    img = cv2.resize(img, target_size)
    img = img.astype(np.float32)

    # z-score normalization
    mean, std = img.mean(), img.std()
    img = (img - mean) / (std + 1e-8)

    # scale to [0,1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def load_image_from_bytes(file_bytes, target_size=TARGET_SIZE):
    """Load image from in-memory bytes (Flask upload)."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image bytes")
    img = cv2.resize(img, target_size).astype(np.float32)
    mean, std = img.mean(), img.std()
    img = (img - mean) / (std + 1e-8)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


# ─── Skull Stripping (simplified) ────────────────────────────────────────────

def skull_strip(img):
    """
    Simplified skull-stripping via Otsu thresholding + morphological ops.
    In production use FSL BET or BrainSuite.
    """
    img_uint8 = (img * 255).astype(np.uint8)
    _, mask = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    result = img.copy()
    result[mask == 0] = 0
    return result


# ─── Data Augmentation ────────────────────────────────────────────────────────

def augment(img, seed=None):
    """Apply random flips and rotations."""
    rng = np.random.default_rng(seed)
    if rng.random() > 0.5:
        img = np.fliplr(img)
    if rng.random() > 0.5:
        img = np.flipud(img)
    angle = rng.uniform(-15, 15)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))
    return img


def load_dataset(folder, augment_data=False, apply_skull_strip=True, target_size=TARGET_SIZE):
    """Load all images from a folder into a numpy array."""
    paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.nii', '.nii.gz'))]
    images = []
    for p in paths:
        try:
            img = load_image(p, target_size=target_size)
            if apply_skull_strip:
                img = skull_strip(img)
            images.append(img)
            if augment_data:
                images.append(augment(img))
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
    arr = np.array(images, dtype=np.float32)
    return arr[..., np.newaxis]   # add channel dim → (N, 128, 128, 1)


# ─── Anomaly Detection ────────────────────────────────────────────────────────

def compute_reconstruction_error(original, reconstructed):
    """Per-pixel squared error map."""
    return (original - reconstructed) ** 2


def generate_heatmap(error_map, colormap=cv2.COLORMAP_JET):
    """Convert error map to colour heatmap (uint8 RGB)."""
    err = error_map.squeeze()
    err = ((err - err.min()) / (err.max() - err.min() + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(err, colormap)
    return heatmap


def threshold_anomaly(error_map, percentile=95):
    """Binary mask: pixels with error above percentile threshold."""
    threshold = np.percentile(error_map, percentile)
    mask = (error_map > threshold).astype(np.uint8) * 255
    return mask, threshold


def overlay_heatmap(original_img, heatmap, alpha=0.5):
    """Overlay heatmap on original grayscale image."""
    orig_rgb = cv2.cvtColor((original_img.squeeze() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(orig_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay


def compute_anomaly_score(error_map):
    """Scalar anomaly score: mean of top-5% pixel errors."""
    threshold = np.percentile(error_map, 95)
    top_errors = error_map[error_map >= threshold]
    return float(np.mean(top_errors))
