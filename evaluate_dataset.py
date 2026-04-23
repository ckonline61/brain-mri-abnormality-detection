"""
Batch evaluation utility for the Brain MRI autoencoder model.

Runs the saved model on the local yes/no dataset, computes anomaly metrics using
the same decision logic as the Flask app, and exports:
- per-image results CSV
- summary text file
- anomaly score distribution chart PNG
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.preprocessing import compute_anomaly_score, compute_reconstruction_error, load_image, skull_strip, threshold_anomaly

BINARY_MAP = {
    "no": "Normal",
    "yes": "Abnormal",
}

AUTOENCODER_THRESHOLD_SCORE = 0.04
AUTOENCODER_THRESHOLD_LOSS = 0.003
AUTOENCODER_THRESHOLD_CLUSTER_PCT = 3.75

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "autoencoder.h5"
DATASET_DIR = BASE_DIR / "Data_Set"
OUTPUT_DIR = BASE_DIR / "reports" / "evaluation"
CSV_PATH = OUTPUT_DIR / "dataset_inference_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "dataset_evaluation_summary.txt"
CHART_PATH = OUTPUT_DIR / "anomaly_score_distribution.png"


def summarize_anomaly_mask(mask: np.ndarray) -> dict[str, float]:
    mask_uint8 = mask.squeeze().astype(np.uint8)
    total_pixels = float(mask_uint8.size) if mask_uint8.size else 1.0
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats((mask_uint8 > 0).astype(np.uint8), 8)
    cluster_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
    largest_cluster = max(cluster_areas) if cluster_areas else 0
    largest_cluster_pct = round((largest_cluster / total_pixels) * 100, 2)
    return {
        "cluster_count": len(cluster_areas),
        "largest_cluster_area": largest_cluster,
        "largest_cluster_pct": largest_cluster_pct,
    }


def autoencoder_label_from_metrics(anomaly_score: float, recon_loss: float, mask: np.ndarray) -> tuple[str, float, dict[str, float]]:
    mask_summary = summarize_anomaly_mask(mask)
    is_abnormal = (
        anomaly_score >= AUTOENCODER_THRESHOLD_SCORE
        or recon_loss >= AUTOENCODER_THRESHOLD_LOSS
        or mask_summary["largest_cluster_pct"] >= AUTOENCODER_THRESHOLD_CLUSTER_PCT
    )
    label = "Abnormal" if is_abnormal else "Normal"
    confidence = min(
        0.99,
        max(
            anomaly_score / AUTOENCODER_THRESHOLD_SCORE,
            recon_loss / AUTOENCODER_THRESHOLD_LOSS,
            mask_summary["largest_cluster_pct"] / AUTOENCODER_THRESHOLD_CLUSTER_PCT,
        ),
    )
    if not is_abnormal:
        confidence = max(0.55, 1.0 - confidence * 0.5)
    return label, float(confidence), mask_summary


def iter_dataset_images(dataset_dir: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for folder_name in ("no", "yes"):
        folder = dataset_dir / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".nii", ".gz"}:
                items.append((folder_name, path))
    return items


def build_chart(df: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.5))

    no_scores = df.loc[df["actual_folder"] == "no", "anomaly_score"]
    yes_scores = df.loc[df["actual_folder"] == "yes", "anomaly_score"]

    bins = np.linspace(df["anomaly_score"].min(), df["anomaly_score"].max(), 24)
    if len(np.unique(bins)) < 2:
        bins = 20

    ax.hist(no_scores, bins=bins, alpha=0.68, color="#2D6A4F", label="Normal MRI (no)", density=True)
    ax.hist(yes_scores, bins=bins, alpha=0.62, color="#BC4749", label="Abnormal MRI (yes)", density=True)

    ax.axvline(AUTOENCODER_THRESHOLD_SCORE, color="#1D3557", linestyle="--", linewidth=2, label="Score threshold")
    ax.set_title("Anomaly Score Distribution on Brain MRI Dataset", fontsize=16, weight="bold")
    ax.set_xlabel("Anomaly Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(frameon=True)

    no_mean = no_scores.mean()
    yes_mean = yes_scores.mean()
    ax.text(
        0.99,
        0.95,
        f"Normal mean: {no_mean:.4f}\nAbnormal mean: {yes_mean:.4f}\nThreshold: {AUTOENCODER_THRESHOLD_SCORE:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#9AA0A6"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame, out_path: Path) -> None:
    total = len(df)
    correct = int((df["actual_label"] == df["predicted_label"]).sum())
    accuracy = correct / total if total else 0.0

    tp = int(((df["actual_label"] == "Abnormal") & (df["predicted_label"] == "Abnormal")).sum())
    tn = int(((df["actual_label"] == "Normal") & (df["predicted_label"] == "Normal")).sum())
    fp = int(((df["actual_label"] == "Normal") & (df["predicted_label"] == "Abnormal")).sum())
    fn = int(((df["actual_label"] == "Abnormal") & (df["predicted_label"] == "Normal")).sum())

    abnormal_precision = tp / (tp + fp) if (tp + fp) else 0.0
    abnormal_recall = tp / (tp + fn) if (tp + fn) else 0.0
    abnormal_f1 = (
        2 * abnormal_precision * abnormal_recall / (abnormal_precision + abnormal_recall)
        if (abnormal_precision + abnormal_recall)
        else 0.0
    )

    lines = [
        "Brain MRI Dataset Evaluation Summary",
        "=" * 40,
        f"Total images: {total}",
        f"Normal images (no): {(df['actual_folder'] == 'no').sum()}",
        f"Abnormal images (yes): {(df['actual_folder'] == 'yes').sum()}",
        "",
        f"Accuracy: {accuracy:.4f} ({correct}/{total})",
        f"Abnormal precision: {abnormal_precision:.4f}",
        f"Abnormal recall: {abnormal_recall:.4f}",
        f"Abnormal F1-score: {abnormal_f1:.4f}",
        "",
        "Confusion matrix",
        f"TN: {tn}",
        f"FP: {fp}",
        f"FN: {fn}",
        f"TP: {tp}",
        "",
        "Anomaly score statistics",
        f"Normal mean score: {df.loc[df['actual_folder'] == 'no', 'anomaly_score'].mean():.6f}",
        f"Abnormal mean score: {df.loc[df['actual_folder'] == 'yes', 'anomaly_score'].mean():.6f}",
        f"Overall mean score: {df['anomaly_score'].mean():.6f}",
        f"Decision threshold (score): {AUTOENCODER_THRESHOLD_SCORE:.6f}",
        f"Decision threshold (loss): {AUTOENCODER_THRESHOLD_LOSS:.6f}",
        f"Decision threshold (largest cluster %): {AUTOENCODER_THRESHOLD_CLUSTER_PCT:.2f}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    input_shape = model.input_shape
    target_size = (int(input_shape[1]), int(input_shape[2]))

    rows = []
    for actual_folder, image_path in iter_dataset_images(DATASET_DIR):
        img = load_image(str(image_path), target_size=target_size)
        img = skull_strip(img)
        img_array = img[..., np.newaxis]
        recon = model.predict(img_array[np.newaxis, ...], verbose=0)[0]
        error = compute_reconstruction_error(img_array, recon)
        anomaly_score = compute_anomaly_score(error)
        mask, _ = threshold_anomaly(error)
        recon_loss = float(np.mean(error))
        predicted_label, confidence, mask_summary = autoencoder_label_from_metrics(anomaly_score, recon_loss, mask)

        rows.append(
            {
                "filename": image_path.name,
                "actual_folder": actual_folder,
                "actual_label": BINARY_MAP[actual_folder],
                "predicted_label": predicted_label,
                "is_correct": BINARY_MAP[actual_folder] == predicted_label,
                "confidence": confidence,
                "anomaly_score": anomaly_score,
                "reconstruction_loss": recon_loss,
                "largest_cluster_pct": mask_summary["largest_cluster_pct"],
                "cluster_count": mask_summary["cluster_count"],
                "largest_cluster_area": mask_summary["largest_cluster_area"],
                "path": str(image_path),
            }
        )

    df = pd.DataFrame(rows).sort_values(["actual_folder", "filename"]).reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    write_summary(df, SUMMARY_PATH)
    build_chart(df, CHART_PATH)

    total = len(df)
    correct = int(df["is_correct"].sum())
    print(f"Model evaluated on {total} images.")
    print(f"Correct predictions: {correct}/{total} ({(correct / total):.2%})")
    print(f"CSV saved to: {CSV_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Chart saved to: {CHART_PATH}")


if __name__ == "__main__":
    main()
