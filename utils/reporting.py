"""
PDF report generation helpers for MRI analysis reports.
"""

import io
import os
from datetime import datetime

import cv2
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

HOSPITAL_NAME = os.environ.get("BRAIN_MRI_HOSPITAL_NAME", "BrainScan AI Diagnostic Center")


def analyze_mask(mask_path):
    if not mask_path or not os.path.exists(mask_path):
        return {
            "cluster_count": 0,
            "largest_cluster_area": 0,
            "largest_cluster_pct": 0.0,
            "summary": "No anomaly mask available for technical summary.",
        }

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {
            "cluster_count": 0,
            "largest_cluster_area": 0,
            "largest_cluster_pct": 0.0,
            "summary": "Mask image could not be read.",
        }

    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clusters = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > 0:
            clusters.append(area)

    total_pixels = int(binary.shape[0] * binary.shape[1]) if binary.size else 0
    largest_area = max(clusters) if clusters else 0
    largest_pct = round((largest_area / total_pixels * 100) if total_pixels else 0, 2)

    if clusters:
        summary = (
            f"The autoencoder-based detection pipeline highlighted {len(clusters)} anomaly cluster(s). "
            f"The largest highlighted region covers about {largest_area} pixels "
            f"(~{largest_pct}% of the processed MRI slice)."
        )
    else:
        summary = "The autoencoder-based detection pipeline did not find any significant abnormal region in the generated mask."

    return {
        "cluster_count": len(clusters),
        "largest_cluster_area": largest_area,
        "largest_cluster_pct": largest_pct,
        "summary": summary,
    }


def _draw_image_pair(pdf, original_path, abnormality_path, y_top):
    image_width = 220
    image_height = 180
    left_x = 50
    right_x = 325

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(left_x, y_top + 10, "Original MRI")
    pdf.drawString(right_x, y_top + 10, "Detected Region Overlay")

    if original_path and os.path.exists(original_path):
        pdf.drawImage(
            ImageReader(original_path),
            left_x,
            y_top - image_height,
            width=image_width,
            height=image_height,
            preserveAspectRatio=True,
            mask="auto",
        )
    else:
        pdf.rect(left_x, y_top - image_height, image_width, image_height)
        pdf.drawString(left_x + 20, y_top - 80, "Original image unavailable")

    if abnormality_path and os.path.exists(abnormality_path):
        pdf.drawImage(
            ImageReader(abnormality_path),
            right_x,
            y_top - image_height,
            width=image_width,
            height=image_height,
            preserveAspectRatio=True,
            mask="auto",
        )
    else:
        pdf.rect(right_x, y_top - image_height, image_width, image_height)
        pdf.drawString(right_x + 20, y_top - 80, "Detected region image unavailable")

    return y_top - image_height - 25


def build_pdf_report(analysis, original_path, abnormality_path, mask_path):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setTitle(f"MRI_Report_{analysis['patient_code']}_{analysis['id']}")

    pdf.setFillColor(colors.HexColor("#1B474D"))
    pdf.rect(0, height - 70, width, 70, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(40, height - 42, HOSPITAL_NAME)
    pdf.setFont("Helvetica", 10)
    pdf.drawRightString(width - 40, height - 42, datetime.now().strftime("%d %b %Y"))

    y = height - 100
    pdf.setFillColor(colors.HexColor("#20808D"))
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(40, y, "Patient Information")

    y -= 22
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Name: {analysis['patient_name']}")
    pdf.drawString(300, y, f"Patient ID: {analysis.get('patient_code') or 'N/A'}")
    y -= 18
    pdf.drawString(50, y, f"Age: {analysis.get('patient_age') if analysis.get('patient_age') is not None else 'N/A'}")
    pdf.drawString(300, y, f"Gender: {analysis.get('patient_gender') or 'Unknown'}")

    y -= 34
    pdf.setFillColor(colors.HexColor("#20808D"))
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(40, y, "Detection Section")

    y -= 26
    y = _draw_image_pair(pdf, original_path, abnormality_path, y)

    technical = analyze_mask(mask_path)
    pdf.setFillColor(colors.HexColor("#20808D"))
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(40, y, "Technical Summary")
    y -= 22

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 11)
    summary_lines = [
        f"Detection Result: {analysis['result_label']}",
        f"Confidence Score: {round((analysis.get('confidence') or 0) * 100, 2)}%",
        f"Detected Cluster Count: {technical['cluster_count']}",
        f"Approx. Largest Highlighted Region: {technical['largest_cluster_area']} pixels ({technical['largest_cluster_pct']}% of slice)",
        technical["summary"],
    ]

    for line in summary_lines:
        text = pdf.beginText(50, y)
        text.textLines(line)
        pdf.drawText(text)
        y -= 18

    y -= 25
    pdf.line(width - 210, y, width - 60, y)
    pdf.setFont("Helvetica", 11)
    pdf.drawString(width - 185, y - 16, "Doctor Signature")

    pdf.setFont("Helvetica-Oblique", 9)
    pdf.setFillColor(colors.grey)
    pdf.drawString(
        40,
        30,
        "Autoencoder-based MRI abnormality detection report for clinical review. Final interpretation must be confirmed by a qualified doctor/radiologist.",
    )

    pdf.save()
    buffer.seek(0)
    return buffer
