# College Submission Checklist

This project is prepared for final college submission as a Brain MRI Abnormality Detection System based on an unsupervised Convolutional Autoencoder.

## What to Submit

- Source code folder: `brain_mri_project`
- Model file: `models/autoencoder.h5`
- Dataset folder: `Data_Set`
- Evaluation charts and result assets: `reports/evaluation`
- Project report / PDF documents as required by the college

## How to Run for Demonstration

1. Open terminal in `brain_mri_project`
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the application:

```bash
python app.py
```

4. Open the browser:

```text
http://127.0.0.1:5000
```

## Default Login Accounts

- Admin: `admin / admin123`
- Doctor: `doctor / doctor123`

## Project Highlights

- Brain MRI abnormality detection using a Convolutional Autoencoder
- Binary output: `Normal` or `Abnormal`
- Reconstruction error based anomaly analysis
- Heatmap, mask, and overlay visualization
- PDF report generation
- Role-based login system
- SQLite-based analysis history
- Encrypted backup support

## Important Submission Notes

- The `Data_Set` folder has not been modified during final submission preparation.
- The model architecture used in the project is an unsupervised Convolutional Autoencoder.
- The project includes evaluation charts for report and viva presentation support.
- This project is intended for academic demonstration and not for clinical diagnosis.
