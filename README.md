# Brain MRI Abnormality Detection System

A Flask-based web application for binary Brain MRI abnormality detection using an unsupervised convolutional autoencoder. The system accepts MRI image uploads, performs preprocessing, generates anomaly visualizations, stores analysis history, and exports PDF reports.

## Features

- Binary detection output: `Normal` or `Abnormal`
- MRI preprocessing with grayscale conversion, resizing, normalization, and skull stripping
- Autoencoder-based reconstruction error analysis
- Heatmap, overlay, and anomaly mask generation
- User roles: `Admin`, `Doctor`, `Patient`
- Analysis history and dashboard statistics
- PDF report generation
- Encrypted backup support and audit logging

## Project Structure

```text
brain_mri_project/
├── app.py
├── model.py
├── requirements.txt
├── models/
├── static/
├── templates/
└── utils/
```

## Requirements

- Python 3.10+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

This project supports the following optional environment variables:

- `SECRET_KEY`: Flask session secret key
- `BRAIN_MRI_ENCRYPTION_KEY`: encryption key for protected fields and backups

Example:

```bash
set SECRET_KEY=replace-with-a-strong-random-secret
set BRAIN_MRI_ENCRYPTION_KEY=your-generated-fernet-key
```

On first run, the application will create a fresh SQLite database and local encryption key automatically if they are not already present.

## Run Locally

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Default Users

The application seeds default users on first run:

- `admin / admin123`
- `doctor / doctor123`

Change these credentials after first login if you plan to use the app beyond local academic/demo use.

## Notes for GitHub

- Local database files, encrypted keys, backups, and generated analysis images are excluded through `.gitignore`
- This repository is prepared to share source code safely without personal runtime data
- If you want to publish the trained model, keep `models/autoencoder.h5` in the repository

## Disclaimer

This project is an academic implementation for Brain MRI abnormality detection. It is not a certified medical diagnosis system, and final interpretation should be confirmed by a qualified doctor or radiologist.
