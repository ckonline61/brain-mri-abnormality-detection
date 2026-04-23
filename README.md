# Brain MRI Abnormality Detection System

This project is a Flask-based web application for binary brain MRI abnormality detection using an unsupervised Convolutional Autoencoder. The system accepts MRI image uploads, preprocesses the scan, reconstructs it through the trained model, computes reconstruction error, and produces a final `Normal` or `Abnormal` result with supporting visual evidence.

## Core Features

- Binary MRI analysis result: `Normal` or `Abnormal`
- Unsupervised Convolutional Autoencoder based detection
- MRI preprocessing with grayscale conversion, resizing, normalization, and skull stripping
- Reconstruction error map, anomaly score, heatmap, mask, and overlay generation
- Secure Flask web application with login and role-based access
- User roles: `Admin`, `Doctor`, and `Patient`
- Analysis history, dashboard metrics, and downloadable PDF reports
- Encrypted backup support and audit logging

## Technology Stack

- Python
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy
- ReportLab
- SQLite

## Project Structure

```text
brain_mri_project/
|-- app.py
|-- model.py
|-- evaluate_dataset.py
|-- requirements.txt
|-- models/
|-- static/
|-- templates/
|-- utils/
|-- reports/
`-- Data_Set/
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Optional environment variables:

- `SECRET_KEY`: Flask session secret key
- `BRAIN_MRI_ENCRYPTION_KEY`: encryption key for protected fields and backups

Example on Windows:

```bash
set SECRET_KEY=replace-with-a-strong-random-secret
set BRAIN_MRI_ENCRYPTION_KEY=your-generated-fernet-key
```

If these are not provided, the project can still run locally. On first run, it creates the local database and encryption key automatically if they do not already exist.

## How to Run

From the `brain_mri_project` folder:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

You can also start the project from the root-level batch file:

```bash
run_brain_mri_project.bat
```

## Default Users

The application seeds default users on first run:

- `admin / admin123`
- `doctor / doctor123`

These are intended only for local academic/demo use.

## Evaluation Assets

The `reports/evaluation/` folder contains submission-ready analysis outputs such as:

- confusion matrix
- ROC curve
- precision-recall curve
- threshold sensitivity chart
- reconstruction loss chart
- anomaly score distribution chart
- sample result comparison
- architecture diagram

## Notes

- The trained model file is stored at `models/autoencoder.h5`.
- Runtime-generated uploads, result images, backups, and local database files are excluded through `.gitignore`.
- The dataset folder is included separately for local project use.

## Disclaimer

This project is an academic implementation for brain MRI abnormality detection. It is not a certified medical diagnosis system, and all results should be reviewed by a qualified doctor or radiologist.
