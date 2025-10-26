# Medical Image Data Preprocessing for Anomaly Detection (Synthrad 2023 Pelvis)

This repository contains the data preprocessing pipeline for preparing the **Synthrad 2023 Pelvis** dataset for use in anomaly detection models. The pipeline extracts and processes 2D slices from 3D CT/MR volumes, normalizes them, generates anomaly masks, and saves the output in structured train/validation/test splits.

---

## 📝 Scripts Overview
The data preprocessing pipeline is now modularized into **three main driver scripts**, each tailored to a specific output format and channel configuration, and two utility scripts (`label_generator.py` and `artifact_detector.py`) used by the drivers.

---

### Utility Scripts

| Script Name | Description |
| :--- | :--- |
| **label\_generator.py** | Processes raw anomaly labels from text and Excel and generates structured **JSON** files (e.g., `labels_implant.json`). Extracts the anomaly type and the start/end slice range of abnormalities for each scan ID. |
| **artifact\_detector.py** | A class providing methods for segmenting metal artifacts and other anomalies in CT and MR scans. Functions include **HU-based Thresholding**, **Mask Refinement** (using MR data and flood-fill), and **Morphological Operations** for cleanup. |

---

### Main Dataset Processing Scripts
These three scripts handle the entire data preparation pipeline: dataset splitting, 3D volume loading, slice extraction, anomaly mask generation (for abnormal scans), padding, resizing, and saving the final 2D images and masks.

| Script Name | Output Format | Input Slice Channels | Output Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| **dataset\_processing\_png.py** | **PNG (.png)** | Current Slice (Replicated to 3 Channels) | **(224, 224)** | Extracts the **current 2D slice**, color-maps it using **'bone'**, and saves it as a 3-channel PNG file. |
| **dataset\_processing\_nifti\_rep.py** | **NIfTI (.nii)** | Current Slice (Replicated to 3 Channels) | **(224, 224)** | Extracts the **current 2D slice** and replicates it across 3 channels, saving the result as a 3-channel NIfTI file. |
| **dataset\_processing\_nifti.py** | **NIfTI (.nii)** | 3 Consecutive Slices (Previous, Current, Next) | **(224, 224)** | Extracts **three consecutive 2D slices** (previous, current, next) to form a 3-channel NIfTI volume, capturing inter-slice context. |

#### **Common Pipeline Steps (in all drivers):**
1.  **Dataset Splitting**: Divides the scan IDs into **train, valid, and test** sets. The valid and test sets include a mix of both normal (**good**) and abnormal (**Ungood**) scans.
2.  **Image Processing**:
    - Loads NIfTI (.nii.gz) CT and MR scan volumes.
    - Applies a body mask to the MR images and performs min-max normalization.
    - Extracts **3-channel 2D slices** based on the script's configuration.
    - **Center-pads** all slices to a square shape, **resizes** to the target base size of **(240, 240)**, and then performs a final **center-crop to (224, 224)**.
3.  **Anomaly Masking**: For **Ungood** scans, anomaly masks are generated using the `MetalArtifactDetector` based on HU thresholds, refined with MR data, and post-processed.
4.  **Saves Output**: The processed 2D slices and masks are saved in the chosen format within a structured output directory (`img`, `label`, and `bodymask` subfolders for validation/test sets).

---

## 🚀 Usage

To run the data preprocessing pipeline, execute one of the main driver scripts. For example, to generate the NIfTI dataset using 3 consecutive slices **(224x224)**:

```bash
python dataset_processing_nifti.py
