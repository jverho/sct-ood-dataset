# Medical Image Data Preprocessing for Anomaly Detection (Synthrad 2023 Pelvis)

## 📝 Scripts Overview
The data preprocessing pipeline is now modularized into four main driver scripts, each tailored to a specific output format and processing flow, and two utility scripts (`label_generator.py` and `artifact_detector.py`) used by the drivers.

---

### label_generator.py
Processes raw anomaly labels from text and Excel files and generates structured **JSON** files (e.g., `labels_implant.json`, `labels_others.json`). It extracts key information like the anomaly type and the start/end slice range of abnormalities for each scan ID.

---

### artifact_detector.py
A class that provides methods for segmenting metal artifacts and other anomalies in CT and MR scans. Its main functions include:
- **HU-based Thresholding**: Identifies potential artifacts by their high Hounsfield Unit (HU) values in CT scans.
- **Mask Refinement**: Refines the initial CT masks using a flood-fill algorithm on the MR image to improve segmentation accuracy.
- **Morphological Operations**: Applies conditional morphological opening to clean up small, noisy regions in the final masks.

---

### Main Dataset Processing Scripts
These four scripts handle the entire data preparation pipeline: dataset splitting, 3D volume loading, slice extraction, anomaly mask generation (for abnormal scans), padding, resizing, and saving the final 2D images and masks.

| Script Name | Output Format | Preprocessing | Output Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| **dataset_processing_nifti.py** | **NIfTI (.nii)** | Pad + Resize | **(240, 240)** | Extracts 2D slices, **center-pads and resizes to (240, 240)**, and saves as NIfTI. |
| **dataset_processing_png.py** | **PNG (.png)** | Pad + Resize | **(240, 240)** | Extracts 2D slices, **center-pads and resizes to (240, 240)**, and saves as PNG. |
| **dataset_processing_resize_nifti.py** | **NIfTI (.nii)** | Pad + Resize + Crop | **(224, 224)** | Performs padding/resizing to (240, 240), then a final **center-crop to (224, 224)**. Saves as NIfTI. |
| **dataset_processing_resize_png.py** | **PNG (.png)** | Pad + Resize + Crop | **(224, 224)** | Performs padding/resizing to (240, 240), then a final **center-crop to (224, 224)**. Saves as PNG. |

#### **Common Pipeline Steps (in all drivers):**
1. **Dataset Splitting**: Divides the scan IDs into **train, valid, and test** sets. The valid and test sets include a mix of both normal (**good**) and abnormal (**Ungood**) scans.
2. **Image Processing**:
    - Loads NIfTI (.nii.gz) CT and MR scan volumes.
    - Applies a body mask to the MR images and performs min-max normalization.
    - Extracts **3-channel 2D slices** (previous, current, next slice) from the 3D volume (or a single slice for PNG/non-resize NIfTI variants).
    - **Center-pads** all slices to a square shape, and **resizes** to the target base size of **(240, 240)**.
3. **Anomaly Masking**: For **Ungood** scans, anomaly masks are generated using the `MetalArtifactDetector` based on HU thresholds, refined with MR data, and post-processed.
4. **Saves Output**: The processed 2D slices and masks are saved in the chosen format within a structured output directory (`img`, `label`, and `bodymask` subfolders for validation/test sets).

---

## 🚀 Usage
To run the data preprocessing pipeline, execute one of the main driver scripts. For example, to generate the recommended **resized and cropped NIfTI dataset** for training **(224x224)**:

```bash
python dataset_processing_resize_nifti.py
