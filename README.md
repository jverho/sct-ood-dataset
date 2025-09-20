# Medical Image Data Preprocessing for Anomaly Detection (Synthrad 2023 Pelvis)

## 📝 Scripts Overview
### label_generator.py
Process anomaly labels from text and Excel files and generates structured JSON files. It extracts information such as anomaly type and the range of abnormal slices for both pelvis and brain scans, although the main preprocessing script focuses on pelvis data.

### data_preprocessing.py
Main driver script for data preparation, main steps:
1. **Dataset Splitting**: Divides the scans into train, valid, and test sets. The valid and test sets include a mix of both normal (good) and abnormal (Ungood) scans.
2. **Image Processing**:
   - Loads NIfTI (.nii.gz) CT and MR scan volumes.
   - Applies a body mask to the MR images.
   - Extracts 2D slices from the 3D volumes.
   - For normal scans: Slices are resized and saved to the good directories with empty label masks.
   - For abnormal scans: Anomaly masks are generated using the MetalArtifactDetector class based on HU (Hounsfield Unit) thresholds, refined with MR data, and post-processed. The slices and their corresponding masks are saved to the Ungood directories.
3. **Saves Output**: The processed 2D slices and masks are saved as NIfTI files (.nii) in a structured output directory.

### artifact_detector.py
A class that provides methods for segmenting metal artifacts and other anomalies in CT and MR scans. Its main functions include:
- **HU-based Thresholding**: Identifies potential artifacts by their high Hounsfield Unit values.
- **Mask Refinement**: Refines the initial CT masks using a flood-fill algorithm on the MR image to improve segmentation accuracy.
- **Morphological Operations**: Applies conditional morphological opening to clean up small, noisy regions in the masks.

### utils/
- **path_utils.py**: Contains a simple function to ensure the necessary output directory structure is created before saving files.
- **processing_utils.py**: A collection of helper functions for loading/saving NIfTI images, applying masks, normalizing pixel values, and performing image resizing and padding

## 🚀 Usage
To run the data preprocessing pipeline, execute the main script:

```bash
python data_preprocessing.py
```

Note: The correct JSON files need to exist in the label folder beforehand. Additionally, ensure all required libraries are installed and the data paths (DIR_PELVIS, DIR_LABELS, EXCEL_OVERVIEW_PELVIS) specified in the scripts are correct for your server environment.


