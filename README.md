# sct-ood-dataset

## Labeling
### HU>2000
Using HU as a threshold is not ideal for all cases, as it may not account for variations in tissue density and composition.

Refer [notebook2](notebooks/2_ct_HU_stats_pelvis.ipynb) for following visualizations.

![HU Value Distribution](images/2_hu_value_distribution_example_cts.png)
![HU > 2000 Slice Ratios](images/2_hu_gt2000_slice_ratios.png)
![CT HU EDA](images/2_ct_hu_eda.png)

Refer [notebook3](notebooks/3_ct_HU_label_coverage_analysis.ipynb) for following visualizations.

![HU > 2000 Coverage per OOD Scan](images/3_hu2000_coverage_per_ood_scan.png)
![HU > 2000 Distribution Coverage](images/3_hu2000_distribution_coverage.png)
![HU > 2000 Scan Level Confusion Matrix](images/3_hu2000_scan_level_confusion_matrix.png)

### Per Scan Thresholding

Using max HU value from all other normal slices as global threshold gives good performance on slice-level segmentation. (Example taken from Justin's annotations)

Refer [notebook4](notebooks/4_ct_scan_visualization.ipynb) for following visualizations.
![CT Example](images/4_1PA133_ct_example.png)
![HU Threshold Example](images/4_1PA133_hu_thresh_example.png)

![CT Example](images/4_1PA136_ct_example.png)
![HU Threshold Example](images/4_1PA136_hu_thresh_example.png)

![CT Example](images/4_1PA147_ct_example.png)
![HU Threshold Example](images/4_1PA147_hu_thresh_example.png)

![CT Example](images/4_1PA151_ct_example.png)
![HU Threshold Example](images/4_1PA151_hu_thresh_example.png)

![CT Example](images/4_1PA152_ct_example.png)
![HU Threshold Example](images/4_1PA152_hu_thresh_example.png)

![CT Example](images/4_1PA169_ct_example.png)
![HU Threshold Example](images/4_1PA169_hu_thresh_example.png)

We can even use the plots to refine our labeling process because sometimes it's difficult to decide the boundary between slices.




## BMAD Brats2021 Data Preprocessing Logic
```
Brain/
├── train/
│   └── good/   # 7500 slices
├── valid/
│   ├── good/
│   │   ├── img/   # 39 slices
│   │   └── label/
│   └── Ungood/
│       ├── img/    # 44 slices
│       └── label/
└── test/
    ├── good/
    │   ├── img/    # 640 slices
    │   └── label/
    └── Ungood/
        ├── img/    # 3075 slices
        └── label/
```

### **Data Partitioning**
The script divides the dataset into several **non-overlapping** groups:

- Train (normal only): 424 IDs
- Valid (normal): 39 IDs
- Valid (abnormal): 11 IDs
- Test (normal): 162 IDs
- Test (abnormal): 615 IDs
- *Based on the ratio of good slices / #ids, author might have intentionally selected ids with high proportion of good slices from BraTs dataset into this category.*

### **Data Preprocessing (BMAD)**

Process slices from index 60 to 100 out of 155 (middle slices where brain is most visible)
- **Majority slices in this dataset between 60 to 100 contains tumor.**
- For **training data** (normal cases only):
  - Only saves slices without tumors (where `seg` max value < 1)
  - Saves only the FLAIR images (no labels)
  - Output directory: `/Brain/train/good/`

- For **validation data**:
  - Normal cases: Similar to training but saves both images and labels
  - Abnormal cases: Subsample: Saves every 10th slice (60, 70, 80, 90) including tumor cases 
  - Output directories: `/Brain/valid/good/` and `/Brain/valid/Ungood/`

- For **test data**:
  - Normal cases: Similar to validation normal cases
  - Abnormal cases: Saves every 8th slice (60, 68, 76, 84, 92) including tumor cases
  - Output directories: `/Brain/test/good/` and `/Brain/test/Ungood/`



- Uses bone colormap (`cmap="bone"`) for medical image visualization

## Data Processing for SynthRAD23 pelvis
### Metal Artifact Detection Strategy

Refer [notebook 5](notebooks/5_ct_scan_level_thresh_visualization.ipynb) for detailed implementation.

**Objective:** Extract binary masks identifying metal artifacts in CT scans using HU-based thresholding with morphological refinement.

### Core Algorithm:

1. **HU Analysis**
   - Extract maximum HU value per slice across the volume
   - Build slice-level HU profiles for threshold optimization

2. **Adaptive Thresholding**
   - Calculate `tau_global = max(max_normal_hu, min_abnormal_hu)`
   - Apply *sensitivity buffer* `DELTA`: `final_threshold = tau - DELTA(250 HU)`
   - Generate binary mask: `mask = (ct_vol >= final_threshold)`

3. **Morphological Post-processing**
   - Fill small holes (area < 25 pixels)
   - **Did not** apply binary opening/closing for boundary smoothing **as small metal holes will be closed**



### Example Plots
![CT Label Example 1PA133](images/5_1PA133_label_example.png)
![CT Label Example 1PA136](images/5_1PA136_label_example.png)
![CT Label Example 1PA147](images/5_1PA147_label_example.png)
![CT Label Example 1PA151](images/5_1PA151_label_example.png)
![CT Label Example 1PA152](images/5_1PA152_label_example.png)
![CT Label Example 1PA159](images/5_1PA159_label_example.png)


### MR Preprocessing
0. Did not resample according to pixel spacing and did not remove bias field (takes long)
1. Min max normalization + then clip to (0, 255)
  - MR intensity is clipped to 2000
2. Center image
  - Problem: mask file didn't cover the marker sometimes
  ![Marker fail example](images/9_1PA113_21.png) Use big thresh for contour extraction
3. Pad to square
4. Resize to (240, 240)
  - Problem: small hotspots were not preserved during resizing
5. Apply masks
6. Extract slices and save as png.
  - For normal image, extract slices between [30:-30:2]
  - For abnormal image, extract slices between [15:-15:1] and is abnormal and extract anomaly is at least 3 pixel
  - ![Marker fail example](images/6_1PA168_68.png) the marker/clip still exists after applying mask.
  - ![Background fail example](images/6_1PA117_98.png) the background is not properly masked.
  - Not sure if all the scans have the same orientation

### Processed MR + CT Label
![1PA133](images/6_vis_1PA133.png)
![1PA136](images/6_vis_1PA136.png)
![1PA169](images/6_vis_1PA169.png)
