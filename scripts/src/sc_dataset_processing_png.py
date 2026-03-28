import os
import sys
import json
import random
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import argparse


sys.path.append("../scripts/src/")

from artifact_detector import MetalArtifactDetector
from utils.processing_utils import (
    center_pad_single_slice,
    center_pad_single_slice_by_params,
    resize_image,
    get_ids_from_ungood_test_folder,
    center_crop,
    load_scan,
)
from utils.path_utils import create_output_dirs


# ================================================================
# CONFIGURATION
# ================================================================
DELTA = 200
SLICE_INDEX_START_NORMAL = 25
SLICE_INDEX_END_NORMAL = -20
TARGET_SIZE = (240, 240)
SEED = 24
random.seed(SEED)
np.random.seed(SEED)


# ================================================================
# HELPER FUNCTIONS
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="PNG pelvis preprocessing")
    parser.add_argument("--dir_pelvis", type=str, required=True)
    parser.add_argument("--dir_output", type=str, required=True)
    return parser.parse_args()


def save_png(image, path, cmap="bone"):
    """
    Saves a single image or mask as a PNG.
    - If cmap is 'binary', forces the output to be a single-channel 0/255 mask.
    - Otherwise, uses matplotlib to apply the specified colormap.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if cmap == "binary":
        # 1. Binarize and convert to 8-bit integer (0 or 255)
        # Assuming the input 'image' array is already 0/1 or similar.
        img_data = (image > 0).astype(np.uint8) * 255

        # 2. Save using Pillow in L (8-bit grayscale) mode.
        Image.fromarray(img_data, mode="L").save(path)

    else:
        # Use Matplotlib for images that require a colormap (like 'bone' for MR).
        # This is safe for the MR images as they are not used for pixel-wise metrics.
        plt.imsave(path, image, cmap=cmap)


# ================================================================
# SAVE FUNCTIONS
# ================================================================
def save_train_slice(slice_img, slice_body_mask, output_dir, split, subset, id_, i):
    """Save structure for TRAIN (no label folder)."""
    path_img = os.path.join(output_dir, split, subset, f"{id_}_{i}.png")
    path_bodymask = os.path.join(output_dir, split, "bodymask", f"{id_}_{i}.png")
    save_png(slice_img, path_img)
    save_png(slice_body_mask, path_bodymask, cmap="binary")


def save_eval_slice(slice_img, slice_mask, slice_body_mask, output_dir, split, subset, id_, i):
    """Save structure for VALID and TEST (img/label/bodymask)."""
    base_path = os.path.join(output_dir, split, subset)
    save_png(slice_img, os.path.join(base_path, "img", f"{id_}_{i}.png"))
    save_png(slice_mask, os.path.join(base_path, "label", f"{id_}_{i}.png"), cmap="binary")
    save_png(slice_body_mask, os.path.join(base_path, "bodymask", f"{id_}_{i}.png"), cmap="binary")


# ================================================================
# MAIN PROCESSING FUNCTION
# ================================================================
def process_slices(
    mr_norm, body_mask_vol, id_, split, subset,
    output_dir, start_idx=SLICE_INDEX_START_NORMAL, end_offset=SLICE_INDEX_END_NORMAL,
    mask_vol=None, abnormal_slices=None
):
    """Process and save all slices (good or Ungood) for a scan."""
    slices = mr_norm.shape[2]
    slice_indices = (
        range(start_idx, slices + end_offset)
        if abnormal_slices is None else abnormal_slices
    )

    for i in tqdm(slice_indices, desc=f"{id_}-{split}-{subset}"):
        slice_img = mr_norm[:, :, i]
        slice_body_mask = body_mask_vol[:, :, i]
        slice_mask = mask_vol[:, :, i] if mask_vol is not None else np.zeros_like(slice_body_mask)

        # Center pad
        slice_img_centered, (pad_h, pad_w) = center_pad_single_slice(slice_img)
        slice_body_mask_centered = center_pad_single_slice_by_params(slice_body_mask, pad_h, pad_w)
        slice_mask_centered = center_pad_single_slice_by_params(slice_mask, pad_h, pad_w)

        # Resize and crop
        slice_img_cropped = center_crop(resize_image(slice_img_centered, target_size=TARGET_SIZE))
        slice_body_mask_cropped = center_crop(resize_image(slice_body_mask_centered, target_size=TARGET_SIZE))
        slice_mask_cropped = center_crop(resize_image(slice_mask_centered, target_size=TARGET_SIZE))

        # Skip tiny abnormal masks
        if mask_vol is not None and slice_mask_cropped.sum() < 3:
            continue

        if split == "train":
            save_train_slice(slice_img_cropped, slice_body_mask_cropped, output_dir, split, subset, id_, i)
        else:
            save_eval_slice(slice_img_cropped, slice_mask_cropped, slice_body_mask_cropped, output_dir, split, subset, id_, i)


# ================================================================
# SPLIT HANDLERS
# ================================================================
def process_good_scans(det, ids, split, dir_pelvis, dir_output):
    for id_ in ids:
        mr, mr_norm, _, body_mask_vol = load_scan(dir_pelvis, det, id_)
        process_slices(mr_norm, body_mask_vol, id_, split, "good", dir_output)


def process_ungood_scans(det, ids, split, dir_pelvis, dir_output, anomaly_range):
    for id_ in ids:
        mr, mr_norm, ct, body_mask_vol = load_scan(dir_pelvis, det, id_)
        slices = mr_norm.shape[2]
        abnormal_slices = list(range(anomaly_range[id_][0], anomaly_range[id_][-1]))

        # Compute HU-based mask
        df_hu = det.score_volume_hu(ct, scan_id=id_, slice_axis=2)
        df_hu["label"] = np.isin(df_hu["slice_idx"], abnormal_slices).astype(np.uint8)
        scan_value, _ = det.pick_global_tau_by_hu(df_hu, label_col="label")
        scan_value -= DELTA                   
        tau = min(scan_value, 2000)

        mask_vol = (ct >= tau).astype(np.uint8)
        abnormal_slices = [i for i in abnormal_slices if 15 <= i < slices - 15]

        mask_ref = det.refine_mask_with_mr(mask_vol, mr, lo_diff=5, up_diff=10)
        mask_ref = det.postprocess_mask_volume_morph(mask_ref, disk_size=5, min_area_for_smooth=50, slice_axis=2)

        process_slices(mr_norm, body_mask_vol, id_, split, "Ungood", dir_output, mask_vol=mask_ref, abnormal_slices=abnormal_slices)


# ================================================================
# EXPORT FULL ANOMALOUS TEST CASES - PATIENT-WISE
# ================================================================
def export_full_anomalous_cases_png(det, ids, dir_pelvis, dir_output, anomaly_range):
    """
    For each test anomalous scan, save all slices (except first 30 & last 15)
    in patient-wise folders: img (MR), label (mask), bodymask, as PNG.
    """
    for id_ in ids:
        mr, mr_norm, ct, body_mask_vol = load_scan(dir_pelvis, det, id_)
        slices = mr_norm.shape[2]
        start_idx = 10
        end_idx = slices - 5
        if end_idx <= start_idx:
            continue  # Skip if scan too small

        abn_start, abn_end = anomaly_range.get(id_, (start_idx, end_idx))
        abnormal_slices = list(range(abn_start, abn_end))
        df_hu = det.score_volume_hu(ct, scan_id=id_, slice_axis=2)
        df_hu["label"] = np.isin(df_hu["slice_idx"], abnormal_slices).astype(np.uint8)
        scan_value, _ = det.pick_global_tau_by_hu(df_hu, label_col="label")
        scan_value -= DELTA
        tau = min(scan_value, 2000)
        mask_vol = (ct >= tau).astype(np.uint8)
        mask_ref = det.refine_mask_with_mr(mask_vol, mr, lo_diff=5, up_diff=10)
        mask_ref = det.postprocess_mask_volume_morph(mask_ref, disk_size=5, min_area_for_smooth=50, slice_axis=2)

        # Create patient-wise folders
        base_path = os.path.join(dir_output, "test", "Ungood_whole_patient_scans")
        for subdir in ("img", "label", "bodymask"):
            os.makedirs(os.path.join(base_path, subdir), exist_ok=True)

        for i in tqdm(range(start_idx, end_idx), desc=f"patientwise_anom_test_png_{id_}"):
            # Process each slice the same (including the ground thruth and body mask)
            # Body mask is also processed to be able to be used in the postprocessing
            slice_img = mr_norm[:, :, i]
            slice_body_mask = body_mask_vol[:, :, i]
            slice_mask = mask_ref[:, :, i]

            slice_img_centered, (pad_h, pad_w) = center_pad_single_slice(slice_img)
            slice_body_mask_centered = center_pad_single_slice_by_params(slice_body_mask, pad_h, pad_w)
            slice_mask_centered = center_pad_single_slice_by_params(slice_mask, pad_h, pad_w)

            slice_img_cropped = center_crop(resize_image(slice_img_centered, target_size=TARGET_SIZE))
            slice_body_mask_cropped = center_crop(resize_image(slice_body_mask_centered, target_size=TARGET_SIZE))
            slice_mask_cropped = center_crop(resize_image(slice_mask_centered, target_size=TARGET_SIZE))

            # Save files as PNG
            save_png(slice_img_cropped, os.path.join(base_path, "img", f"{id_}_{i}.png"), cmap="bone")
            save_png(slice_mask_cropped, os.path.join(base_path, "label", f"{id_}_{i}.png"), cmap="binary")
            save_png(slice_body_mask_cropped, os.path.join(base_path, "bodymask", f"{id_}_{i}.png"), cmap="binary")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    args = parse_args()
    dir_pelvis = args.dir_pelvis
    dir_output = args.dir_output
    excel_overview = os.path.join(dir_pelvis, "overview", "1_pelvis_train.xlsx")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    create_output_dirs(dir_output)

    # ---------------- Load IDs and splits ----------------
    df_overview = pd.read_excel(excel_overview, sheet_name="MR")
    ids_all = [i for i in df_overview["ID"].tolist() if i.startswith("1PA")]

    with open("/home/user/jverhoek/sct-ood-dataset/labels/labels_implant.json") as f:
        data_implant = json.load(f)

    data_abnormal = data_implant['type1']
    df_labels_1 = pd.DataFrame([{"id": k, **v} for item in data_abnormal for k, v in item.items()])
    list_na_ids = df_labels_1[df_labels_1.isna().any(axis=1)]['id'].tolist()
    df_labels_1 = df_labels_1.dropna()
    df_labels_1 = df_labels_1[df_labels_1['body_part'] == 'pelvis']
    df_labels_1["anomaly_start"] = df_labels_1[["ct_start", "mr_start"]].min(axis=1)
    df_labels_1["anomaly_end"] = df_labels_1[["ct_end", "mr_end"]].max(axis=1)
    anomaly_range = {
        id_: (int(start), int(end))
        for id_, start, end in zip(df_labels_1["id"], df_labels_1["anomaly_start"], df_labels_1["anomaly_end"])
    }

    ids_abnormal_all = [i for i in df_labels_1['id'].tolist() if not i.startswith("1PC")]
    if '1PA030' in ids_abnormal_all:
        ids_abnormal_all.remove('1PA030')
    if '1PA170' in ids_abnormal_all:
        ids_abnormal_all.remove('1PA170')

    with open("/home/user/jverhoek/sct-ood-dataset/labels/labels_others.json") as f:
        data_other = json.load(f)
    ids_other = [
        pid for item in data_other['types_2_to_7']
        for pid, info in item.items()
        if pid.startswith("1P") and str(info.get("type")) in {"2", "3", "4", "5", "6"}
    ]

    ids_used = list(set(ids_all) - set(list_na_ids))
    ids_normal_all = sorted(set(ids_used) - set(ids_abnormal_all) - set(ids_other))

    # --- Splits ---
    val_frac, test_frac = 0.1, 0.2
    random.shuffle(ids_normal_all)
    random.shuffle(ids_abnormal_all)

    n_val_abn = max(4, int(len(ids_abnormal_all) * val_frac))
    n_test_abn = max(1, int(len(ids_abnormal_all) * test_frac))
    ids_abnormal_valid = ids_abnormal_all[:n_val_abn]
    ids_abnormal_test = ids_abnormal_all[n_val_abn:n_val_abn + n_test_abn] + ids_abnormal_all[n_val_abn + n_test_abn:]

    n_val_norm = max(1, int(len(ids_normal_all) * val_frac))
    n_test_norm = max(1, int(len(ids_normal_all) * test_frac))
    ids_normal_valid = ids_normal_all[:n_val_norm]
    ids_normal_test = ids_normal_all[n_val_norm:n_val_norm + n_test_norm]
    ids_normal_train = ids_normal_all[n_val_norm + n_test_norm:]

    logger.info(f"Train Normal: {len(ids_normal_train)}, Val Normal: {len(ids_normal_valid)}, Test Normal: {len(ids_normal_test)}")
    logger.info(f"Val Abnormal: {len(ids_abnormal_valid)}, Test Abnormal: {len(ids_abnormal_test)}")

    # ---------------- Processing ----------------
    det = MetalArtifactDetector()

    logger.info("=== TRAIN GOOD ===")
    process_good_scans(det, ids_normal_train, "train", dir_pelvis, dir_output)

    logger.info("=== VALID GOOD ===")
    process_good_scans(det, ids_normal_valid, "valid", dir_pelvis, dir_output)

    logger.info("=== VALID UNGOOD ===")
    process_ungood_scans(det, ids_abnormal_valid, "valid", dir_pelvis, dir_output, anomaly_range)

    logger.info("=== TEST GOOD ===")
    process_good_scans(det, ids_normal_test, "test", dir_pelvis, dir_output)

    logger.info("=== TEST UNGOOD ===")
    process_ungood_scans(det, ids_abnormal_test, "test", dir_pelvis, dir_output, anomaly_range)

    logger.info("Finished all splits - NIFTI dataset created.")

    # Only patients that actually have slices saved in test/Ungood
    ids_with_ungood_slices = get_ids_from_ungood_test_folder(dir_output)
    ids_abnormal_test_effective = sorted(ids_with_ungood_slices.intersection(ids_abnormal_test))

    logger.info(f"Test Ungood patients on disk: {len(ids_abnormal_test_effective)}")

    export_full_anomalous_cases_png(det, ids_abnormal_test_effective, dir_pelvis, dir_output, anomaly_range)
    logger.info("✅ Finished - PATIENT-WISE ANOMALOUS TEST CASES")