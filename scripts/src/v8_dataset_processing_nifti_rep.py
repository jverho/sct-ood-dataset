import os
import sys
import json
import random
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

sys.path.append("../scripts/src/")

from artifact_detector import MetalArtifactDetector
from utils.processing_utils import (
    load_nifti_image,
    apply_mask,
    center_pad_single_slice,
    center_pad_single_slice_by_params,
    resize_image,
    minmax_normalize_numpy,
    save_np_to_nifti,
)
from utils.path_utils import create_output_dirs


# ================================================================
# CONFIGURATION
# ================================================================
DIR_PELVIS = "/local/scratch/jverhoek/datasets/Task1/pelvis/"
DIR_OUTPUT = os.path.join("/local/scratch/jverhoek/datasets/", "synth23_pelvis_v8_nifti_rep")

DELTA = 200
THRESH_MR_MASK = 0.1
SLICE_INDEX_START_NORMAL = 25
SLICE_INDEX_END_NORMAL = -20

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET_SIZE = (240, 240)
TARGET_SIZE_CROP = (224, 224)
EXCEL_OVERVIEW = "/local/scratch/jverhoek/datasets/Task1/pelvis/overview/1_pelvis_train.xlsx"


# ================================================================
# HELPER FUNCTIONS
# ================================================================
def center_crop(slice_, target_size=TARGET_SIZE_CROP):
    h, w = slice_.shape
    th, tw = target_size
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return slice_[i:i + th, j:j + tw]


def load_scan(det, id_):
    """Load MR, CT, and mask volumes and return both raw and normalized MR + body mask."""
    dir_scan = os.path.join(DIR_PELVIS, id_)
    mr = load_nifti_image(os.path.join(dir_scan, "mr.nii.gz"))
    ct = load_nifti_image(os.path.join(dir_scan, "ct.nii.gz"))
    mask = load_nifti_image(os.path.join(dir_scan, "mask.nii.gz"))

    body_mask_vol = det.get_body_mask_threshold(mr * mask, threshold_ct_body_mask=THRESH_MR_MASK)
    body_mask_vol = np.logical_and(body_mask_vol > 0, mask > 0).astype(np.uint8)

    masked_mr = apply_mask(mr, body_mask_vol)
    mr_norm = minmax_normalize_numpy(masked_mr)

    return mr, mr_norm, ct, body_mask_vol


def extract_3ch_slice(volume, i):
    """Extract a 3-channel slice (current, current, current)."""
    idxs = [i, i, i]
    return volume[:, :, idxs].copy()


# ================================================================
# SAVE FUNCTIONS
# ================================================================
def save_train_slice(slice_imgs_cropped, slice_body_mask_cropped, output_dir, split, subset, id_, i):
    """Save slice structure for TRAIN (no label folder)."""
    save_np_to_nifti(slice_imgs_cropped, os.path.join(output_dir, split, subset, f"{id_}_{i}.nii"))
    save_np_to_nifti(slice_body_mask_cropped, os.path.join(output_dir, split, "bodymask", f"{id_}_{i}.nii"))


def save_eval_slice(slice_imgs_cropped, slice_mask_cropped, slice_body_mask_cropped, output_dir, split, subset, id_, i):
    """Save slice structure for VALID and TEST (with img/label/bodymask)."""
    base_path = os.path.join(output_dir, split, subset)
    save_np_to_nifti(slice_imgs_cropped, os.path.join(base_path, "img", f"{id_}_{i}.nii"))
    save_np_to_nifti(slice_mask_cropped, os.path.join(base_path, "label", f"{id_}_{i}.nii"))
    save_np_to_nifti(slice_body_mask_cropped, os.path.join(base_path, "bodymask", f"{id_}_{i}.nii"))


# ================================================================
# MAIN PROCESSING FUNCTION
# ================================================================
def process_slices(
    mr_norm, body_mask_vol, id_, split, subset,
    output_dir, start_idx=SLICE_INDEX_START_NORMAL, end_offset=SLICE_INDEX_END_NORMAL,
    mask_vol=None, abnormal_slices=None
):
    """Process all slices (good or Ungood) for a scan."""
    slices = mr_norm.shape[2]
    slice_indices = (
        range(start_idx, slices + end_offset)
        if abnormal_slices is None else abnormal_slices
    )

    for i in tqdm(slice_indices, desc=f"{id_}-{split}-{subset}"):
        slice_imgs = extract_3ch_slice(mr_norm, i)
        slice_img = slice_imgs[:, :, 1]
        slice_body_mask = body_mask_vol[:, :, i]
        slice_mask = mask_vol[:, :, i] if mask_vol is not None else np.zeros_like(slice_body_mask)

        # Center-pad using same params for all 3 channels + masks
        slice_img_centered, (pad_h, pad_w) = center_pad_single_slice(slice_img)
        slice_body_mask_centered = center_pad_single_slice_by_params(slice_body_mask, pad_h, pad_w)
        slice_mask_centered = center_pad_single_slice_by_params(slice_mask, pad_h, pad_w)

        # Resize + crop all channels
        slice_imgs_cropped = np.stack([
            center_crop(
                resize_image(
                    center_pad_single_slice_by_params(slice_imgs[:, :, c], pad_h, pad_w),
                    target_size=TARGET_SIZE,
                )
            )
            for c in range(slice_imgs.shape[2])
        ], axis=-1)
        slice_imgs_cropped = np.expand_dims(slice_imgs_cropped, axis=2)  # (H, W, 1, 3)

        slice_body_mask_cropped = center_crop(
            resize_image(slice_body_mask_centered, target_size=TARGET_SIZE)
        )
        slice_mask_cropped = center_crop(
            resize_image(slice_mask_centered, target_size=TARGET_SIZE)
        )

        # Skip small masks for Ungood
        if mask_vol is not None and slice_mask_cropped.sum() < 3:
            continue

        # Save depending on split
        if split == "train":
            save_train_slice(slice_imgs_cropped, slice_body_mask_cropped, output_dir, split, subset, id_, i)
        else:
            save_eval_slice(slice_imgs_cropped, slice_mask_cropped, slice_body_mask_cropped, output_dir, split, subset, id_, i)


# ================================================================
# SPLIT HANDLERS
# ================================================================
def process_good_scans(det, ids, split, output_dir):
    for id_ in ids:
        mr, mr_norm, _, body_mask_vol = load_scan(det, id_)
        process_slices(mr_norm, body_mask_vol, id_, split, "good", output_dir)


def process_ungood_scans(det, ids, split, output_dir, anomaly_range):
    for id_ in ids:
        mr, mr_norm, ct, body_mask_vol = load_scan(det, id_)
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

        process_slices(mr_norm, body_mask_vol, id_, split, "Ungood", output_dir, mask_vol=mask_ref, abnormal_slices=abnormal_slices)


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    create_output_dirs(DIR_OUTPUT)

    # ---------------- Load IDs and splits ----------------
    df_overview = pd.read_excel(EXCEL_OVERVIEW, sheet_name="MR")
    ids_all = [i for i in df_overview["ID"].tolist() if i.startswith("1P")]

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

    ids_abnormal_all = df_labels_1['id'].tolist()
    if '1PA030' in ids_abnormal_all:
        ids_abnormal_all.remove('1PA030')
    if '1PA170' in ids_abnormal_all:
        ids_abnormal_all.remove('1PA170')
    if '1PC029' in ids_abnormal_all:
        ids_abnormal_all.remove('1PC029')
    if '1PC015' in ids_abnormal_all:
        ids_abnormal_all.remove('1PC015')

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
    process_good_scans(det, ids_normal_train, "train", DIR_OUTPUT)

    logger.info("=== VALID GOOD ===")
    process_good_scans(det, ids_normal_valid, "valid", DIR_OUTPUT)

    logger.info("=== VALID UNGOOD ===")
    process_ungood_scans(det, ids_abnormal_valid, "valid", DIR_OUTPUT, anomaly_range)

    logger.info("=== TEST GOOD ===")
    process_good_scans(det, ids_normal_test, "test", DIR_OUTPUT)

    logger.info("=== TEST UNGOOD ===")
    process_ungood_scans(det, ids_abnormal_test, "test", DIR_OUTPUT, anomaly_range)

    logger.info("✅ Finished all splits - NIFTI dataset created.")
