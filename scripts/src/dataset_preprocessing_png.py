import os
import sys
import json
import random

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from tqdm import tqdm

sys.path.append("../scripts/src/")

from artifact_detector import MetalArtifactDetector

from utils.processing_utils import (
    load_nifti_image,
    apply_mask,
    center_pad_single_slice,
    center_pad_single_slice_by_params,
    center_pad_single_slice_by_params,
    resize_image,
    minmax_normalize_numpy,
    save_np_to_nifti
)

from utils.path_utils import create_output_dirs

DIR_PELVIS = "/local/scratch/jverhoek/datasets/Task1/pelvis/"
DIR_OUTPUT = os.path.join(os.getcwd(), "output", "synth23_pelvis_v6_png")

DELTA = 200
# THRESH_MR_MASK = 15
THRESH_MR_MASK = 0.1

SEED = 24
random.seed(SEED)
np.random.seed(SEED)
EXCEL_OVERVIEW = "/local/scratch/jverhoek/datasets/Task1/pelvis/overview/1_pelvis_train.xlsx"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- Create output directories for PNGs
    logger.info("Creating output directories...")
    create_output_dirs(DIR_OUTPUT)
    logger.info("Output directories created.")

    # fractions for val/test
    val_frac = 0.1
    test_frac = 0.2

    # --- Load overview Excel ---
    df_overview = pd.read_excel(EXCEL_OVERVIEW, sheet_name="MR")
    ids_all = df_overview["ID"].tolist()
    ids_all = [i for i in ids_all if i.startswith("1PA")]  # pelvis only

    # --- Load implant labels (type1 anomalies) ---
    with open("/home/user/jverhoek/sct-ood-dataset/labels/labels_implant.json") as f:
        data_implant = json.load(f)

    data_abnormal = data_implant['type1']
    df_labels_1 = pd.DataFrame([
        {"id": k, **v}
        for item in data_abnormal
        for k, v in item.items()
    ])

    # filter pelvis only, drop NAs
    list_na_ids = df_labels_1[df_labels_1.isna().any(axis=1)]['id'].tolist()
    df_labels_1 = df_labels_1.dropna()
    df_labels_1 = df_labels_1[df_labels_1['body_part'] == 'pelvis']

    # compute anomaly ranges
    df_labels_1["anomaly_start"] = df_labels_1[["ct_start", "mr_start"]].min(axis=1)
    df_labels_1["anomaly_end"] = df_labels_1[["ct_end", "mr_end"]].max(axis=1)
    anomaly_range = {
        id_: (int(start), int(end))
        for id_, start, end in zip(df_labels_1["id"], df_labels_1["anomaly_start"], df_labels_1["anomaly_end"])
    }

    ids_abnormal_all = df_labels_1['id'].tolist()

    # Metal Artifact outside of body leads to messed up anomaly mask (everything outside of the body is taken as anomaly)
    if '1PA030' in ids_abnormal_all:
        ids_abnormal_all.remove('1PA030')

    # --- Load other labels (only categories 2–6, pelvis only) ---
    with open("/home/user/jverhoek/sct-ood-dataset/labels/labels_others.json") as f:
        data_other = json.load(f)

    ids_other = []
    for item in data_other['types_2_to_7']:
        pid, info = list(item.items())[0]
        if pid.startswith("1P") and str(info.get("type")) in {"2", "3", "4", "5", "6"}:
            ids_other.append(pid)

    # --- Build normal set ---
    ids_used = list(set(ids_all) - set(list_na_ids))
    ids_normal_all = list(set(ids_used) - set(ids_abnormal_all) - set(ids_other))

    # --- Filter out 1PC IDs ---
    ids_normal_all = [i for i in ids_normal_all if not i.startswith("1PC")]
    ids_abnormal_all = [i for i in ids_abnormal_all if not i.startswith("1PC")]

    ids_normal_all.sort()  # Ensure a canonical (reproducible) starting order

    # Shuffle
    random.shuffle(ids_normal_all)
    random.shuffle(ids_abnormal_all)

    # --- Abnormal splits ---
    n_val_abn = max(4, int(len(ids_abnormal_all) * val_frac))
    n_test_abn = max(1, int(len(ids_abnormal_all) * test_frac))

    ids_abnormal_valid = ids_abnormal_all[:n_val_abn]
    ids_abnormal_test = ids_abnormal_all[n_val_abn:n_val_abn + n_test_abn]
    extra_abn = ids_abnormal_all[n_val_abn + n_test_abn:]
    ids_abnormal_test += extra_abn  # all disjoint

    # --- Normal splits ---
    n_val_norm = max(1, int(len(ids_normal_all) * val_frac))
    n_test_norm = max(1, int(len(ids_normal_all) * test_frac))

    ids_normal_valid = ids_normal_all[:n_val_norm]
    ids_normal_test = ids_normal_all[n_val_norm:n_val_norm + n_test_norm]
    ids_normal_train = ids_normal_all[n_val_norm + n_test_norm:]  # remaining for training

    # --- Logging ---
    logger.info(f"Train Normal scans: {len(ids_normal_train)}")
    logger.info(f"Valid Normal scans: {len(ids_normal_valid)}")
    logger.info(f"Valid Abnormal scans: {len(ids_abnormal_valid)}")
    logger.info(f"Test Normal scans: {len(ids_normal_test)}")
    logger.info(f"Test Abnormal scans: {len(ids_abnormal_test)}")
    logger.info(f"Valid Abnormal scans IDs: {ids_abnormal_valid}")
    logger.info(f"Test Abnormal scans IDs: {ids_abnormal_test}")

    det = MetalArtifactDetector()  # instantiate once

    # Train: good only
    logger.info("Processing training normal scans...")
    for id_ in ids_normal_train:
        dir_scan = os.path.join(DIR_PELVIS, id_)
        path_mr = os.path.join(dir_scan, "mr.nii.gz")
        path_mask = os.path.join(dir_scan, "mask.nii.gz")
        path_ct = os.path.join(dir_scan, "ct.nii.gz")

        mr_image = load_nifti_image(path_mr)
        ct_image = load_nifti_image(path_ct)
        mask = load_nifti_image(path_mask)

        body_mask = det.get_body_mask_threshold(mr_image * mask, threshold_ct_body_mask=THRESH_MR_MASK)
        body_mask = np.logical_and(body_mask > 0, mask > 0)

        masked_mr = apply_mask(mr_image, body_mask)
        mr_normalized = minmax_normalize_numpy(masked_mr)

        slices = mr_normalized.shape[2]

        for i in tqdm(range(25, slices - 20, 1)):
            slice_image = mr_normalized[:, :, i]
            slice_image_centered, (pad_h, pad_w) = center_pad_single_slice(slice_image)
            slice_image_resized = resize_image(slice_image_centered, target_size=[240, 240])

            plt.imsave(os.path.join(DIR_OUTPUT, "train", "good", f"{id_}_{i}.png"), slice_image_resized, cmap="bone")
    logger.info("Finished processing training normal scans.")

    # Valid: good only
    logger.info("Processing validation normal scans...")
    for id_ in ids_normal_valid:
        dir_scan = os.path.join(DIR_PELVIS, id_)
        path_mr = os.path.join(dir_scan, "mr.nii.gz")

        path_mask = os.path.join(dir_scan, "mask.nii.gz")
        path_ct = os.path.join(dir_scan, "ct.nii.gz")

        mr_image = load_nifti_image(path_mr)
        ct_image = load_nifti_image(path_ct)
        mask = load_nifti_image(path_mask)

        body_mask = det.get_body_mask_threshold(mr_image * mask, threshold_ct_body_mask=THRESH_MR_MASK)
        body_mask = np.logical_and(body_mask > 0, mask > 0)

        masked_mr = apply_mask(mr_image, body_mask)
        mr_normalized = minmax_normalize_numpy(masked_mr)

        slices = mr_normalized.shape[2]

        for i in tqdm(range(25, slices - 20, 1)):
            slice_image = mr_normalized[:, :, i]
            slice_image_centered, (pad_h, pad_w) = center_pad_single_slice(slice_image)
            slice_image_resized = resize_image(slice_image_centered, target_size=[240, 240])
            slice_mask = np.zeros_like(slice_image_resized)

            plt.imsave(os.path.join(DIR_OUTPUT, "valid", "good", "img", f"{id_}_{i}.png"), slice_image_resized, cmap="bone")
            plt.imsave(os.path.join(DIR_OUTPUT, "valid", "good", "label", f"{id_}_{i}.png"), slice_mask, cmap="bone")
    logger.info("Finished processing validation normal scans.")

    # Valid: Ungood
    logger.info("Processing validation normal scans for Ungood slices...")
    for id_ in ids_abnormal_valid:
        dir_scan = os.path.join(DIR_PELVIS, id_)
        path_mr = os.path.join(dir_scan, "mr.nii.gz")

        path_mask = os.path.join(dir_scan, "mask.nii.gz")
        path_ct = os.path.join(dir_scan, "ct.nii.gz")

        mr_image = load_nifti_image(path_mr)
        ct_image = load_nifti_image(path_ct)
        mask = load_nifti_image(path_mask)

        body_mask = det.get_body_mask_threshold(mr_image * mask, threshold_ct_body_mask=THRESH_MR_MASK)
        body_mask = np.logical_and(body_mask > 0, mask > 0)

        masked_mr = apply_mask(mr_image, body_mask)
        mr_normalized = minmax_normalize_numpy(masked_mr)

        slices = mr_normalized.shape[2]
        abnormal_slices = list(range(anomaly_range[id_][0], anomaly_range[id_][-1]))

        # Extract label masks
        df_hu = det.score_volume_hu(ct_image, scan_id=id_, slice_axis=2)
        df_hu["label"] = np.isin(df_hu["slice_idx"], abnormal_slices).astype(np.uint8)
        scan_value, info = det.pick_global_tau_by_hu(df_hu, label_col="label")
        scan_value -= DELTA
        tau = min(scan_value, 2000)
        logger.info("tau: %s | info: %s", tau, info)

        mask_vol = (ct_image >= tau).astype(np.uint8)

        abnormal_slices = [i for i in abnormal_slices if i >= 15 and i < slices - 15]
        # logger.info(id_, "abnormal slices:", abnormal_slices)
        logger.info("%s abnormal slices: %s", id_, abnormal_slices)

        lo_diff_val = 5
        up_diff_val = 10
        mask_vol_refined = det.refine_mask_with_mr(mask_vol, mr_image, lo_diff=lo_diff_val, up_diff=up_diff_val)
        mask_vol_refined = det.postprocess_mask_volume_morph(mask_vol_refined, disk_size=5, min_area_for_smooth=50,
                                                             slice_axis=2)
        mask_vol = mask_vol_refined

        for i in tqdm(abnormal_slices):
            slice_image = mr_normalized[:, :, i]
            slice_mask = mask_vol[:, :, i]

            slice_image_centered, (pad_h, pad_w) = center_pad_single_slice(slice_image)
            slice_mask = center_pad_single_slice_by_params(slice_mask, pad_h, pad_w)

            slice_image_resized = resize_image(slice_image_centered, target_size=[240, 240])
            slice_mask = resize_image(slice_mask, target_size=[240, 240])

            if slice_mask.sum() < 3:
                # print(f"Skipping slice {i} for {id_} due to insufficient mask data.")
                continue

            plt.imsave(os.path.join(DIR_OUTPUT, "valid", "Ungood", "img", f"{id_}_{i}.png"), slice_image_resized, cmap="bone")
            plt.imsave(os.path.join(DIR_OUTPUT, "valid", "Ungood", "label", f"{id_}_{i}.png"), slice_mask, cmap="bone")

    logger.info("Finished processing validation normal scans for Ungood slices.")

    # Test: good only
    logger.info("Processing test normal scans...")
    for id_ in ids_normal_test:
        dir_scan = os.path.join(DIR_PELVIS, id_)
        path_mr = os.path.join(dir_scan, "mr.nii.gz")

        path_mask = os.path.join(dir_scan, "mask.nii.gz")
        path_ct = os.path.join(dir_scan, "ct.nii.gz")

        mr_image = load_nifti_image(path_mr)
        ct_image = load_nifti_image(path_ct)
        mask = load_nifti_image(path_mask)

        body_mask = det.get_body_mask_threshold(mr_image * mask, threshold_ct_body_mask=THRESH_MR_MASK)
        body_mask = np.logical_and(body_mask > 0, mask > 0)

        masked_mr = apply_mask(mr_image, body_mask)
        mr_normalized = minmax_normalize_numpy(masked_mr)

        slices = mr_normalized.shape[2]

        for i in tqdm(range(25, slices - 20, 1)):
            slice_image = mr_normalized[:, :, i]
            slice_image_centered, (pad_h, pad_w) = center_pad_single_slice(slice_image)
            slice_image_resized = resize_image(slice_image_centered, target_size=[240, 240])
            slice_mask = np.zeros_like(slice_image_resized)

            plt.imsave(os.path.join(DIR_OUTPUT, "test", "good", "img", f"{id_}_{i}.png"), slice_image_resized, cmap="bone")
            plt.imsave(os.path.join(DIR_OUTPUT, "test", "good", "label", f"{id_}_{i}.png"), slice_mask, cmap="bone")
    logger.info("Finished processing test normal scans.")

    # Test: Ungood
    logger.info("Processing test abnormal scans for Ungood slices...")
    for id_ in ids_abnormal_test:
        dir_scan = os.path.join(DIR_PELVIS, id_)
        path_mr = os.path.join(dir_scan, "mr.nii.gz")

        path_mask = os.path.join(dir_scan, "mask.nii.gz")
        path_ct = os.path.join(dir_scan, "ct.nii.gz")

        mr_image = load_nifti_image(path_mr)
        ct_image = load_nifti_image(path_ct)
        mask = load_nifti_image(path_mask)

        body_mask = det.get_body_mask_threshold(mr_image * mask, threshold_ct_body_mask=THRESH_MR_MASK)
        body_mask = np.logical_and(body_mask > 0, mask > 0)

        masked_mr = apply_mask(mr_image, body_mask)
        mr_normalized = minmax_normalize_numpy(masked_mr)

        slices = mr_normalized.shape[2]
        abnormal_slices = list(range(anomaly_range[id_][0], anomaly_range[id_][-1]))

        # Extract label masks

        df_hu = det.score_volume_hu(ct_image, scan_id=id_, slice_axis=2)
        df_hu["label"] = np.isin(df_hu["slice_idx"], abnormal_slices).astype(np.uint8)
        scan_value, info = det.pick_global_tau_by_hu(df_hu, label_col="label")
        scan_value -= DELTA
        tau = min(scan_value, 2000)

        print("tau:", tau, "| info:", info)

        mask_vol = (ct_image >= tau).astype(np.uint8)

        abnormal_slices = [i for i in abnormal_slices if i >= 15 and i < slices - 15]

        lo_diff_val = 5
        up_diff_val = 10
        mask_vol_refined = det.refine_mask_with_mr(mask_vol, mr_image, lo_diff=lo_diff_val, up_diff=up_diff_val)
        mask_vol_refined = det.postprocess_mask_volume_morph(mask_vol_refined, disk_size=5, min_area_for_smooth=50,
                                                             slice_axis=2)
        mask_vol = mask_vol_refined

        for i in tqdm(abnormal_slices):
            slice_image = mr_normalized[:, :, i]
            slice_mask = mask_vol[:, :, i]

            slice_image_centered, (pad_h, pad_w) = center_pad_single_slice(slice_image)
            slice_mask = center_pad_single_slice_by_params(slice_mask, pad_h, pad_w)

            slice_image_resized = resize_image(slice_image_centered, target_size=[240, 240])
            slice_mask = resize_image(slice_mask, target_size=[240, 240])

            if slice_mask.sum() < 3:
                continue

            plt.imsave(os.path.join(DIR_OUTPUT, "test", "Ungood", "img", f"{id_}_{i}.png"), slice_image_resized, cmap="bone")
            plt.imsave(os.path.join(DIR_OUTPUT, "test", "Ungood", "label", f"{id_}_{i}.png"), slice_mask, cmap="bone")
    logger.info("Finished processing test abnormal scans for Ungood slices.")