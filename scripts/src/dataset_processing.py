import os
import sys
import json

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
    center_pad_single_slice_by_params,
    resize_image,
    minmax_normalize_numpy,
    save_np_to_nifti
)

from utils.path_utils import create_output_dirs


DIR_PELVIS = os.path.join(os.getcwd(), "data", "Task1", "pelvis")
DIR_OUTPUT = os.path.join(os.getcwd(), "output", "nifti")
DIR_LABELS = os.path.join(os.getcwd(), "labels")
FILE_MISSING = os.path.join(DIR_LABELS, "missing_Ömer.txt")
LABELS_RAW = os.path.join(DIR_LABELS, "labels_raw.json")

NYUL_MIN_VALUE = 0
NYUL_MAX_VALUE = 255
NYUL_MIN_PERCENTILE = 1
NYUL_MAX_PERCENTILE = 99


DELTA=250
# THRESH_MR_MASK = 15
THRESH_MR_MASK = 0.1





if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    df_overview = pd.read_excel("/home/user/lyeyang/projects/sct-ood-dataset/data/Task1/pelvis/overview/1_pelvis_train.xlsx",
                            sheet_name="MR",)
    
    ids_all = df_overview["ID"].tolist()
    ids_all = [i for i in ids_all if i.startswith("1PA")]

    with open("/home/user/lyeyang/projects/sct-ood-dataset/labels/labels_raw.json") as f:
        data = json.load(f)

    data_abnormal = data['type1']

    ids_abnormal = [list(item.keys())[0] for item in data['type1']]
    ids_abnormal = [i for i in ids_abnormal if i.startswith("1PA")] # TODO:

    df_labels_1 = pd.DataFrame([
        {"id": k, **v} 
        for item in data_abnormal 
        for k, v in item.items()
    ])
    list_na_ids = df_labels_1[df_labels_1.isna().any(axis=1)]['id'].tolist()

    df_labels_1 = df_labels_1.dropna()
    df_labels_1 = df_labels_1[df_labels_1['body_part'] == 'pelvis']

    df_labels_1["anomaly_start"] = df_labels_1[["ct_start", "mr_start"]].min(axis=1)
    df_labels_1["anomaly_end"] = df_labels_1[["ct_end", "mr_end"]].max(axis=1)

    anomaly_range = {id: (int(start), int(end)) for id, start, end 
                    in zip(df_labels_1["id"], df_labels_1["anomaly_start"], df_labels_1["anomaly_end"])}

    ids_abnormal_all = df_labels_1['id'].tolist()
    with open("/home/user/lyeyang/projects/sct-ood-dataset/labels/missing_Ömer.txt") as f:
        ids_omer = f.read().splitlines()
    ids_not_included = list_na_ids + ids_omer
    ids_used = list(set(ids_all) - set(ids_not_included))
    ids_abnormal = list(set(ids_used) & set(ids_abnormal))
    ids_normal = list(set(ids_used) - set(ids_abnormal))

    ids_abnormal_valid = ["1PA133", "1PA136", "1PA169", "1PA178", '1PC015', '1PC037'] #  
    ids_normal_valid = ['1PA171', '1PA156', '1PA119', '1PA113', '1PA100', '1PA177', '1PC032', '1PC080', '1PC000'] # 

    ids_abnormal_test = list(set(ids_abnormal) - set(ids_abnormal_valid))

    # TODO: !!! fix for now
    # ids_normal_test = list(set(ids_normal)-set(ids_normal_valid))[::3]
    ids_normal_test = ['1PA062', '1PA081', '1PA084', '1PA086', '1PA088', '1PA090', '1PA093', '1PA098', '1PA101', '1PA105', '1PA109', '1PA112', '1PA116', '1PA141', '1PA145', '1PA148', '1PA150', '1PA181']
    # TODO: !!! fix for now
    # ids_normal_train = list(set(ids_normal) - set(ids_normal_valid) - set(ids_normal_test))
    ids_normal_train = ['1PA063', '1PA064', '1PA070', '1PA073', '1PA074', '1PA076', '1PA080', '1PA083', '1PA091', '1PA094', '1PA095', '1PA097', '1PA106', '1PA107', '1PA108', '1PA110', '1PA111', '1PA115', '1PA117', '1PA121', '1PA126', '1PA127', '1PA134', '1PA137', '1PA138', '1PA140', '1PA142', '1PA144', '1PA146', '1PA154', '1PA157', '1PA159', '1PA161', '1PA164', '1PA174', '1PA187']
    ids_abnormal_valid

    logger.info(f"Used scans: {len(ids_used)}")
    logger.info(f"Abnormal scans: {len(ids_abnormal)}")
    logger.info(f"Normal scans: {len(ids_normal)}")
    logger.info(f"Valid Abnormal scans: {len(ids_abnormal_valid)}")
    logger.info(f"Valid Normal scans: {len(ids_normal_valid)}")
    logger.info(f"Test Abnormal scans: {len(ids_abnormal_test)}")
    logger.info(f"Test Normal scans: {len(ids_normal_test)}")
    logger.info(f"Train Normal scans: {len(ids_normal_train)}")


    create_output_dirs(dir_output=DIR_OUTPUT)

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

        det = MetalArtifactDetector()
        body_mask = det.get_body_mask_threshold(mr_image * mask, threshold_ct_body_mask=THRESH_MR_MASK)
        body_mask = np.logical_and(body_mask>0, mask > 0)

        masked_mr = apply_mask(mr_image, body_mask)
        mr_normalized = minmax_normalize_numpy(masked_mr)
        
        slices = mr_image.shape[2]

        for i in tqdm(range(25, slices - 20, 1)):
            slice_image = masked_mr[:, :, i]
            slice_image_centered, (pad_h, pad_w) = center_pad_single_slice(slice_image)
            slice_image_resized = resize_image(slice_image_centered, target_size=[240, 240])

            save_np_to_nifti(slice_image_resized, os.path.join(DIR_OUTPUT, "train", "good", f"{id_}_{i}.nii"))
            
            # plt.imsave(os.path.join(DIR_OUTPUT, "train", "good", f"{id_}_{i}.png"), slice_image_resized, cmap="bone")
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
        body_mask = np.logical_and(body_mask>0, mask > 0)

        masked_mr = apply_mask(mr_image, body_mask)
        mr_normalized = minmax_normalize_numpy(masked_mr)
        
        slices = mr_image.shape[2]
        
        for i in tqdm(range(25, slices - 20, 1)):
            slice_image = masked_mr[:, :, i]
            slice_image_centered, (pad_h, pad_w) = center_pad_single_slice(slice_image)
            slice_image_resized = resize_image(slice_image_centered, target_size=[240, 240])
            slice_mask = np.zeros_like(slice_image_resized)

            save_np_to_nifti(slice_image_resized, os.path.join(DIR_OUTPUT, "valid", "good", "img", f"{id_}_{i}.nii"))
            save_np_to_nifti(slice_mask, os.path.join(DIR_OUTPUT, "valid", "good", "label", f"{id_}_{i}.nii"))

            # plt.imsave(os.path.join(DIR_OUTPUT, "valid", "good", "img", f"{id_}_{i}.png"), slice_image_resized, cmap="bone")
            # plt.imsave(os.path.join(DIR_OUTPUT, "valid", "good", "label", f"{id_}_{i}.png"), slice_mask, cmap="bone")
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
        body_mask = np.logical_and(body_mask>0, mask > 0)

        masked_mr = apply_mask(mr_image, body_mask)
        mr_normalized = minmax_normalize_numpy(masked_mr)

        slices = mr_image.shape[2]
        abnormal_slices = list(range(anomaly_range[id_][0], anomaly_range[id_][-1]))
        
        # Extract label masks
        df_hu = det.score_volume_hu(ct_image, scan_id=id_, slice_axis=2)
        df_hu["label"] = np.isin(df_hu["slice_idx"], abnormal_slices).astype(np.uint8)
        tau, info = det.pick_global_tau_by_hu(df_hu, label_col="label")
        # logger.info("tau:", tau, "| info:", info)
        logger.info("tau: %s | info: %s", tau, info)

        tau -= DELTA
        mask_vol = (ct_image >= tau).astype(np.uint8)

        abnormal_slices = [i for i in abnormal_slices if i >= 15 and i < slices - 15]
        # logger.info(id_, "abnormal slices:", abnormal_slices)
        logger.info("%s abnormal slices: %s", id_, abnormal_slices)


        lo_diff_val = 5
        up_diff_val = 10
        mask_vol_refined = det.refine_mask_with_mr(mask_vol, mr_image, lo_diff=lo_diff_val, up_diff=up_diff_val)
        mask_vol_refined = det.postprocess_mask_volume(mask_vol_refined, min_hole_size=20, smooth=True, disk_size=0)
        mask_vol = mask_vol_refined
        
        for i in tqdm(abnormal_slices):
            slice_image = masked_mr[:, :, i]
            slice_mask = mask_vol[:, :, i]
            
            slice_image_centered, (pad_h, pad_w) = center_pad_single_slice(slice_image)
            slice_mask = center_pad_single_slice_by_params(slice_mask, pad_h, pad_w)

            slice_image_resized = resize_image(slice_image_centered, target_size=[240, 240])
            slice_mask = resize_image(slice_mask, target_size=[240, 240])

            if slice_mask.sum()<3:
                # print(f"Skipping slice {i} for {id_} due to insufficient mask data.")
                continue
            
            save_np_to_nifti(slice_image_resized, os.path.join(DIR_OUTPUT, "valid", "Ungood", "img", f"{id_}_{i}.nii"))
            save_np_to_nifti(slice_mask, os.path.join(DIR_OUTPUT, "valid", "Ungood", "label", f"{id_}_{i}.nii"))
            # plt.imsave(os.path.join(DIR_OUTPUT, "valid", "Ungood", "img", f"{id_}_{i}.png"), slice_image_resized, cmap="bone")
            # plt.imsave(os.path.join(DIR_OUTPUT, "valid", "Ungood", "label", f"{id_}_{i}.png"), slice_mask, cmap="bone")

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

        det = MetalArtifactDetector()
        body_mask = det.get_body_mask_threshold(mr_image * mask, threshold_ct_body_mask=THRESH_MR_MASK)
        body_mask = np.logical_and(body_mask>0, mask > 0)

        masked_mr = apply_mask(mr_image, body_mask)
        mr_normalized = minmax_normalize_numpy(masked_mr)

        slices = mr_image.shape[2]

        for i in tqdm(range(25, slices - 20, 1)):
            slice_image = masked_mr[:, :, i]
            slice_image_centered, (pad_h, pad_w) = center_pad_single_slice(slice_image)
            slice_image_resized = resize_image(slice_image_centered, target_size=[240, 240])
            slice_mask = np.zeros_like(slice_image_resized)
            save_np_to_nifti(slice_image_resized, os.path.join(DIR_OUTPUT, "test", "good", "img", f"{id_}_{i}.nii"))
            save_np_to_nifti(slice_mask, os.path.join(DIR_OUTPUT, "test", "good", "label", f"{id_}_{i}.nii"))
            # plt.imsave(os.path.join(DIR_OUTPUT, "test", "good", "img", f"{id_}_{i}.png"), slice_image_resized, cmap="bone")
            # plt.imsave(os.path.join(DIR_OUTPUT, "test", "good", "label", f"{id_}_{i}.png"), slice_mask, cmap="bone")
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

        det = MetalArtifactDetector()
        body_mask = det.get_body_mask_threshold(mr_image * mask, threshold_ct_body_mask=THRESH_MR_MASK)
        body_mask = np.logical_and(body_mask>0, mask > 0)

        masked_mr = apply_mask(mr_image, body_mask)
        mr_normalized = minmax_normalize_numpy(masked_mr)
        
        slices = mr_image.shape[2]
        abnormal_slices = list(range(anomaly_range[id_][0], anomaly_range[id_][-1]))
        
        # Extract label masks
        
        df_hu = det.score_volume_hu(ct_image, scan_id=id_, slice_axis=2)
        df_hu["label"] = np.isin(df_hu["slice_idx"], abnormal_slices).astype(np.uint8)
        tau, info = det.pick_global_tau_by_hu(df_hu, label_col="label")
        print("tau:", tau, "| info:", info)
        tau -= DELTA
        mask_vol = (ct_image >= tau).astype(np.uint8)
        
        abnormal_slices = [i for i in abnormal_slices if i >= 15 and i < slices - 15]

        lo_diff_val = 5
        up_diff_val = 10
        mask_vol_refined = det.refine_mask_with_mr(mask_vol, mr_image, lo_diff=lo_diff_val, up_diff=up_diff_val)
        mask_vol_refined = det.postprocess_mask_volume(mask_vol_refined, min_hole_size=20, smooth=True, disk_size=0)
        mask_vol = mask_vol_refined
        
        for i in tqdm(abnormal_slices):
            slice_image = masked_mr[:, :, i]
            slice_mask = mask_vol[:, :, i]
            
            slice_image_centered, (pad_h, pad_w) = center_pad_single_slice(slice_image)
            slice_mask = center_pad_single_slice_by_params(slice_mask, pad_h, pad_w)

            slice_image_resized = resize_image(slice_image_centered, target_size=[240, 240])
            slice_mask = resize_image(slice_mask, target_size=[240, 240])

            if slice_mask.sum()<3:
                continue

            save_np_to_nifti(slice_image_resized, os.path.join(DIR_OUTPUT, "test", "Ungood", "img", f"{id_}_{i}.nii"))
            save_np_to_nifti(slice_mask, os.path.join(DIR_OUTPUT, "test", "Ungood", "label", f"{id_}_{i}.nii"))    
            # plt.imsave(os.path.join(DIR_OUTPUT, "test", "Ungood", "img", f"{id_}_{i}.png"), slice_image_resized, cmap="bone")
            # plt.imsave(os.path.join(DIR_OUTPUT, "test", "Ungood", "label", f"{id_}_{i}.png"), slice_mask, cmap="bone")
    logger.info("Finished processing test abnormal scans for Ungood slices.")