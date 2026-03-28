import numpy as np
import nibabel as nib
import cv2
import os



def load_nifti_image(image_path):
    img = nib.load(image_path)
    return img.get_fdata()

def apply_mask(mr_image, mask):
    if mr_image.shape != mask.shape:
        raise ValueError(f"Unmatched Image and shape {mr_image.shape} vs {mask.shape}")
    return mr_image * (mask > 0).astype(mr_image.dtype)

def center_pad_single_slice(image):
    h, w = image.shape
    max_size = max(h, w)
    
    pad_h = (max_size - h) // 2
    pad_w = (max_size - w) // 2
    
    square_slice = np.zeros((max_size, max_size), dtype=image.dtype)
    square_slice[pad_h:pad_h+h, pad_w:pad_w+w] = image

    return square_slice, (pad_h, pad_w)

def center_pad_single_slice_by_params(image, pad_h, pad_w):
    h, w = image.shape
    max_size = max(h, w)

    square_slice = np.zeros((max_size, max_size), dtype=image.dtype)
    square_slice[pad_h:pad_h+h, pad_w:pad_w+w] = image
    return square_slice

def resize_image(image, target_size=[240, 240]):
    """Resize the image to the target size."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST_EXACT)

def minmax_normalize_numpy(volume, clip_range=(0, 2000)):
    v = volume.astype(np.float32)
    v = v.clip(*clip_range)
    v_min, v_max = np.min(v), np.max(v)
    if v_max > v_min:  # avoid divide by zero
        v = (v - v_min) / (v_max - v_min) * 255
    else:
        v = np.zeros_like(v)
    return v.astype(np.uint8)

def save_np_to_nifti(array: np.ndarray, filepath: str, affine: np.ndarray | None = None) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if affine is None:
        # TODO: can add metadata later
        affine = np.eye(4)

    nifti_img = nib.Nifti1Image(array.astype(np.float32), affine)
    nib.save(nifti_img, filepath)
    return True

def get_ids_from_ungood_test_folder(output_dir):
    """
    Look into DIR_OUTPUT/test/Ungood/img and infer unique patient IDs
    from filenames like '<ID>_<slice>.nii', '.nii.gz', or '.png'.
    """
    img_dir = os.path.join(output_dir, "test", "Ungood", "img")
    if not os.path.isdir(img_dir):
        return set()

    ids = set()
    for fname in os.listdir(img_dir):
        # accept NIfTI and PNG images
        if not fname.endswith((".nii", ".nii.gz", ".png")):
            continue

        stem = fname
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        elif stem.endswith(".nii"):
            stem = stem[:-4]
        elif stem.endswith(".png"):
            stem = stem[:-4]

        parts = stem.split("_")
        if len(parts) < 2:
            continue
        pid = "_".join(parts[:-1])
        ids.add(pid)

    return ids

def center_crop(slice_, target_size = (224, 224)):
    """Crop the center region of a slice."""
    h, w = slice_.shape
    th, tw = target_size
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return slice_[i:i + th, j:j + tw]

def load_scan(dir_pelvis, det, id_, thresh_mr_mask=0.1):
    """Load MR, CT, and mask volumes and return both raw and normalized MR + body mask."""
    dir_scan = os.path.join(dir_pelvis, id_)
    mr = load_nifti_image(os.path.join(dir_scan, "mr.nii.gz"))
    ct = load_nifti_image(os.path.join(dir_scan, "ct.nii.gz"))
    mask = load_nifti_image(os.path.join(dir_scan, "mask.nii.gz"))

    # Get body mask from MR and mask, then apply it to MR and normalize
    body_mask_vol = det.get_body_mask_threshold(mr * mask, threshold_ct_body_mask=thresh_mr_mask)
    body_mask_vol = np.logical_and(body_mask_vol > 0, mask > 0).astype(np.uint8)
    masked_mr = apply_mask(mr, body_mask_vol)
    mr_norm = minmax_normalize_numpy(masked_mr)

    return mr, mr_norm, ct, body_mask_vol