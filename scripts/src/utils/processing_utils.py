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

def center_pad_to_square_by_slice(image):
    if len(image.shape) == 2:
        return center_pad_single_slice(image)
    elif len(image.shape) == 3:
        h, w, num_slices = image.shape
        max_size = max(h, w)
        square_volume = np.zeros((max_size, max_size, num_slices), dtype=image.dtype)

        for slice_idx in range(num_slices):
            current_slice = image[:, :, slice_idx]
            square_slice = center_pad_single_slice(current_slice)
            square_volume[:, :, slice_idx] = square_slice
        return square_volume
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")

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

def center_pad_to_square(image):
    h, w = image.shape[:2]
    max_size = max(h, w)
    
    pad_h = (max_size - h) // 2
    pad_w = (max_size - w) // 2
    
    if len(image.shape) == 2:
        square_image = np.zeros((max_size, max_size), dtype=image.dtype)
        square_image[pad_h:pad_h+h, pad_w:pad_w+w] = image
    else:
        square_image = np.zeros((max_size, max_size, image.shape[2]), dtype=image.dtype)
        square_image[pad_h:pad_h+h, pad_w:pad_w+w, :] = image
    
    return square_image

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