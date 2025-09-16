import numpy as np
import pandas as pd
import cv2

from skimage import morphology, measure
from scipy.ndimage import binary_erosion, binary_dilation

import matplotlib.pyplot as plt


class MetalArtifactDetector:
    def __init__(self, metric='f1'):
        """
        metric: ('f1', 'youden', 'balanced_acc', 'recall', 'precision', 'accuracy')
        """
        self.metric = metric
        self.tau_global = None
        self.tau_map = {}

    @staticmethod
    def _confusion_binary(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.uint8)
        y_pred = np.asarray(y_pred, dtype=np.uint8)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        return tp, fp, fn, tn

    @staticmethod
    def _metrics_from_cm(tp, fp, fn, tn, eps=1e-9):
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        tpr  = rec
        tnr  = tn / (tn + fp + eps)
        bal_acc = 0.5 * (tpr + tnr)
        youden  = tpr + tnr - 1.0
        acc     = (tp + tn) / (tp + fp + fn + tn + eps)
        return {
            "precision": prec, "recall": rec, "f1": f1,
            "balanced_acc": bal_acc, "youden": youden, "accuracy": acc
        }

    @staticmethod
    def _norm01(x, hu_window=None, clip_quantiles=None, eps=1e-6):
        x = np.asarray(x, dtype=float)
        if hu_window is not None:
            lo, hi = hu_window
            x = np.clip(x, lo, hi)
        elif clip_quantiles is not None:
            qlo, qhi = clip_quantiles
            lo = np.percentile(x, qlo*100.0)
            hi = np.percentile(x, qhi*100.0)
            x = np.clip(x, lo, hi)
        mn, mx = x.min(), x.max()
        if mx - mn < eps:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn + eps)

    def score_volume_hu(self, vol, scan_id="scan1", slice_axis=0):
        """return max HU for each slice"""
        vol_z = np.moveaxis(vol, slice_axis, 0)  # (Z,H,W)
        z, h, w = vol_z.shape
        slice_max = np.max(vol_z.reshape(z, -1), axis=1)
        return pd.DataFrame({
            "scan_id": scan_id,
            "slice_idx": np.arange(z),
            "slice_max_hu": slice_max.astype(float),
        })

    def pick_global_tau_by_hu(self, df, metal_slices=None,
                              label_col='label', hu_col='slice_max_hu'):
        """
        global_tau = max(max_normal_hu, min_max_abnormal_hu)
        """
        if metal_slices is None:
            if label_col not in df.columns:
                raise ValueError("Need metal_slices or df[label_col].")
            labels = df[label_col].values.astype(int)
            pos_idx = np.where(labels == 1)[0]
            neg_idx = np.where(labels == 0)[0]
        else:
            sl = df["slice_idx"].values
            pos_mask = np.isin(sl, np.asarray(metal_slices, dtype=int))
            pos_idx = np.where(pos_mask)[0]
            neg_idx = np.where(~pos_mask)[0]

        vals = df[hu_col].values.astype(float)

        max_normal_hu = float(vals[neg_idx].max()) if len(neg_idx) > 0 else -np.inf
        min_max_abnormal_hu = float(vals[pos_idx].min()) if len(pos_idx) > 0 else np.inf

        global_tau = max(max_normal_hu, min_max_abnormal_hu)
        self.tau_global = global_tau
        return global_tau, {
            "max_normal_hu": max_normal_hu,
            "min_max_abnormal_hu": min_max_abnormal_hu
        }

    def apply_tau_by_hu(self, df, hu_col='slice_max_hu'):
        """classify slices based on HU values"""
        if self.tau_global is None or not np.isfinite(self.tau_global):
            raise ValueError("Global tau not set. Run pick_global_tau_by_hu first.")
        out = df.copy()
        out["pred"] = (out[hu_col].values >= self.tau_global).astype(np.uint8)
        return out

    def evaluate(self, df, label_col='label', pred_col='pred', scan_col='scan_id'):
        # slice-level
        tp, fp, fn, tn = self._confusion_binary(df[label_col].values, df[pred_col].values)
        sl = self._metrics_from_cm(tp, fp, fn, tn)
        sl.update(dict(tp=tp, fp=fp, fn=fn, tn=tn))

        # scan-level
        scan_true = df.groupby(scan_col, sort=False)[label_col].max().values
        scan_pred = df.groupby(scan_col, sort=False)[pred_col].max().values
        tp, fp, fn, tn = self._confusion_binary(scan_true, scan_pred)
        sc = self._metrics_from_cm(tp, fp, fn, tn)
        sc.update(dict(tp=tp, fp=fp, fn=fn, tn=tn))
        return {"slice_level": sl, "scan_level": sc}


    def extract_mask_volume(self, vol, tau, slice_axis=0):
        return (vol >= tau).astype(np.uint8)

    def postprocess_mask(self, mask2d, min_hole_size=20, smooth=True, disk_size=3):
        mask = mask2d.astype(bool)

        mask = morphology.remove_small_holes(mask, area_threshold=min_hole_size)

        if smooth:
            selem = morphology.disk(disk_size)
            mask = morphology.binary_opening(mask, selem)
            mask = morphology.binary_closing(mask, selem)

        return mask.astype(np.uint8)

    def postprocess_mask_volume(self, mask_vol, slice_axis=0,
                                min_hole_size=20, smooth=True, disk_size=0):
        vol_z = np.moveaxis(mask_vol, slice_axis, 0)
        out_z = []
        for z in range(vol_z.shape[0]):
            out_z.append(self.postprocess_mask(vol_z[z],
                                               min_hole_size=min_hole_size,
                                               smooth=smooth,
                                               disk_size=disk_size))
        out = np.stack(out_z, axis=0)
        return np.moveaxis(out, 0, slice_axis)
    
    def show_ct_mask_mr(
        self, ct_vol, mr_vol, raw_mask_vol, post_mask_vol, slice_indices,
        body_mask = None,
        slice_axis=0, suptitle="CT (raw vs postprocessed mask) & MR",
        ct_hu_window=None, ct_clip_quantiles=None,
        mr_clip_quantiles=None,
        fill_alpha=0.6, outline_width=1.5,
        raw_color=(0.0, 0, 1.0),   # blue
        post_color=(0.0, 1.0, 0.0), # green
        outline_color="red",
        figsize=(12, 3)
    ):
        if isinstance(slice_indices, int):
            slice_indices = [slice_indices]

        ct = np.moveaxis(ct_vol, slice_axis, 0)
        mr = np.moveaxis(mr_vol, slice_axis, 0)
        if body_mask is not None:
            # ValueError: operands could not be broadcast together with shapes (149,428,277) (428,277,149) 
            mr = mr * np.moveaxis(body_mask, slice_axis, 0)

        raw_mask = np.moveaxis(raw_mask_vol, slice_axis, 0)
        post_mask = np.moveaxis(post_mask_vol, slice_axis, 0)

        print(ct.shape, mr.shape, raw_mask.shape, post_mask.shape)
        assert ct.shape == mr.shape == raw_mask.shape == post_mask.shape, "Shapes must match"

        n = len(slice_indices)
        fig, axs = plt.subplots(n, 3, figsize=(figsize[0], figsize[1] * n))

        if n == 1:
            axs = np.expand_dims(axs, 0)

        for i, z in enumerate(slice_indices):
            ct_img = self._norm01(ct[z], hu_window=ct_hu_window, clip_quantiles=ct_clip_quantiles)
            mr_img = self._norm01(mr[z], clip_quantiles=mr_clip_quantiles)
            raw_mask_img = raw_mask[z].astype(bool)
            post_mask_img = post_mask[z].astype(bool)

            ct_img  = np.rot90(ct_img,  k=-1)
            mr_img  = np.rot90(mr_img,  k=-1)
            raw_mask_img  = np.rot90(raw_mask_img,  k=-1)
            post_mask_img = np.rot90(post_mask_img, k=-1)

            # CT + raw mask
            axs[i, 0].imshow(ct_img, cmap="gray")
            overlay_raw = np.zeros((*raw_mask_img.shape, 4), dtype=float)
            r, g, b = raw_color
            overlay_raw[raw_mask_img] = [r, g, b, float(fill_alpha)]
            axs[i, 0].imshow(overlay_raw)
            contours = measure.find_contours(raw_mask_img, level=0.5)
            for contour in contours:
                axs[i, 0].plot(contour[:, 1], contour[:, 0], color=outline_color, linewidth=outline_width)
            axs[i, 0].set_title(f"Raw Mask+CT z={z}")
            axs[i, 0].axis("off")

            # CT + postprocessed mask
            axs[i, 1].imshow(ct_img, cmap="gray")
            overlay_post = np.zeros((*post_mask_img.shape, 4), dtype=float)
            r, g, b = post_color
            overlay_post[post_mask_img] = [r, g, b, float(fill_alpha)]
            axs[i, 1].imshow(overlay_post)
            contours = measure.find_contours(post_mask_img, level=0.5)
            for contour in contours:
                axs[i, 1].plot(contour[:, 1], contour[:, 0], color=outline_color, linewidth=outline_width)
            axs[i, 1].set_title(f"Post Mask+CT z={z}")
            axs[i, 1].axis("off")

            # MR
            axs[i, 2].imshow(mr_img, cmap="gray")
            axs[i, 2].set_title(f"MR z={z}")
            axs[i, 2].axis("off")

        plt.suptitle(suptitle, fontsize=14)
        plt.tight_layout()
        plt.show()

    def get_mask_biggest_contour(self, mask_ct):
        for i in range(mask_ct.shape[2]):
            inmask = np.expand_dims(mask_ct[:, :, i].astype(np.uint8), axis=2)
            ret, bin_img = cv2.threshold(inmask, 0.5, 1, cv2.THRESH_BINARY)
            (cnts, _) = cv2.findContours(np.expand_dims(bin_img, axis=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # return None, if no contours detected
            if len(cnts) != 0:
                # based on contour area, get the maximum contour which is a body contour
                segmented = max(cnts, key=cv2.contourArea)
                bin_img[bin_img > 0] = 0
                a = cv2.drawContours(np.expand_dims(bin_img, axis=2), [segmented], 0, (255, 255, 255), -1)
                a[a > 0] = 1
                mask_ct[:, :, i] = a.squeeze()
        return mask_ct.astype(np.uint8)

    def get_body_mask_threshold(self, nii_array, threshold_ct_body_mask):
        mask_ct = np.zeros(nii_array.shape)
        mask_ct[nii_array > threshold_ct_body_mask] = 1
        mask_ct[nii_array <= threshold_ct_body_mask] = 0
        mask_ct = binary_erosion(mask_ct, iterations=2).astype(np.uint8)
        mask_ct = self.get_mask_biggest_contour(mask_ct)
        mask_ct = binary_dilation(mask_ct, iterations=5).astype(np.int16)
        return mask_ct
    
    
    def refine_mask_with_mr(self, ct_mask_vol, mr_vol, lo_diff=10, up_diff=150, min_contour_area=5):
        """
        Refines a 3D CT mask volume using a corresponding MR volume by performing
        a flood-fill operation on each slice. Handles small, single-pixel anomalies
        by preserving the original CT mask.

        Args:
            ct_mask_vol (np.ndarray): The 3D volume of the initial CT masks (binary, uint8).
            mr_vol (np.ndarray): The 3D volume of the MR images.
            lo_diff (int): The lower boundary difference from the seed pixel's value.
            up_diff (int): The upper boundary difference from the seed pixel's value.
                           These values determine the sensitivity to gradient changes.
            min_contour_area (int): The minimum area of a contour to trigger the flood fill.
                                    If a contour is smaller than this, the original mask is used.

        Returns:
            np.ndarray: The refined 3D mask volume (binary, uint8).
        """
        assert ct_mask_vol.shape == mr_vol.shape, "CT mask and MR must have the same shape"
        
        # Create an empty volume to store the refined masks
        refined_mask = np.zeros_like(ct_mask_vol, dtype=np.uint8)

        # Iterate over each slice in the 3D volumes
        for i in range(ct_mask_vol.shape[2]):
            slice_ct = ct_mask_vol[:, :, i].astype(np.uint8)
            slice_mr = mr_vol[:, :, i]

            # If there's no mask on this slice, skip it
            if slice_ct.sum() == 0:
                continue

            # Find all contours on this slice
            contours, _ = cv2.findContours(slice_ct, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue

            # This will store the combined result of all flood fills and preserved masks
            refined_mask_slice = np.zeros_like(slice_ct)

            for cnt in contours:
                contour_area = cv2.contourArea(cnt)
                
                # Check if the contour is too small for flood fill
                if contour_area < min_contour_area:
                    # If it's too small, just keep the original mask for this area
                    mask_cnt = np.zeros(slice_ct.shape, dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 1, -1)
                    refined_mask_slice = np.logical_or(refined_mask_slice, mask_cnt).astype(np.uint8)
                    continue

                # Calculate the centroid (x, y) coordinates for this contour
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # --- Perform the flood-fill operation for this specific contour ---
                flood_fill_slice = slice_mr.copy().astype(np.float32)

                # Create an empty mask to store the result of the flood fill.
                h, w = flood_fill_slice.shape
                mask = np.zeros((h + 2, w + 2), np.uint8)

                cv2.floodFill(flood_fill_slice, mask, (cx, cy), 255, lo_diff, up_diff, cv2.FLOODFILL_MASK_ONLY)

                # Remove the extra border from the mask to get the final refined mask for the slice.
                current_fill = mask[1:-1, 1:-1]
                
                # Combine the new flood fill result with the existing refined mask for the slice
                refined_mask_slice = np.logical_or(refined_mask_slice, current_fill).astype(np.uint8)

            refined_mask[:, :, i] = refined_mask_slice

        return refined_mask