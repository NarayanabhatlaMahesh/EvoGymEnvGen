import cv2
import numpy as np


def normalize_frame(img, target_w, target_h):

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    elif img.shape[2] == 4:
        img = img[:, :, :3]

    # Correct comparison
    if img.shape[1] != target_w or img.shape[0] != target_h:
        img = cv2.resize(
            img,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

    return img
