import cv2
import numpy as np
import skimage


def morph_open(masks, kernel_size=3, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for i in range(masks.shape[0]):
        masks[i] = cv2.morphologyEx(
            masks[i].astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=iterations
        )
    return masks

def morph_close(masks, kernel_size=3, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for i in range(masks.shape[0]):
        masks[i] = cv2.morphologyEx(
            masks[i].astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=iterations
        )
    return masks



def remove_small_objects(masks, min_size=100):
    for i in range(masks.shape[0]):
        objects = skimage.measure.label(masks[i], connectivity=2)
        large_objects = skimage.morphology.remove_small_objects(
            objects, min_size=min_size
        )
        masks[i] = large_objects != 0

    return masks
