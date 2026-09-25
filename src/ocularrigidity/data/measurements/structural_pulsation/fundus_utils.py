
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

def get_line_orientation(file: Path | str | np.ndarray) -> float:
    """
    Get the orientation of a line in an image using Hough Transform.

    Args:
        file (Path | str | np.ndarray): Path to the image file or the image array.

    Returns:
        float: The orientation of the line in degrees.
    """
    if isinstance(file, (Path, str)):
        pil_img = Image.open(file).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        img = file
        
        
    bw_mask = (img[:, :, 0] == img[:, :, 1]) & (
            img[:, :, 1] == img[:, :, 2]
        )
        # Find the magenta line in the image
    magenta_line = (
        (img[:, :, 0] > 0)
        & (img[:, :, 1] < 200)
        & (img[:, :, 2] > 0)
        & (~bw_mask)
    )
    coords = cv2.findNonZero(magenta_line.astype("uint8"))
    coords = np.asarray(coords).reshape(-1, 2)

    # Fit a line
    fitted = np.polyfit(coords[:, 0], coords[:, 1], 1)
    # Get the angle of the line
    angle = np.arctan(fitted[0]) * 180 / np.pi
    # Cast the angle to the range [0, 180]
    angle = angle if angle >= 0 else angle + 180
    return angle