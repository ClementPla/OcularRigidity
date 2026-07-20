from pathlib import Path
from ocularrigidity.consts import ROOT_DATA_SMB
import numpy as np
from ocularrigidity.data.io import save_mask
from ocularrigidity.segmentation.utils import get_choroid_segmentation_model
from ocularrigidity.segmentation.inference import infer
from ocularrigidity.data.compression import mp4_to_cube, read_gray
from ocularrigidity.data.io import load_cube

path_general = Path("E:/SANSORI")

for path_astro in path_general.iterdir()  :
    if path_astro.is_dir() :
        for path_moment in path_astro.iterdir() :
            if path_moment.match("*rigidity") :
                for path_condi in path_moment.iterdir() :
                    print(path_condi)
                    # path_video = Path(path_condi / "RawImages" / "registeredBscans" / "compressedOCT_woOutliers_median_threshold.mj2")
                    # OUTPUT_MASK_PATH = Path(path_condi / "RawImages" / "registeredBscans" / "mask.npz")
                    path_video = Path(path_condi / "RawImages" / "oneCycle_regAveBin" / "aveBinV4.mj2")
                    OUTPUT_MASK_PATH = Path(path_condi / "RawImages" / "oneCycle_regAveBin" / "mask_oneCycle.npz")
                    if path_video.exists() and (not OUTPUT_MASK_PATH.exists()) :
                        try :
                            data = read_gray(path_video )  # This time, you need to provide the full path to the video, not just the folder.
                            model = (get_choroid_segmentation_model())  # Model is automatically downloaded on first call.
                            mask = infer(model, data, scale_factor=(2), batch_size=16, device="cuda:0")
                            save_mask(mask, OUTPUT_MASK_PATH)  # Use packed + zstd compressed, so very light
                        except Exception as e:
                            print(f"Error occurred while processing {path_video}: {e}")