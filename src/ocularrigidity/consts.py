from pathlib import Path

MEASUREMENTS_PATH = Path("/home/clement/Documents/data/OcularRigidity/Biomechanics.db")

STUDY_PATH = Path("/home/clement/Documents/data/OcularRigidity/Studies.db")

OIMHS_ROOT = Path("/home/clement/Documents/data/OIMHS/Images/")


ROOT_DATA_SMB = "smb://192.168.11.16/database/dataFiles/"

ROOT_DATA_MNT = Path("/mnt/smb/")

ROOT_MASKS = Path("/media/clement/HD/Santiago/OcularRigidity/outputs/masks/")

ROOT_MEASURES = Path("/media/clement/HD/Santiago/OcularRigidity/outputs/measures/")

ROOT_COMPRESSED_VIDEO = Path(
    "/media/clement/HD/Santiago/OcularRigidity/outputs/compressed/"
)

ROOT_ONE_CYCLE = Path("/media/clement/HD/Santiago/OcularRigidity/outputs/one_cycle/")


CHECKPOINT_PATH = Path(
    "/home/clement/Documents/Projets/OcularRigidity/checkpoints/choroid-segmentation-epoch=29-dice=0.99.ckpt"
)
