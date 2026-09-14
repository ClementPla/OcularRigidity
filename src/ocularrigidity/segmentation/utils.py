from ocularrigidity.segmentation.trainer.pl_module import ChoroidSegmentationModule

DEFAULT_SEGMENTATION_REPO = "ClementP/ChoroidSegmentationModule"


def get_choroid_segmentation_model(
    repo_id: str = DEFAULT_SEGMENTATION_REPO,
    revision: str | None = None,
):
    """Automatically fetch the choroid segmentation model's weights from Hugging Face.

    Args:
        repo_id (str, optional): Hugging Face repository holding the weights.
        revision (str | None, optional): Branch or tag to load. ``None`` (the
            default) loads the repository's default branch, i.e. the historical
            U-Net ``se_resnet50``. ``"version-2.0.0"`` loads the SegFormer
            ``mit_b2`` weights instead -- a different architecture, so a mask
            produced with it is NOT comparable to a mask produced without it.

    Returns:
        ChoroidSegmentationModule: The segmentation model, loaded with pretrained weights from Hugging Face.
    """
    kwargs = {"revision": revision} if revision else {}
    return ChoroidSegmentationModule.from_pretrained(repo_id, **kwargs).eval()
