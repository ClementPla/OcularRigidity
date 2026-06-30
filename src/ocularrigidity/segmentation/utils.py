from ocularrigidity.segmentation.trainer.pl_module import ChoroidSegmentationModule


def get_choroid_segmentation_model():
    """Automatically fetch the choroid segmentation model's weights from Hugging Face.

    Returns:
        ChoroidSegmentationModule: The segmentation model, loaded with pretrained weights from Hugging Face.
    """
    return ChoroidSegmentationModule.from_pretrained(
        "ClementP/ChoroidSegmentationModule"
    ).eval()
