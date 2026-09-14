from ocularrigidity.registration.deep_learning.models.regressor import (
    RegistrationRegressor,
)
from ocularrigidity.segmentation.trainer.pl_module import ChoroidSegmentationModule


def get_choroid_segmentation_model():
    """Automatically fetch the choroid segmentation model's weights from Hugging Face.

    Returns:
        ChoroidSegmentationModule: The segmentation model, loaded with pretrained weights from Hugging Face.
    """
    return ChoroidSegmentationModule.from_pretrained(
        "ClementP/ChoroidSegmentationModule", revision="version-2.0.0"
    ).eval()


def get_registration_model():
    """Automatically fetch the registration model's weights from Hugging Face.

    Returns:
        RegistrationRegressor: The registration model, loaded with pretrained weights from Hugging Face.
    """
    return RegistrationRegressor.from_pretrained(
        "ClementP/OCTVideoRegistration", revision="cascade_v9"
    ).eval()
