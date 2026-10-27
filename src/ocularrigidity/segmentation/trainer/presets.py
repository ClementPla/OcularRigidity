"""Named model configurations for the choroid segmentation sweep.

* SegFormer (Xie et al., NeurIPS 2021) at three MiT sizes: the current model
  and two bigger encoders.
* Other decoders on MiT-B2: U-Net, FPN, UPerNet (Xiao et al., ECCV 2018).
* Pure CNNs: U-Net and UPerNet on ConvNeXt-T (Liu et al., CVPR 2022), and U-Net
  on ResNet-34 / SE-ResNeXt-50 / EfficientNet-B4, the usual baselines for
  choroid segmentation in OCT. "nnU-Net Revisited" (Isensee et al., MICCAI
  2024) found that well-trained CNN U-Nets match or beat transformer models
  in medical segmentation.
* U-Net++ on EfficientNet-B4 (nested skip connections).

Models on a MiT-B1..B5 encoder keep the (64, 128, 320, 512) feature pyramid the
registration regressor is built on (``RegistrationRegressor(in_channels=...)``,
via ``encoder.forward_features``); the others would need an adapter there.
"""

# preset -> train.py options (argparse dest names)
MODELS: dict[str, dict] = {
    "segformer_mit_b1": {"arch": "segformer", "encoder": "mit_b1"},
    "segformer_mit_b2": {"arch": "segformer", "encoder": "mit_b2"},
    "segformer_mit_b3": {"arch": "segformer", "encoder": "mit_b3"},
    "unet_mit_b2": {"arch": "unet", "encoder": "mit_b2"},
    "fpn_mit_b2": {"arch": "fpn", "encoder": "mit_b2"},
    "upernet_mit_b2": {"arch": "upernet", "encoder": "mit_b2"},
    "unet_convnext_t": {"arch": "unet", "encoder": "tu-convnext_tiny"},
    "upernet_convnext_t": {"arch": "upernet", "encoder": "tu-convnext_tiny"},
    "unet_resnet34": {"arch": "unet", "encoder": "resnet34"},
    "unet_se_resnext50": {"arch": "unet", "encoder": "se_resnext50_32x4d"},
    "unet_effnet_b4": {"arch": "unet", "encoder": "efficientnet-b4"},
    # The nested decoder is memory hungry at 1024x768: same effective batch
    "unetpp_effnet_b4": {
        "arch": "unetplusplus",
        "encoder": "efficientnet-b4",
        "batch_size": 4,
        "accumulate": 2,
    },
}


def registration_ok(encoder: str) -> bool:
    """Whether the registration regressor can sit on this encoder unchanged."""
    return encoder in {f"mit_b{i}" for i in range(1, 6)}
