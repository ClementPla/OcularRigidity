# -*- coding: utf-8 -*-
"""
Encodeur du PhysNet de rPPG-Toolbox (``PhysNetRPPG.py``), pour un entrainement SiNC.

On part de ``PhysNet_padding_Encoder_Decoder_MAX`` et :

* on garde l'encodeur, ``ConvBlock1`` a ``ConvBlock9``, avec ses pools spatiaux ;
* les deux ``MaxpoolSpaTem`` (2, 2, 2) deviennent (1, 2, 2) : le temps reste a T. Sinon
  la sortie serait a T/4, soit 2,4 Hz pour nos videos a 9,57 Hz (Nyquist 72 BPM, sous
  une partie des FC). Un pool n'a pas de poids : rien n'est perdu ;
* le decodeur (``upsample``, ``upsample2``) est abandonne ;
* la tete est un pool spatial moyen puis ``ConvBlock10`` (1x1x1, 64 -> 1), initialisee
  par ses poids preentraines. Sortie ``(B, T)``, comme le PhysNet de SiNC ;
* avec ``input_channels=1``, la premiere convolution recoit la SOMME de ses poids RGB :
  c'est exactement le reseau d'origine applique a une image grise repliquee 3 fois.

``pretrained=None`` donne la meme architecture initialisee au hasard (controle).
"""

import torch
import torch.nn as nn

from .PhysNetRPPG import PhysNet_padding_Encoder_Decoder_MAX

ENCODER_PREFIXES = tuple(f"ConvBlock{i}." for i in range(1, 10)) + ("ConvBlock10.",)


class PhysNetRPPGEncoder(nn.Module):
    def __init__(self, input_channels=1, pretrained=None, drop_p=0.0):
        super().__init__()
        full = PhysNet_padding_Encoder_Decoder_MAX()
        if pretrained:
            state = torch.load(pretrained, map_location="cpu", weights_only=False)
            state = {k.removeprefix("module."): v for k, v in state.items()}
            missing, _ = full.load_state_dict(state, strict=False)
            needed = [k for k in full.state_dict() if k.startswith(ENCODER_PREFIXES)]
            absent = [k for k in needed if k in missing]
            if absent:
                raise KeyError(f"poids d'encodeur absents de {pretrained} : {absent}")
        self.pretrained = str(pretrained) if pretrained else None

        for i in range(1, 10):
            setattr(self, f"ConvBlock{i}", getattr(full, f"ConvBlock{i}"))
        self.ConvBlock10 = full.ConvBlock10
        if input_channels != 3:
            conv = self.ConvBlock1[0]
            new = nn.Conv3d(input_channels, conv.out_channels, conv.kernel_size,
                            stride=conv.stride, padding=conv.padding)
            with torch.no_grad():
                w = conv.weight.sum(dim=1, keepdim=True)  # (16, 1, 1, 5, 5)
                new.weight.copy_(w.repeat(1, input_channels, 1, 1, 1) / input_channels)
                new.bias.copy_(conv.bias)
            self.ConvBlock1[0] = new
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.poolspa = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.drop = nn.Dropout3d(drop_p) if drop_p > 0 else nn.Identity()

    def encoder_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("ConvBlock10.")]

    def head_parameters(self):
        return list(self.ConvBlock10.parameters())

    def forward(self, x):  # (B, C, T, H, W)
        x = self.MaxpoolSpa(self.ConvBlock1(x))
        x = self.ConvBlock3(self.ConvBlock2(x))
        x = self.MaxpoolSpa(x)  # etait MaxpoolSpaTem
        x = self.ConvBlock5(self.ConvBlock4(x))
        x = self.MaxpoolSpa(x)  # etait MaxpoolSpaTem
        x = self.ConvBlock7(self.ConvBlock6(x))
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock9(self.ConvBlock8(x))
        x = self.drop(x)
        x = self.ConvBlock10(self.poolspa(x))  # (B, 1, T, 1, 1)
        return torch.flatten(x, start_dim=1)  # (B, T)
