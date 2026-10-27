# -*- coding: utf-8 -*-
"""
Crops de choroide OCT pour SiNC (non supervise) -- l'equivalent de ``PURE_unsupervised``.

Entree : ``datasets/metadata/OCT.csv`` (``preprocessing/OCT_choroid/make_metadata.py``),
une ligne par video avec ``path`` vers ``crop.npz`` (``make_dataset.py``) et ``split``.

Un echantillon = (video, debut de clip, patch lateral) :
* clip de ``fpc`` frames, pas de ``step`` ; un clip contenant une frame invalide
  (``frame_valid == False``) est exclu ;
* patch de 64 x 64 a la resolution NATIVE : les 60 lignes du crop sont completees a 64
  par replication du bord, et les 492 colonnes donnent 7 patchs cote a cote (centres).
  A l'entrainement, chaque patch est decale au hasard dans la marge laterale.
Le reseau garde ainsi sa tete 64 x 64 ; reduire 492 colonnes a 64 moyennerait ~8
A-scans par pixel et effacerait la texture vasculaire.

Augmentations (entrainement) : s = vitesse (``speed_slow``-``speed_fast``, interpolation
temporelle, le facteur est renvoye pour decaler la bande des pertes), f = miroir
lateral, r = inversion du temps, g = bruit gaussien (sigma 0,05 apres z-score).
Chaque clip est centre-reduit (z-score sur T, H, W) : les intensites des crops sont des
niveaux de gris corriges, pas des [0, 255]. ``--input_norm diffnorm`` donne a la place
l'entree DiffNormalized de rPPG-Toolbox (``normalize_clip``).

Pas de reechantillonnage entre videos : la selection ne garde qu'une cadence (9,52 ou
9,62 Hz) et ``self.fps`` est leur moyenne, utilisee par les pertes spectrales.
"""

import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset

PATCH = 64
GAUSS_SIGMA = 0.05
DIFF_DENOM_FLOOR = 0.2  # fraction de 2 x la moyenne du clip


def normalize_clip(clip, mode):
    """``clip`` (C, T, H, W) -> meme forme.

    zscore   : centre-reduit sur tout le clip (intensite).
    diffnorm : ``diff_normalize_data`` de rPPG-Toolbox, l'entree des poids PhysNet :
               (x[t+1] - x[t]) / (x[t+1] + x[t]), divise par l'ecart-type, 0 en
               derniere frame. Le denominateur est BORNE par le bas a
               DIFF_DENOM_FLOOR x 2 x la moyenne du clip : nos intensites corrigees
               approchent 0 (et passent dessous) dans les vaisseaux.
    """
    if mode == "zscore":
        return (clip - clip.mean()) / (clip.std() + 1e-6)
    floor = DIFF_DENOM_FLOOR * 2 * clip.mean().abs() + 1e-6
    den = (clip[:, 1:] + clip[:, :-1]).clamp(min=floor)
    d = (clip[:, 1:] - clip[:, :-1]) / den
    d = d / (d.std() + 1e-6)
    return torch.cat([d, torch.zeros_like(clip[:, :1])], dim=1)


class OCTUnsupervised(VisionDataset):
    def __init__(self, split, arg_obj):
        super().__init__(split)
        self.split = split.lower()
        self.frames_per_clip = int(arg_obj.fpc)
        self.step = int(arg_obj.step)
        self.aug = arg_obj.augmentation.lower() if self.split == "train" else ""
        self.speed_slow = float(arg_obj.speed_slow)
        self.speed_fast = float(arg_obj.speed_fast)
        self.debug = bool(int(arg_obj.debug))
        self.input_norm = getattr(arg_obj, "input_norm", "zscore").lower()
        if self.input_norm not in ("zscore", "diffnorm"):
            raise ValueError(f"input_norm inconnu : {self.input_norm!r}")

        self.load_data(arg_obj.metadata)
        self.build_samples()
        print(self.split)
        print("Videos: ", len(self.data), "| fps: %.3f" % self.fps)
        print("Samples: ", self.samples.shape)
        print("Total frames: ", self.samples.shape[0] * self.frames_per_clip)

    # ------------------------------------------------------------------ #
    def load_data(self, metadata_path):
        meta = pd.read_csv(metadata_path)
        if self.split != "all":
            meta = meta[meta["split"] == self.split]
        if meta.empty:
            print(f"Aucune video pour le split {self.split!r} dans {metadata_path}. Exiting.")
            sys.exit(-1)
        if self.debug:
            meta = meta.iloc[:2]
        self.meta = meta.reset_index(drop=True)
        self.data = []
        for _, row in self.meta.iterrows():
            npz = np.load(row["path"])
            video = npz["video"]  # (T, depth, W) float16
            pad = PATCH - video.shape[1]
            if pad > 0:
                video = np.pad(video, ((0, 0), (0, pad), (0, 0)), mode="edge")
            self.data.append({
                "slug": row["slug"], "video": video,
                "frame_valid": npz["frame_valid"].astype(bool),
                "fps": float(npz["fps"]), "t_s": npz["t_s"],
            })
        self.fps = float(np.mean([d["fps"] for d in self.data]))
        # Compatibilite avec les utilitaires de SiNC (aucune onde de reference ici).
        self.waves = [np.zeros(len(d["video"])) for d in self.data]

    def patch_offsets(self, W):
        n = W // PATCH
        x0 = (W - n * PATCH) // 2
        return x0, n

    def build_samples(self):
        samples = []
        for v, d in enumerate(self.data):
            T, _, W = d["video"].shape
            _, n_patch = self.patch_offsets(W)
            bad = np.flatnonzero(~d["frame_valid"])
            for start in range(0, T - self.frames_per_clip + 1, self.step):
                end = start + self.frames_per_clip
                if np.any((bad >= start) & (bad < end)):
                    continue
                for k in range(n_patch):
                    samples.append((v, start, k))
        self.samples = np.asarray(samples, dtype=np.int64).reshape(-1, 3)

    # ------------------------------------------------------------------ #
    def max_length(self, d, start):
        """Frames utilisables depuis ``start`` sans traverser une frame invalide."""
        T = len(d["video"])
        bad = np.flatnonzero(~d["frame_valid"][start:])
        return int(bad[0]) if bad.size else T - start

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        v, start, k = (int(x) for x in self.samples[idx])
        d = self.data[v]
        T, H, W = d["video"].shape
        x0, n_patch = self.patch_offsets(W)
        fpc = self.frames_per_clip

        # Patch lateral, decale au hasard dans la marge a l'entrainement.
        c0 = x0 + k * PATCH
        if "train" == self.split and x0 > 0:
            c0 = int(np.clip(c0 + np.random.randint(-x0, x0 + 1), 0, W - PATCH))

        speed = 1.0
        n_src = fpc
        if "s" in self.aug:
            c_max = min(self.speed_fast, self.max_length(d, start) / fpc)
            speed_c = np.random.uniform(min(self.speed_slow, c_max), c_max)
            n_src = max(2, int(round(fpc * speed_c)))
            speed = n_src / fpc
        clip = d["video"][start:start + n_src, :, c0:c0 + PATCH].astype(np.float32)  # (t, H, P)
        clip = torch.from_numpy(clip)[None, None]  # (1, 1, t, H, P)
        if n_src != fpc:
            clip = F.interpolate(clip, (fpc, H, PATCH), mode="trilinear", align_corners=True)
        clip = clip[0]  # (1, T, H, P)

        if "f" in self.aug and np.random.rand() > 0.5:
            clip = torch.flip(clip, dims=(3,))
        if "r" in self.aug and np.random.rand() > 0.5:
            clip = torch.flip(clip, dims=(1,))
        clip = normalize_clip(clip, self.input_norm)
        if "g" in self.aug:
            clip = clip + GAUSS_SIGMA * torch.randn_like(clip)

        idcs = np.linspace(start, start + n_src - 1, fpc)
        if self.split == "train":
            return clip.float(), v, idcs, speed
        return clip.float(), v, idcs
