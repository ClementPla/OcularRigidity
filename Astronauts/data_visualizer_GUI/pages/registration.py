# -*- coding: utf-8 -*-
"""PAGE 1 -- Recalage et segmentation : la video recalee, masque de choroide en surimpression."""

from __future__ import annotations

from pages.base import NOT_COMPUTED, Page
from widgets import VideoOverlay


class RegistrationPage(Page):
    title = "Recalage et segmentation"

    def build(self, layout):
        self.video = VideoOverlay()
        layout.addWidget(self.video, stretch=1)

    def show_result(self, ds, _res):
        T, H, W = ds.frames.shape
        t = ds.t
        self.video.set_data(ds.frames, ds.mask,
                            label_fn=(lambda i: f"t = {t[i]:.2f} s") if t is not None else None)
        if ds.mask is None:
            self.video.cb_mask.setText(f"Masque : {NOT_COMPUTED}")
        else:
            self.video.cb_mask.setText("Afficher le masque de la choroide")
        lines = [f"{ds.cond.name} | {T} frames {H}x{W}",
                 f"video : {ds.frames_path}", f"masque : {ds.mask_path}"]
        lines += [f"! {n}" for n in ds.notes]
        self.status.setText("\n".join(lines))

    def clear(self):
        self.video.clear()

    def not_computed(self, what):
        self.video.clear(NOT_COMPUTED)
