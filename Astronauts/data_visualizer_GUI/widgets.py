# -*- coding: utf-8 -*-
"""Widgets partages entre les pages."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

MASK_RGBA = (255, 0, 0, 128)  # rouge, alpha 0,5
NOT_COMPUTED = "non calculé"


class VideoOverlay(QtWidgets.QWidget):
    """Video (T, H, W) uint8 + masque (T, H, W) bool en rouge alpha 0,5.

    Case « afficher le masque », lecture/pause, vitesse, curseur. ``label_fn(i)``
    donne le texte affiche a cote du curseur pour la frame ``i``.
    """

    frameChanged = QtCore.pyqtSignal(int)

    def __init__(self, mask_label="Afficher le masque de la choroide", fps=15, parent=None):
        super().__init__(parent)
        self.frames = self.mask = None
        self.label_fn = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        ctrl = QtWidgets.QHBoxLayout()
        self.cb_mask = QtWidgets.QCheckBox(mask_label)
        self.cb_mask.setChecked(True)
        self.cb_mask.toggled.connect(self.redraw)
        self.btn_play = QtWidgets.QPushButton("Lecture")
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._toggle_play)
        self.sp_fps = QtWidgets.QSpinBox()
        self.sp_fps.setRange(1, 60)
        self.sp_fps.setValue(fps)
        self.sp_fps.setSuffix(" i/s")
        self.sp_fps.valueChanged.connect(lambda v: self.timer.setInterval(int(1000 / v)))
        self.lbl = QtWidgets.QLabel()
        self.lbl.setMinimumWidth(220)
        for w in (self.cb_mask, self.btn_play, QtWidgets.QLabel("Vitesse"), self.sp_fps, self.lbl):
            ctrl.addWidget(w)
        self.extra = QtWidgets.QHBoxLayout()  # controles propres a la page
        ctrl.addLayout(self.extra)
        ctrl.addStretch(1)
        lay.addLayout(ctrl)

        self.glw = pg.GraphicsLayoutWidget()
        self.vb = self.glw.addViewBox(lockAspect=True, invertY=True)
        self.img, self.overlay = pg.ImageItem(), pg.ImageItem()
        self.vb.addItem(self.img)
        self.vb.addItem(self.overlay)
        self.msg = pg.TextItem("", color=(200, 200, 200), anchor=(0.5, 0.5))
        self.vb.addItem(self.msg)
        lay.addWidget(self.glw, stretch=1)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.redraw)
        self.slider.valueChanged.connect(self.frameChanged.emit)
        lay.addWidget(self.slider)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(int(1000 / fps))
        self.timer.timeout.connect(self._next)

    @property
    def index(self) -> int:
        return self.slider.value()

    def set_data(self, frames, mask=None, label_fn=None, keep_index=True):
        first = self.frames is None or self.frames.shape[1:] != frames.shape[1:]
        self.frames, self.mask, self.label_fn = frames, mask, label_fn
        self.msg.setText("")
        T = frames.shape[0]
        self.slider.blockSignals(True)
        self.slider.setRange(0, T - 1)
        self.slider.setValue(min(self.slider.value(), T - 1) if keep_index else 0)
        self.slider.blockSignals(False)
        self.cb_mask.setEnabled(mask is not None)
        self.redraw()
        if first:
            self.vb.autoRange(padding=0)

    def set_mask(self, mask):
        self.mask = mask
        self.cb_mask.setEnabled(mask is not None)
        self.redraw()

    def clear(self, message: str = ""):
        self.btn_play.setChecked(False)
        self.frames = self.mask = None
        self.img.clear()
        self.overlay.clear()
        self.lbl.setText("")
        self.msg.setText(message)
        if message:
            r = self.vb.viewRect()
            self.msg.setPos(r.center())

    def redraw(self, *_):
        if self.frames is None:
            return
        i = self.slider.value()
        self.img.setImage(self.frames[i], levels=(0, 255), autoLevels=False)
        if self.mask is not None and self.cb_mask.isChecked():
            m = self.mask[i]
            rgba = np.zeros((*m.shape, 4), dtype=np.uint8)
            rgba[m] = MASK_RGBA
            self.overlay.setImage(rgba, levels=(0, 255), autoLevels=False)
            self.overlay.setVisible(True)
        else:
            self.overlay.setVisible(False)
        extra = f" | {self.label_fn(i)}" if self.label_fn else ""
        self.lbl.setText(f"{i + 1}/{self.frames.shape[0]}{extra}")

    def _toggle_play(self, on):
        self.btn_play.setText("Pause" if on else "Lecture")
        (self.timer.start if on else self.timer.stop)()

    def _next(self):
        if self.frames is not None:
            self.slider.setValue((self.slider.value() + 1) % self.frames.shape[0])
