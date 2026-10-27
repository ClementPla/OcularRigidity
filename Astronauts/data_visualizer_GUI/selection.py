# -*- coding: utf-8 -*-
"""
Selection partagee par toutes les pages : astronaute, oeil, replicat, moment,
recalage, segmentation.

Un seul ``SelectionState`` pour toute l'application ; chaque page porte sa
propre ``SelectionBar`` branchee dessus, si bien que changer de page garde la
condition choisie. Les menus sont en cascade dans l'ordre d'affichage : chacun ne
propose que ce qui existe pour les choix situes a sa gauche.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from catalog import MOMENT_LABELS, Condition, available, discover_conditions

FIELDS = ("astro", "eye", "rep", "moment", "reg", "seg")
LABELS = {"astro": "Astronaute", "eye": "Oeil", "rep": "Replicat", "moment": "Moment",
          "reg": "Recalage", "seg": "Segmentation"}


class SelectionState(QtCore.QObject):
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.conditions = discover_conditions()
        self.values: dict[str, object] = dict.fromkeys(FIELDS)
        self.options: dict[str, list] = {f: [] for f in FIELDS}
        self._resolve()

    # -- cascade -------------------------------------------------------------
    def _pick(self, field, opts):
        self.options[field] = opts
        if self.values[field] not in opts:
            self.values[field] = opts[0] if opts else None

    def _resolve(self):
        v = self.values
        cs = self.conditions
        self._pick("astro", sorted({c.astro for c in cs}))
        cs = [c for c in cs if c.astro == v["astro"]]
        self._pick("eye", sorted({c.eye for c in cs}))
        cs = [c for c in cs if c.eye == v["eye"]]
        self._pick("rep", sorted({c.rep for c in cs}))
        cs = [c for c in cs if c.rep == v["rep"]]
        self._pick("moment", sorted({c.moment for c in cs}, key=lambda m: m != "before"))
        cs = [c for c in cs if c.moment == v["moment"]]
        self.condition: Condition | None = cs[0] if cs else None
        avail = available(self.condition) if self.condition else {}
        self._pick("reg", list(avail))
        self._pick("seg", avail.get(v["reg"], []))

    def set(self, field: str, value):
        if self.values[field] == value:
            return
        self.values[field] = value
        self._resolve()
        self.changed.emit()

    @property
    def key(self):
        return (self.condition.rel if self.condition else None,
                self.values["reg"], self.values["seg"])

    @property
    def complete(self) -> bool:
        return self.condition is not None and self.values["reg"] and self.values["seg"]


def _display(field, value):
    if field == "moment":
        return MOMENT_LABELS.get(value, value)
    return str(value)


class SelectionBar(QtWidgets.QWidget):
    def __init__(self, state: SelectionState, parent=None):
        super().__init__(parent)
        self.state = state
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.combos: dict[str, QtWidgets.QComboBox] = {}
        for f in FIELDS:
            box = QtWidgets.QVBoxLayout()
            box.setSpacing(1)
            box.addWidget(QtWidgets.QLabel(LABELS[f]))
            cb = QtWidgets.QComboBox()
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            cb.currentIndexChanged.connect(lambda _i, f=f: self._on_combo(f))
            box.addWidget(cb)
            lay.addLayout(box)
            self.combos[f] = cb
        lay.addStretch(1)
        state.changed.connect(self.refresh)
        self.refresh()

    def _on_combo(self, field):
        cb = self.combos[field]
        if cb.signalsBlocked() or cb.currentIndex() < 0:
            return
        self.state.set(field, cb.currentData())

    def refresh(self):
        for f, cb in self.combos.items():
            cb.blockSignals(True)
            cb.clear()
            for opt in self.state.options[f]:
                cb.addItem(_display(f, opt), opt)
            cur = self.state.values[f]
            if cur in self.state.options[f]:
                cb.setCurrentIndex(self.state.options[f].index(cur))
            cb.setEnabled(bool(self.state.options[f]))
            cb.blockSignals(False)
