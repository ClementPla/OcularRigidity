# -*- coding: utf-8 -*-
"""
data_visualizer_GUI -- explorateur PyQt5 des donnees SANS (astronautes).

Pages (onglets) :
  1. Recalage et segmentation : video recalee + masque de choroide (rouge, alpha 0,5).
  2. Pouls : pouls des masques et de la SVD des intensites, bruts et filtres (FIR),
     spectres Lomb-Scargle avec la FC de visit_data.csv en pointille.

Toutes les pages partagent la meme selection (astronaute, oeil, replicat, moment,
recalage, segmentation). Lecture seule : rien n'est ecrit sur disque.

Lancer (kernel pyOR, depuis la racine du depot) :
    python Astronauts/data_visualizer_GUI/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ordre de chargement impose par les DLL de l'env pyOR (Windows) : pandas, PUIS
# torch, PUIS decord/av/Qt. torch apres decord/av -> c10.dll echoue (WinError
# 1114) ; torch avant pandas -> plantage natif a l'import de pandas.
import pandas  # noqa: E402, F401
import torch  # noqa: E402, F401

import pyqtgraph as pg  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

from catalog import DataStore  # noqa: E402
from pages import PAGES  # noqa: E402
from selection import SelectionState  # noqa: E402

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SANS -- data visualizer")
        self.state = SelectionState(self)
        self.store = DataStore()
        tabs = QtWidgets.QTabWidget()
        for cls in PAGES:
            tabs.addTab(cls(self.state, self.store), cls.title)
        self.setCentralWidget(tabs)
        self.resize(1500, 950)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
