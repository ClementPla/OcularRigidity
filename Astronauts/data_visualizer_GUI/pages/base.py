# -*- coding: utf-8 -*-
"""
Socle commun des pages : barre de selection en haut, ligne d'etat, et un
chargement en arriere-plan qui ne bloque pas l'interface.

Une page nouvelle derive de ``Page`` et implemente :
  * ``build(layout)``  -- ses widgets, sous la barre de selection ;
  * ``fetch(cond, reg, seg)`` -- ce qu'elle charge (defaut : la video recalee +
    son masque, ``DataStore.load``) ; dans un fil secondaire ;
  * ``work(dataset)``  -- calcul lourd, dans le meme fil (optionnel) ;
  * ``show_result(dataset, result)`` -- affichage, dans le fil de l'interface ;
  * ``not_computed(quoi)`` -- (optionnel) ce que la page montre quand la donnee
    n'existe pas encore sur disque ; la ligne d'etat dit deja « non calcule ».
Elle ne recharge que lorsqu'elle est visible : changer la selection sur une page
ne declenche pas le calcul de toutes les autres.

Regle de tout le GUI : une donnee absente du disque s'affiche « non calcule »
(``raise NotComputed(quoi)`` depuis ``fetch``/``work``), jamais comme une erreur.
"""

from __future__ import annotations

import traceback

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from selection import SelectionBar
from widgets import NOT_COMPUTED


class NotComputed(Exception):
    """La donnee demandee n'existe pas (pas encore calculee) : ``str(e)`` dit laquelle."""


# Ancien nom, garde pour les pages qui l'importent encore.
Unavailable = NotComputed


def show_not_computed(plot: pg.PlotItem, detail: str = "") -> None:
    """Vide un graphique et y pose « non calcule » au centre."""
    plot.clear()
    txt = pg.TextItem(NOT_COMPUTED + (f"\n{detail}" if detail else ""), color=(120, 120, 120),
                      anchor=(0.5, 0.5))
    txt.setFont(QtWidgets.QApplication.font())
    plot.addItem(txt)
    plot.setXRange(0, 1, padding=0)
    plot.setYRange(0, 1, padding=0)
    txt.setPos(0.5, 0.5)


class _Signals(QtCore.QObject):
    done = QtCore.pyqtSignal(int, object, object)
    failed = QtCore.pyqtSignal(int, str)


class _Job(QtCore.QRunnable):
    def __init__(self, req, fn):
        super().__init__()
        self.req, self.fn = req, fn
        self.signals = _Signals()

    def run(self):
        try:
            ds, res = self.fn()
            self.signals.done.emit(self.req, ds, res)
        except NotComputed as e:
            self.signals.failed.emit(self.req, "!" + str(e))
        except Exception:  # noqa: BLE001
            self.signals.failed.emit(self.req, traceback.format_exc())


class Page(QtWidgets.QWidget):
    title = "page"
    # False : la page ne lit que la condition (pas le couple recalage/segmentation).
    needs_registration = True

    def __init__(self, state, store, parent=None):
        super().__init__(parent)
        self.state, self.store = state, store
        self._req = 0
        self._shown_key = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(SelectionBar(state))
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.status)
        self.build(lay)
        state.changed.connect(self._maybe_reload)

    # -- a fournir par les pages ------------------------------------------------
    def build(self, layout):
        raise NotImplementedError

    def fetch(self, cond, reg, seg):
        return self.store.load(cond, reg, seg)

    def work(self, dataset):
        return None

    def show_result(self, dataset, result):
        raise NotImplementedError

    def clear(self):
        pass

    def not_computed(self, what: str):
        pass

    # -- mecanique ----------------------------------------------------------------
    @property
    def selection_key(self):
        if self.needs_registration:
            return self.state.key
        return (self.state.condition.rel if self.state.condition else None,)

    def reload(self):
        """Force le rechargement (ex. un controle propre a la page a change)."""
        self._shown_key = None
        self._maybe_reload()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._maybe_reload()

    def _set_not_computed(self, what: str):
        self.clear()
        self.status.setText(f"{NOT_COMPUTED} : {what}" if what else NOT_COMPUTED)
        self.not_computed(what)

    def _maybe_reload(self):
        if not self.isVisible() or self.selection_key == self._shown_key:
            return
        self._shown_key = self.selection_key
        self._req += 1
        cond, reg, seg = self.state.condition, self.state.values["reg"], self.state.values["seg"]
        if cond is None or (self.needs_registration and not self.state.complete):
            self._set_not_computed("aucune video recalee + masque pour cette condition")
            return
        what = f"{cond.name} | {reg} | {seg}" if self.needs_registration else cond.name
        self.status.setText(f"Chargement de {what} ...")

        def fn():
            ds = self.fetch(cond, reg, seg)
            return ds, self.work(ds)

        job = _Job(self._req, fn)
        job.signals.done.connect(self._on_done)
        job.signals.failed.connect(self._on_failed)
        QtCore.QThreadPool.globalInstance().start(job)

    def _on_done(self, req, ds, res):
        if req != self._req:
            return  # une selection plus recente est en cours
        self.show_result(ds, res)

    def _on_failed(self, req, tb):
        if req != self._req:
            return
        if tb.startswith("!"):  # NotComputed : donnee absente, pas une erreur
            self._set_not_computed(tb[1:])
            return
        self.clear()
        self.status.setText("Erreur :\n" + tb.strip().splitlines()[-1])
        print(tb, flush=True)
