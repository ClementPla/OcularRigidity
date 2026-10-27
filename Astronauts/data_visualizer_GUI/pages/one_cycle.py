# -*- coding: utf-8 -*-
"""PAGE 3 -- One-cycle : tous les cycles, frames par bin et epaisseur, one-cycle en video.

Lit les sorties de ``Astronauts/compute_one_cycle_plex.py`` (variante cascade_v9 PLEX pleine
largeur, quel que soit le recalage choisi dans la barre) par ``one_cycle_figure`` -- le meme
code de lecture et la meme presentation que la figure Plotly (celle des notebooks) : tous
les one-cycles bout a bout, frames par bin en barres grises (axe de droite), epaisseur des
masques replies par cycle (+ fondamental en tirets) et du one-cycle resegmente (pointille noir).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from pages.base import NOT_COMPUTED, NotComputed, Page, show_not_computed
from widgets import VideoOverlay

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Astronauts/
import one_cycle_figure as ocf  # noqa: E402

PULSES = {"ct": "Epaisseur des masques", "svd": "Intensite (SVD)", "sinc": "SiNC (appris)"}
MASKS = {"seg": "masque recalcule (resegmente)", "fold": "masque replie (> 0,5)",
         "reg": "one-cycle recale cascade_v9 (frames + masque)"}
VARIANTS = {"6beats": "6 battements par cycle", "1per_video": "1 one-cycle par video",
            "3per_video": "3 one-cycles par video"}
CURSOR_PEN = pg.mkPen((0, 0, 0, 140), width=1, style=QtCore.Qt.DashLine)
FOLD_COLOR = (214, 39, 40)  # epaisseur repliee, tous les cycles
SEG_COLOR = (0, 0, 0)  # epaisseur resegmentee, tous les cycles


def cycle_stats(fold: np.ndarray, seg: np.ndarray) -> dict:
    """Covariance (um^2), Pearson et Spearman entre epaisseur repliee et resegmentee,
    sur les bins renseignes d'un one-cycle (NaN sous 3 bins)."""
    from scipy.stats import pearsonr, spearmanr

    ok = np.isfinite(fold) & np.isfinite(seg)
    n = int(ok.sum())
    out = {"n": n, "cov": np.nan, "r": np.nan, "rho": np.nan}
    if n < 3:
        return out
    a, b = fold[ok], seg[ok]
    out["cov"] = float(np.cov(a, b, ddof=1)[0, 1])
    if a.std() > 0 and b.std() > 0:
        out["r"] = float(pearsonr(a, b)[0])
        out["rho"] = float(spearmanr(a, b)[0])
    return out


def _fmt(v, nd=2):
    return "n/d" if not np.isfinite(v) else f"{v:.{nd}f}".replace(".", ",")


class OneCyclePage(Page):
    title = "One-cycle"
    needs_registration = False

    def build(self, layout):
        self.data = None
        self.cursor = self.highlight = None
        banner = QtWidgets.QLabel(
            "One-cycles de cascade_v9 PLEX pleine largeur (grille PLEX, 1,95 x 5,9 um/px), "
            "groupes de battements x bins -- recalage et segmentation de la barre ignores.")
        banner.setStyleSheet("color: gray")
        layout.addWidget(banner)

        ctrl = QtWidgets.QHBoxLayout()
        self.cb_variant = QtWidgets.QComboBox()
        for k, v in VARIANTS.items():
            self.cb_variant.addItem(v, k)
        self.cb_pulse = QtWidgets.QComboBox()
        self.cb_cycle = QtWidgets.QComboBox()
        self.cb_mask_src = QtWidgets.QComboBox()
        for k, v in MASKS.items():
            self.cb_mask_src.addItem(v, k)
        for lbl, w in (("Variante", self.cb_variant), ("Phase du pouls", self.cb_pulse), ("Cycle (video)", self.cb_cycle),
                       ("Masque affiche", self.cb_mask_src)):
            ctrl.addWidget(QtWidgets.QLabel(lbl))
            ctrl.addWidget(w)
            ctrl.addSpacing(12)
        ctrl.addStretch(1)
        layout.addLayout(ctrl)
        self.cb_variant.currentIndexChanged.connect(lambda _i: self._maybe_reload())
        self.cb_pulse.currentIndexChanged.connect(lambda _i: self.reload())
        self.cb_cycle.currentIndexChanged.connect(lambda _i: self._show_cycle())
        self.cb_mask_src.currentIndexChanged.connect(lambda _i: self._show_cycle(keep_index=True))

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.p_th = self.glw.addPlot(row=0, col=0, title="Epaisseur de la choroide et frames par bin")
        self.p_th.setLabel("bottom", "bin (one-cycles bout a bout ; bin 0 = maximum d'epaisseur)")
        self.p_th.setLabel("left", "epaisseur", "um")
        self.p_th.showGrid(x=True, y=True, alpha=0.2)
        self.p_th.addLegend(offset=(-10, 10))
        # Frames par bin : second ViewBox (axe de droite), derriere les courbes.
        self.vb_count = pg.ViewBox()
        self.p_th.showAxis("right")
        self.p_th.getAxis("right").setLabel("frames par bin")
        self.p_th.scene().addItem(self.vb_count)
        self.p_th.getAxis("right").linkToView(self.vb_count)
        self.vb_count.setXLink(self.p_th)
        self.vb_count.setZValue(-100)
        self.p_th.vb.sigResized.connect(self._sync_views)
        self.p_th.scene().sigMouseClicked.connect(self._on_click)

        self.video = VideoOverlay(fps=5)
        self.video.frameChanged.connect(self._move_cursor)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(self.glw)
        split.addWidget(self.video)
        split.setSizes([320, 600])
        layout.addWidget(split, stretch=1)

    # -- donnees --------------------------------------------------------------------
    @property
    def pulse(self) -> str:
        return self.cb_pulse.currentData() or "ct"

    @property
    def variant(self) -> str:
        return self.cb_variant.currentData() or "6beats"

    @property
    def selection_key(self):
        return (*super().selection_key, self.variant, self.pulse)

    def _refresh_pulse_combo(self, cond):
        """Libelles du choix de pouls, « non calcule » quand le .npz manque."""
        cur = self.pulse
        self.cb_pulse.blockSignals(True)
        self.cb_pulse.clear()
        for k, v in PULSES.items():
            ok = ocf.npz_path(cond.astro, cond.name, k, ocf.cycles_dir(self.variant)).exists()
            self.cb_pulse.addItem(v if ok else f"{v} ({NOT_COMPUTED})", k)
        self.cb_pulse.setCurrentIndex(list(PULSES).index(cur))
        self.cb_pulse.blockSignals(False)

    def fetch(self, cond, reg, seg):
        # Fil secondaire : le pouls a ete fige dans le fil de l'interface (_maybe_reload).
        pulse, variant = self._pulse_req, self._variant_req
        path = ocf.npz_path(cond.astro, cond.name, pulse, ocf.cycles_dir(variant))
        if not path.exists():
            raise NotComputed(f"one-cycle ({VARIANTS[variant]}), phase {PULSES[pulse].lower()}, "
                              f"de {cond.name}")
        return ocf.load_one_cycle(path)

    def _maybe_reload(self):
        if self.state.condition is not None:
            self._refresh_pulse_combo(self.state.condition)
        self._pulse_req = self.pulse
        self._variant_req = self.variant
        super()._maybe_reload()

    def show_result(self, data, _res):
        self.data = data
        cur = self.cb_cycle.currentIndex()
        self.cb_cycle.blockSignals(True)
        self.cb_cycle.clear()
        for g in range(data["n_groups"]):
            n = int(data["counts"][g * data["n_bins"]:(g + 1) * data["n_bins"]].sum())
            self.cb_cycle.addItem(f"{g + 1}/{data['n_groups']} ({n} frames)", g + 1)
        self.cb_cycle.setCurrentIndex(min(max(cur, 0), data["n_groups"] - 1))
        self.cb_cycle.blockSignals(False)
        self._draw_all()
        self._show_cycle()
        self.status.setText(
            f"{data['condition']} | {PULSES[data['pulse']]} | {data['n_groups']} cycles de "
            f"{float(data.get('beats_per_group', data['n_cycle'])):.1f} battements x {data['n_bins']} bins | FC {float(data['hr']):.0f} BPM"
            + (f" (SiNC : {float(data['sinc_bpm']):.0f} BPM, {data['sinc_confidence']})"
               if data["pulse"] == "sinc" and "sinc_bpm" in data else "") + " | "
            f"pixel {data['pixel_um'][0]:.2f} x {data['pixel_um'][1]:.2f} um | "
            "clic sur le graphique : choisir le cycle et le bin de la video\n"
            + "replie vs resegmente par cycle : " + " ; ".join(
                f"c{c} cov {_fmt(st['cov'])} um2, r {_fmt(st['r'])}, rho {_fmt(st['rho'])} (n={st['n']})"
                for c, _x, st in getattr(self, "stats", [])) + "\n"
            f"pouls : {data.get('pulse_info', '')}")

    def clear(self):
        self.data = None
        self.p_th.clear()
        self.vb_count.clear()
        self.cursor = self.highlight = None
        self.video.clear()

    def not_computed(self, what):
        self.vb_count.clear()
        show_not_computed(self.p_th)
        self.video.clear(NOT_COMPUTED)
        self.cb_cycle.clear()

    # -- affichage --------------------------------------------------------------------
    def _sync_views(self):
        self.vb_count.setGeometry(self.p_th.vb.sceneBoundingRect())
        self.vb_count.linkedViewChanged(self.p_th.vb, self.vb_count.XAxis)

    def _draw_all(self):
        """Tous les cycles, une fois par chargement."""
        d = self.data
        nb, ng = d["n_bins"], d["n_groups"]
        df = ocf.bin_table(d)
        slot = df["slot"].to_numpy()
        counts = df["n_frames"].to_numpy()
        self.p_th.clear()
        self.vb_count.clear()
        self.vb_count.addItem(pg.BarGraphItem(x=slot, height=counts, width=0.85,
                                              brush=(190, 190, 190, 150), pen=pg.mkPen(None)))
        self.vb_count.setYRange(0, max(int(counts.max()), 1) * 1.15, padding=0)
        self.stats = []
        for c in range(1, ng + 1):
            sub = df[df["cycle"] == c]
            x = sub["slot"].to_numpy()
            fold, seg = sub["thickness_fold_um"].to_numpy(), sub["thickness_seg_um"].to_numpy()
            self.p_th.plot(x, fold, pen=pg.mkPen(FOLD_COLOR, width=2),
                           symbol="o", symbolSize=5, symbolBrush=FOLD_COLOR, symbolPen=None,
                           connect="finite", name="masques replies" if c == 1 else None)
            self.p_th.plot(x, sub["thickness_fold_fit_um"].to_numpy(),
                           pen=pg.mkPen(FOLD_COLOR, width=1.5, style=QtCore.Qt.DashLine),
                           name="fondamental (replie)" if c == 1 else None)
            self.p_th.plot(x, seg,
                           pen=pg.mkPen(SEG_COLOR, width=1.5, style=QtCore.Qt.DotLine),
                           symbol="s", symbolSize=5, symbolBrush=SEG_COLOR, symbolPen=None,
                           connect="finite", name="one-cycle resegmente" if c == 1 else None)
            self.stats.append((c, x, cycle_stats(fold, seg)))
            if c > 1:
                self.p_th.addItem(pg.InfiniteLine(
                    pos=(c - 1) * nb - 0.5, angle=90,
                    pen=pg.mkPen((128, 128, 128), style=QtCore.Qt.DotLine)))
        y = df[["thickness_fold_um", "thickness_seg_um"]].to_numpy()
        if np.isfinite(y).any():
            lo, hi = np.nanmin(y), np.nanmax(y)
            pad = max(1.0, 0.15 * (hi - lo))
            top = hi + 2.5 * pad  # place pour les statistiques au-dessus des courbes
            self.p_th.setYRange(lo - pad, top, padding=0)
            for c, x, st in self.stats:
                t = pg.TextItem(f"cov {_fmt(st['cov'])} um2\nr {_fmt(st['r'])} | rho {_fmt(st['rho'])}",
                                color=(60, 60, 60), anchor=(0.5, 0))
                t.setFont(pg.QtGui.QFont("Segoe UI", 8))
                self.p_th.addItem(t)
                t.setPos(float(np.mean(x)), top)
            for s_ in slot[counts == 0]:  # bin vide : rien a mesurer
                t = pg.TextItem(NOT_COMPUTED, color=(140, 140, 140), anchor=(0.5, 0.5), angle=90)
                self.p_th.addItem(t)
                t.setPos(s_, 0.5 * (lo + hi))
        else:
            show_not_computed(self.p_th, "aucun bin rempli")
        self.p_th.setXRange(-0.5, nb * ng - 0.5, padding=0.01)
        self.highlight = pg.LinearRegionItem(values=(-0.5, nb - 0.5), movable=False,
                                             brush=(255, 200, 0, 45), pen=pg.mkPen(None))
        self.highlight.setZValue(-50)
        self.p_th.addItem(self.highlight)
        self.cursor = pg.InfiniteLine(pos=0, angle=90, pen=CURSOR_PEN)
        self.p_th.addItem(self.cursor)
        self._sync_views()

    def _show_cycle(self, keep_index=False):
        """Le cycle choisi : surligne en jaune dans le graphique, joue dans la video."""
        d = self.data
        if d is None or self.cb_cycle.currentIndex() < 0:
            return
        c = self.cb_cycle.currentData()
        nb = d["n_bins"]
        sl = slice((c - 1) * nb, c * nb)
        counts = d["counts"][sl]
        if self.highlight is not None:
            self.highlight.setRegion(((c - 1) * nb - 0.5, c * nb - 0.5))
        src = self.cb_mask_src.currentData()
        if src == "reg" and "reg_frames" not in d:
            src = "seg"  # variante 6 battements : pas de one-cycle recale
        frames = d["reg_frames"][sl] if src == "reg" else d["cycles"][sl]
        mask = {"seg": lambda: d["seg_masks"][sl], "fold": lambda: d["occupancy"][sl] > 0.5,
                "reg": lambda: d["reg_masks"][sl]}[src]()
        self.video.set_data(frames, mask, keep_index=keep_index,
                            label_fn=lambda i: (f"cycle {c}, bin {i} | {counts[i]} frames"
                                                + ("" if counts[i] else f" | {NOT_COMPUTED}")))
        self._move_cursor(self.video.index)

    def _move_cursor(self, i):
        d = self.data
        if d is None or self.cursor is None or self.cb_cycle.currentIndex() < 0:
            return
        self.cursor.setValue((self.cb_cycle.currentData() - 1) * d["n_bins"] + i)

    def _on_click(self, ev):
        """Clic dans le graphique : la video va au cycle et au bin cliques."""
        d = self.data
        if d is None or not self.p_th.vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        k = int(round(self.p_th.vb.mapSceneToView(ev.scenePos()).x()))
        if not 0 <= k < d["n_bins"] * d["n_groups"]:
            return
        c, b = k // d["n_bins"] + 1, k % d["n_bins"]
        if self.cb_cycle.currentIndex() != c - 1:
            self.cb_cycle.setCurrentIndex(c - 1)
        self.video.slider.setValue(b)
