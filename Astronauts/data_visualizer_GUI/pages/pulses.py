# -*- coding: utf-8 -*-
"""PAGE 2 -- Pouls : masques, SVD des intensites et SiNC superposes (pouls + spectres), dephasage.

Les pouls sont lus du cache sur E: (``pulse_compute.cached_compute``) ; calcules puis
enregistres la premiere fois seulement."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

import pulse_compute
from pages.base import NOT_COMPUTED, NotComputed, Page, show_not_computed

COLORS = {"mask": (220, 50, 47), "svd": (38, 139, 210), "sinc": (40, 160, 60)}
FILT_PEN = {k: pg.mkPen(c, width=2) for k, c in COLORS.items()}
RAW_PEN_LIGHT = {k: pg.mkPen((*c, 110), width=1) for k, c in COLORS.items()}
NAMES = {"mask": "masque", "svd": "intensite (SVD)", "sinc": "SiNC"}
HR_PEN = pg.mkPen((0, 0, 0), width=1.5, style=QtCore.Qt.DashLine)
BAND_BRUSH = pg.mkBrush(255, 200, 0, 40)


class PulsePage(Page):
    title = "Pouls"

    def build(self, layout):
        ctrl = QtWidgets.QHBoxLayout()
        self.cb_src = {"mask": QtWidgets.QCheckBox("Masque (epaisseur)"),
                       "svd": QtWidgets.QCheckBox("Intensite (SVD)"),
                       "sinc": QtWidgets.QCheckBox("SiNC (appris)")}
        self.cb_raw = QtWidgets.QCheckBox("Brut")
        self.cb_filt = QtWidgets.QCheckBox("Filtre (FIR)")
        for i, cb in enumerate((*self.cb_src.values(), self.cb_raw, self.cb_filt)):
            if i == len(self.cb_src):
                ctrl.addSpacing(30)
            cb.setChecked(True)
            cb.toggled.connect(self._redraw)
            ctrl.addWidget(cb)
        ctrl.addStretch(1)
        layout.addLayout(ctrl)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        # Ligne 1 : pouls superposes. Unites differentes (px contre u.a.) : chaque
        # trace est centree-reduite, chaque spectre ramene a son maximum.
        self.p_pulse = self.glw.addPlot(row=0, col=0, title="Pouls (centres-reduits)")
        self.p_pulse.setLabel("bottom", "temps", "s")
        self.p_pulse.setLabel("left", "amplitude (ecarts-types)")
        self.p_spec = self.glw.addPlot(row=0, col=1, title="Spectres Lomb-Scargle (normalises au maximum)")
        self.p_spec.setLabel("bottom", "frequence (BPM)")
        self.p_spec.setLabel("left", "puissance (normalisee)")
        for p in (self.p_pulse, self.p_spec):
            p.addLegend(offset=(-10, 10))
        # Ligne 2 : phase instantanee (Hilbert) de chaque pouls, brut et filtre.
        self.p_iphase = self.glw.addPlot(row=1, col=0, title="Phase instantanee (Hilbert)")
        self.p_iphase.setLabel("bottom", "temps", "s")
        self.p_iphase.setLabel("left", "phase (deg)")
        self.p_iphase.setYRange(-180, 180, padding=0.02)
        self.p_iphase.getAxis("left").setTicks([[(v, str(v)) for v in range(-180, 181, 90)]])
        self.p_iphase.addLegend(offset=(-10, 10))
        # Ligne 3 : dephasage instantane (Hilbert) des pouls filtres, et sa distribution.
        self.p_phase = self.glw.addPlot(row=2, col=0, title="Dephasage SVD - masque (Hilbert)")
        self.p_phase.setLabel("bottom", "temps", "s")
        self.p_phase.setLabel("left", "dphi (deg)")
        self.p_phase.setYRange(-180, 180, padding=0.02)
        self.p_phase.getAxis("left").setTicks([[(v, str(v)) for v in range(-180, 181, 90)]])
        self.p_hist = self.glw.addPlot(row=2, col=1, title="Distribution du dephasage (hors bords)")
        self.p_hist.setLabel("bottom", "dphi (deg)")
        self.p_hist.setLabel("left", "fraction du temps")
        self.p_hist.getAxis("bottom").setTicks([[(v, str(v)) for v in range(-180, 181, 90)]])
        for p in self._plots:
            p.showGrid(x=True, y=True, alpha=0.2)
        self.p_phase.setXLink(self.p_pulse)
        self.p_iphase.setXLink(self.p_pulse)
        self.glw.ci.layout.setColumnStretchFactor(0, 3)
        self.glw.ci.layout.setColumnStretchFactor(1, 2)
        layout.addWidget(self.glw, stretch=1)
        self.ds = self.res = None

    @property
    def _plots(self):
        return (self.p_pulse, self.p_spec, self.p_iphase, self.p_phase, self.p_hist)

    def work(self, ds):
        if ds.mask is None:
            raise NotComputed("pouls -- aucun masque compatible avec la video ("
                              + "; ".join(ds.notes) + ")")
        if ds.t is None:
            raise NotComputed("pouls -- " + "; ".join(ds.notes))
        path = pulse_compute.cache_path(ds.reg, ds.seg, ds.cond.astro, ds.cond.name)
        return pulse_compute.cached_compute(path, ds.frames, ds.mask, ds.t, ds.hr_bpm, ds.pixel_mm)

    def show_result(self, ds, res):
        self.ds, self.res = ds, res
        self._redraw()
        lines = [f"{ds.cond.name} | {ds.reg} | {ds.seg}",
                 f"FC {res.hr:.1f} BPM ({res.hr_source}) | bande FIR "
                 f"{res.band_bpm[0]:.0f}-{res.band_bpm[1]:.0f} BPM | "
                 f"fs {res.fs:.2f} Hz | {res.t.size} frames, {res.t[-1]:.1f} s",
                 f"masque : {res.mask_pulse.info}",
                 "SVD : " + (res.svd_pulse.info if res.svd_pulse else "trop peu de pixels sous la RPE"),
                 "SiNC : " + (res.sinc_pulse.info if res.sinc_pulse is not None
                              else f"{NOT_COMPUTED} ({res.sinc_note})")]
        lines[0] += (" | pouls lus du cache" if res.from_cache else " | pouls calcules et enregistres")
        if res.phase is not None:
            ph = res.phase
            lines.append(
                f"dephasage (Hilbert, pouls filtres, bords 20 % exclus) : moyenne circulaire "
                f"{np.degrees(ph.mean):+.0f} deg = {ph.dt_ms:+.0f} ms a {res.hr:.0f} BPM, "
                f"PLV {ph.plv:.2f} (0 = aucun lien, 1 = verrouilles) -- dphi > 0 : SVD en avance")
        self.status.setText("\n".join(lines))

    def clear(self):
        self.ds = self.res = None
        for p in self._plots:
            p.clear()

    def not_computed(self, what):
        for p in self._plots:
            show_not_computed(p)

    def _redraw(self, *_):
        for p in self._plots:
            p.clear()
        if self.res is None:
            return
        r = self.res
        self._draw_overlay()
        self._draw_inst_phase()
        self._draw_phase()
        self.p_spec.setXRange(r.bpm[0], min(r.bpm[-1], 3.2 * r.hr), padding=0)
        self.p_pulse.setXRange(r.t[0], r.t[-1], padding=0.01)

    def _draw_overlay(self):
        r = self.res
        pt, ps = self.p_pulse, self.p_spec
        band = pg.LinearRegionItem(values=r.band_bpm, movable=False, brush=BAND_BRUSH,
                                   pen=pg.mkPen(None))
        band.setZValue(-10)
        ps.addItem(band)
        ps.addItem(pg.InfiniteLine(pos=r.hr, angle=90, pen=HR_PEN, label=f"FC {r.hr:.0f}",
                                   labelOpts={"position": 0.92, "color": (0, 0, 0)}))
        if r.sinc_pulse is not None and self.cb_src["sinc"].isChecked() and np.isfinite(r.sinc_bpm):
            ps.addItem(pg.InfiniteLine(
                pos=r.sinc_bpm, angle=90, pen=pg.mkPen(COLORS["sinc"], width=1.5, style=QtCore.Qt.DotLine),
                label=f"SiNC {r.sinc_bpm:.0f}", labelOpts={"position": 0.8, "color": COLORS["sinc"]}))
        traces = {k: tr for k, tr in _traces(r) if tr is not None}
        for key, tr in traces.items():
            if not self.cb_src[key].isChecked():
                continue
            if self.cb_raw.isChecked():
                pt.plot(r.t, _z(tr.raw), pen=RAW_PEN_LIGHT[key], name=f"{NAMES[key]} brut")
                ps.plot(r.bpm, _unit_max(tr.spec_raw), pen=RAW_PEN_LIGHT[key],
                        name=f"{NAMES[key]} brut")
            if self.cb_filt.isChecked():
                pt.plot(r.u, _z(tr.filtered), pen=FILT_PEN[key], name=f"{NAMES[key]} FIR")
                ps.plot(r.bpm, _unit_max(tr.spec_filtered), pen=FILT_PEN[key],
                        name=f"{NAMES[key]} FIR")
        title = "Pouls (centres-reduits)"
        if "mask" in traces and "svd" in traces:
            a, b = traces["mask"].filtered, traces["svd"].filtered
            m = int(0.2 * a.size)  # bords exclus : transitoire du FIR
            c = float(np.corrcoef(a[m:a.size - m], b[m:b.size - m])[0, 1])
            title += f" -- r (FIR, hors bords) = {c:+.2f}"
        pt.setTitle(title)

    def _draw_inst_phase(self):
        """Phase de Hilbert de chaque pouls coche : brut (points clairs) et filtre (points
        fonces). Points, pas de ligne : la phase saute de +180 a -180 a chaque cycle."""
        r, pt = self.res, self.p_iphase
        for key, tr in _traces(r):
            if tr is None or not self.cb_src[key].isChecked() or tr.phase_filtered is None:
                continue
            col = FILT_PEN[key].color()
            if self.cb_raw.isChecked():
                c = (col.red(), col.green(), col.blue(), 90)
                pt.plot(r.u, np.degrees(tr.phase_raw), pen=None, symbol="o", symbolSize=3,
                        symbolPen=None, symbolBrush=c, name=f"{NAMES[key]} brut")
            if self.cb_filt.isChecked():
                pt.plot(r.u, np.degrees(tr.phase_filtered), pen=None, symbol="o", symbolSize=4,
                        symbolPen=None, symbolBrush=col, name=f"{NAMES[key]} FIR")
        pt.setYRange(-180, 180, padding=0.02)

    def _draw_phase(self):
        r, ph = self.res, self.res.phase
        pt, ps = self.p_phase, self.p_hist
        if ph is None:
            pt.setTitle("Dephasage SVD - masque (Hilbert) -- pas de pouls SVD")
            return
        deg = np.degrees(ph.dphi)
        # Plages ou les deux pouls sont en phase (|dphi| <= tol), en jaune comme
        # la bande du passe-bande sur les spectres.
        tol = pulse_compute.IN_PHASE_TOL_DEG
        for t0, t1 in pulse_compute.in_phase_runs(r.u, ph, tol):
            box = pg.LinearRegionItem(values=(t0, t1), movable=False, brush=BAND_BRUSH,
                                      pen=pg.mkPen(None))
            box.setZValue(-10)
            pt.addItem(box)
        for y in (-tol, tol):
            pt.addItem(pg.InfiniteLine(pos=y, angle=0,
                                       pen=pg.mkPen((200, 150, 0, 120), style=QtCore.Qt.DotLine)))
        in_phase = float((np.abs(deg[ph.core]) <= tol).mean())
        # Hors coeur (transitoire du FIR) en gris, coeur en noir. Points, pas de
        # ligne : dphi saute de +180 a -180 sans que rien ne se passe.
        pt.plot(r.u[~ph.core], deg[~ph.core], pen=None, symbol="o", symbolSize=3,
                symbolPen=None, symbolBrush=(170, 170, 170))
        pt.plot(r.u[ph.core], deg[ph.core], pen=None, symbol="o", symbolSize=3,
                symbolPen=None, symbolBrush=(40, 40, 40))
        mean_deg = float(np.degrees(ph.mean))
        mean_pen = pg.mkPen((220, 50, 47), width=2)
        pt.addItem(pg.InfiniteLine(pos=mean_deg, angle=0, pen=mean_pen))
        pt.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen((0, 0, 0, 80))))
        flipped = " -- SVD inverse" if r.svd_pulse and "INVERSE" in r.svd_pulse.info else ""
        pt.setTitle(f"Dephasage SVD - masque (Hilbert) -- {mean_deg:+.0f} deg, "
                    f"{ph.dt_ms:+.0f} ms, PLV {ph.plv:.2f}, en phase (+/-{tol:.0f} deg) "
                    f"{in_phase:.0%}{flipped}")

        hist, edges = np.histogram(deg[ph.core], bins=36, range=(-180, 180))
        ps.plot(edges, hist / max(hist.sum(), 1), stepMode="center", fillLevel=0,
                brush=(40, 40, 40, 90), pen=pg.mkPen((40, 40, 40)))
        ps.addItem(pg.InfiniteLine(pos=mean_deg, angle=90, pen=mean_pen,
                                   label=f"{mean_deg:+.0f} deg",
                                   labelOpts={"position": 0.92, "color": (220, 50, 47)}))
        ps.setXRange(-180, 180, padding=0.02)


def _traces(r):
    return (("mask", r.mask_pulse), ("svd", r.svd_pulse), ("sinc", r.sinc_pulse))


def _z(y):
    y = np.asarray(y, dtype=float)
    return (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)


def _unit_max(spec):
    return spec / (spec.max() + 1e-12)
