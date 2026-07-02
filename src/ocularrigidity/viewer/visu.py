"""
Mask Browser + Viewer (pygame_gui edition with registration, smoothing, charts).

Usage:
    python mask_browser.py
    python mask_browser.py --input-root /mnt/smb/dataFiles --output-root /path/to/masks
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
import pygame_gui
from pygame_gui.core import ObjectID

import matplotlib

matplotlib.use("Agg")  # headless, for off-screen rendering into pygame surfaces
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ocularrigidity.consts import (
    ROOT_DATA_MNT,
    ROOT_MASKS,
    ROOT_COMPRESSED_VIDEO,
    ROOT_REGISTERED_CACHE,
)
from ocularrigidity.registration.registration_engine import VideoRegistrator
from ocularrigidity.segmentation.postprocess.interfaces import (
    extract_boundaries_gpu,
    rebuild_mask,
    smooth_boundary_2d,
)
from ocularrigidity.thickness.features import (
    compute_deltaY_masks,
    compute_deltaY_boundaries,
)


# --- Theme ------------------------------------------------------------------
BG = (18, 18, 24)
PANEL = (28, 28, 36)
PANEL_BORDER = (48, 48, 60)
ACCENT = (255, 85, 85)
TEXT = (230, 230, 235)
TEXT_DIM = (150, 150, 160)
TEXT_FAINT = (100, 100, 115)

SIDEBAR_W = 400
CONTROL_H = 70
SIDEBAR_PAD = 15
CHART_W = 520  # width of the chart column next to the video
CHART_H = 180  # height of each of the 3 stacked charts

FONT_NAME = "dejavusans,helvetica,arial,sans"

UI_THEME = {
    "defaults": {
        "colours": {
            "normal_bg": "#1c1c24",
            "hovered_bg": "#2a2a36",
            "disabled_bg": "#1c1c24",
            "selected_bg": "#38282c",
            "active_bg": "#38282c",
            "dark_bg": "#12121c",
            "normal_text": "#e6e6eb",
            "hovered_text": "#ffffff",
            "disabled_text": "#646473",
            "selected_text": "#ffffff",
            "active_text": "#ffffff",
            "normal_border": "#30303c",
            "hovered_border": "#ff5555",
            "disabled_border": "#30303c",
            "selected_border": "#ff5555",
            "active_border": "#ff5555",
            "link_text": "#ff5555",
            "link_hover": "#ff8080",
            "link_selected": "#ff5555",
            "text_shadow": "#12121cff",
            "filled_bar": "#ff5555",
            "unfilled_bar": "#30303c",
        },
        "font": {"name": FONT_NAME, "size": "13"},
    },
    "label": {
        "colours": {"normal_text": "#9696a0"},
        "font": {"name": FONT_NAME, "size": "13"},
    },
    "#header_label": {
        "colours": {"normal_text": "#9696a0"},
        "font": {"name": FONT_NAME, "size": "11", "bold": "1"},
    },
    "#status_label": {
        "colours": {"normal_text": "#9696a0"},
        "font": {"name": FONT_NAME, "size": "12"},
    },
    "button": {"misc": {"border_width": "1", "shape_corner_radius": "4"}},
    "#toggle_on": {
        "colours": {
            "normal_bg": "#38282c",
            "normal_border": "#ff5555",
            "normal_text": "#ff5555",
            "hovered_bg": "#402c32",
            "hovered_border": "#ff8080",
            "hovered_text": "#ff8080",
            "active_bg": "#38282c",
            "active_border": "#ff5555",
            "active_text": "#ff5555",
        },
    },
    "#toggle_off": {
        "colours": {
            "normal_bg": "#1c1c24",
            "normal_border": "#30303c",
            "normal_text": "#9696a0",
            "hovered_bg": "#2a2a36",
            "hovered_border": "#404050",
            "hovered_text": "#e6e6eb",
            "active_bg": "#1c1c24",
            "active_border": "#30303c",
            "active_text": "#9696a0",
        },
    },
    "text_entry_line": {
        "misc": {"border_width": "1", "shape_corner_radius": "4"},
        "colours": {"dark_bg": "#16161e", "normal_text": "#e6e6eb"},
    },
    "selection_list": {"misc": {"border_width": "1", "shape_corner_radius": "4"}},
    "#filter_input": {
        "misc": {"border_width": "1", "shape_corner_radius": "4"},
        "colours": {"dark_bg": "#16161e", "normal_text": "#e6e6eb"},
    },
    "#filter_label": {
        "colours": {"normal_text": "#9696a0"},
        "font": {"name": FONT_NAME, "size": "11", "bold": "1"},
    },
    "horizontal_slider": {
        "colours": {
            "dark_bg": "#16161e",
            "normal_bg": "#30303c",
            "hovered_bg": "#404050",
            "selected_bg": "#ff5555",
        },
    },
}

# --- File discovery ---------------------------------------------------------


def find_masks(output_root: Path) -> list[str]:
    if not output_root.exists():
        return []
    return sorted(
        str(p.parent.relative_to(output_root)) for p in output_root.rglob("mask.npz")
    )


def probe_cube(input_bin: Path, input_compressed: Path, rel: str) -> dict:
    mp4 = input_compressed / rel / "cube.mp4"
    bin_file = input_bin / rel / "cube.bin"
    return {
        "mp4": mp4.exists(),
        "bin": bin_file.exists(),
        "mp4_path": mp4 if mp4.exists() else None,
        "bin_path": bin_file if bin_file.exists() else None,
    }


# --- Frame composition ------------------------------------------------------


def compose_overlay_frame(
    frame: np.ndarray,
    mask: Optional[np.ndarray],
    show_mask: bool,
    mask_alpha: float = 0.4,
) -> np.ndarray:
    """Compose a single (H, W) frame into an (H, W, 3) uint8 overlay.

    Composed lazily per displayed frame; pre-composing the whole video would
    allocate gigabytes for long sequences.
    """
    out = np.repeat(frame[..., None], 3, axis=2)
    if not show_mask or mask is None:
        return out
    overlay = np.array([255, 85, 85], dtype=np.float32)
    gray = np.arange(256, dtype=np.float32)
    lut = (gray[:, None] * (1 - mask_alpha) + overlay[None, :] * mask_alpha).astype(
        np.uint8
    )
    out[mask] = lut[frame[mask]]
    return out


def np_to_surface(arr: np.ndarray) -> pygame.Surface:
    return pygame.surfarray.make_surface(arr.swapaxes(0, 1))


# --- Chart rendering --------------------------------------------------------


def render_charts(
    delta_y: np.ndarray,  # (T, W) or (T,)
    thickness: np.ndarray,  # (T, W) or (T,)
    area: np.ndarray,  # (T,)
    width_px: int,
    height_px: int,
    dpi: int = 100,
) -> tuple[pygame.Surface, tuple[float, float]]:
    """Render 3 stacked charts into a pygame Surface (no playhead).

    The playhead is drawn separately as a cheap pygame line so the (expensive)
    matplotlib render only happens when the data changes, not every frame.
    Returns ``(surface, (x0_px, x1_px))`` where the tuple is the pixel x-range
    of the shared data area, used to position the playhead.
    """
    fig = plt.figure(
        figsize=(width_px / dpi, (3 * height_px) / dpi), dpi=dpi, facecolor="#1c1c24"
    )

    series = [
        (delta_y, "deltaY (thickness)", "#ff5555"),
        (thickness, "thickness (span)", "#5599ff"),
        (area, "Area (pixel count)", "#55ff88"),
    ]

    for i, (data, name, color) in enumerate(series):
        ax = fig.add_subplot(3, 1, i + 1)
        ax.set_facecolor("#1c1c24")

        if data.ndim == 2:
            mean = np.nanmean(data, axis=1)
            std = np.nanstd(data, axis=1)
            T = len(mean)
            x = np.arange(T)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
            ax.plot(x, mean, color=color, linewidth=1.2)
        else:
            T = len(data)
            ax.plot(np.arange(T), data, color=color, linewidth=1.2)

        ax.set_xlim(0, max(T - 1, 1))
        ax.set_title(name, color="#e6e6eb", fontsize=9, loc="left")
        ax.tick_params(colors="#9696a0", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30303c")
        ax.grid(True, color="#30303c", linewidth=0.5, alpha=0.5)

    fig.tight_layout(pad=0.8)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw = renderer.buffer_rgba()
    size = canvas.get_width_height()
    # Shared data-area x-range in pixels (kept inside every subplot's axes box).
    x0_px = max(ax.get_position().x0 for ax in fig.axes) * size[0]
    x1_px = min(ax.get_position().x1 for ax in fig.axes) * size[0]
    plt.close(fig)

    surf = pygame.image.frombuffer(raw, size, "RGBA").convert_alpha()
    return surf, (x0_px, x1_px)


# --- App --------------------------------------------------------------------


class App:
    def __init__(self, input_root: str, output_root: str, window_size=(1800, 950)):
        pygame.init()
        pygame.display.set_caption("Mask Browser")
        self.screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        theme_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(UI_THEME, theme_file)
        theme_file.close()
        self.ui = pygame_gui.UIManager(window_size, theme_file.name)

        family = "DejaVu Sans,Helvetica Neue,Helvetica,Arial"
        self.font = pygame.font.SysFont(family, 14)
        self.font_sm = pygame.font.SysFont(family, 12)
        self.font_lg = pygame.font.SysFont(family, 16, bold=True)

        self.input_root = input_root
        self.output_root = output_root
        self.use_compressed = True
        # Registration cache: when on, RegisteredVideo reads/writes the
        # registered frames+masks under ROOT_REGISTERED_CACHE so reloads are fast.
        self.use_cache = True
        self.cache_root = Path(ROOT_REGISTERED_CACHE)
        self.apply_smoothing = True
        self.sigma_time = 5.0
        self.sigma_col = 2.0

        self.entries: list[str] = []
        self._entry_info: dict[str, dict] = {}
        self._filtered_entries: list[str] = []
        self.selected_rel: Optional[str] = None

        # Raw (registered, unsmoothed) data — kept so we can re-smooth without reload
        self._raw_masks: Optional[np.ndarray] = None
        self._raw_video: Optional[np.ndarray] = None

        # Derived (smoothed mask + composed frames + features)
        self.masks: Optional[np.ndarray] = None
        self.video: Optional[np.ndarray] = None
        self._features_cache: Optional[dict] = None
        # Cached current-frame surface (recomposed only when something changes)
        self._frame_surf: Optional[pygame.Surface] = None
        self._frame_key = None
        # Bumped whenever masks/video change, to invalidate the frame cache
        self._render_version = 0
        # Static chart surface + playhead x-range; re-rendered only on data/resize
        self._chart_surface: Optional[pygame.Surface] = None
        self._chart_xmap: Optional[tuple[float, float]] = None

        self.idx = 0
        self.playing = False
        self.fps = 30
        self.last_tick = 0
        self.status_msg = "Select a mask from the sidebar"

        self._bar_rect: Optional[pygame.Rect] = None

        self.show_mask = True
        self.mask_alpha = 0.4

        self._build_sidebar()
        self.refresh_entries()

    # -- Sidebar construction ------------------------------------------------

    def _build_sidebar(self):
        w, h = self.screen.get_size()
        pad = SIDEBAR_PAD
        col_w = SIDEBAR_W - 2 * pad
        y = pad

        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((pad, y), (col_w, 20)),
            text="MASK BROWSER",
            manager=self.ui,
            object_id=ObjectID(object_id="#header_label"),
        )
        y += 26

        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((pad, y), (col_w, 16)),
            text="Input root",
            manager=self.ui,
        )
        y += 18
        self.input_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((pad, y), (col_w, 30)),
            manager=self.ui,
            initial_text=str(self.input_root),
        )
        y += 38

        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((pad, y), (col_w, 16)),
            text="Output root",
            manager=self.ui,
        )
        y += 18
        self.output_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((pad, y), (col_w, 30)),
            manager=self.ui,
            initial_text=str(self.output_root),
        )
        y += 42

        self.compressed_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((pad, y), (col_w, 30)),
            text=self._toggle_label("Prefer compressed (mp4)", self.use_compressed),
            manager=self.ui,
            object_id=ObjectID(
                object_id="#toggle_on" if self.use_compressed else "#toggle_off"
            ),
        )
        y += 38

        self.cache_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((pad, y), (col_w, 30)),
            text=self._toggle_label("Use registration cache", self.use_cache),
            manager=self.ui,
            object_id=ObjectID(
                object_id="#toggle_on" if self.use_cache else "#toggle_off"
            ),
        )
        y += 38

        self.refresh_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((pad, y), (col_w, 30)),
            text="↻  Refresh",
            manager=self.ui,
        )
        y += 34

        self.count_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((pad, y), (col_w, 18)),
            text="0 masks",
            manager=self.ui,
            object_id=ObjectID(object_id="#status_label"),
        )
        y += 24

        # Smoothing controls
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((pad, y), (col_w, 16)),
            text="SMOOTHING",
            manager=self.ui,
            object_id=ObjectID(object_id="#header_label"),
        )
        y += 22

        self.smooth_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((pad, y), (col_w, 30)),
            text=self._toggle_label("Apply smoothing", self.apply_smoothing),
            manager=self.ui,
            object_id=ObjectID(
                object_id="#toggle_on" if self.apply_smoothing else "#toggle_off"
            ),
        )
        y += 38

        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((pad, y), (col_w, 16)),
            text=f"sigma_time: {self.sigma_time:.1f}",
            manager=self.ui,
        )
        self.sigma_time_label_rect = pygame.Rect((pad, y), (col_w, 16))
        y += 18
        self.sigma_time_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((pad, y), (col_w, 20)),
            start_value=self.sigma_time,
            value_range=(0.5, 30.0),
            manager=self.ui,
        )
        y += 28

        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((pad, y), (col_w, 16)),
            text=f"sigma_col: {self.sigma_col:.1f}",
            manager=self.ui,
        )
        self.sigma_col_label_rect = pygame.Rect((pad, y), (col_w, 16))
        y += 18
        self.sigma_col_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((pad, y), (col_w, 20)),
            start_value=self.sigma_col,
            value_range=(0.5, 15.0),
            manager=self.ui,
        )
        y += 30

        # We store slider labels as mutable labels so we can refresh their text
        self.sigma_time_label = pygame_gui.elements.UILabel(
            relative_rect=self.sigma_time_label_rect,
            text=f"sigma_time: {self.sigma_time:.1f}",
            manager=self.ui,
        )
        self.sigma_col_label = pygame_gui.elements.UILabel(
            relative_rect=self.sigma_col_label_rect,
            text=f"sigma_col: {self.sigma_col:.1f}",
            manager=self.ui,
        )

        # Filter
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((pad, y), (col_w, 16)),
            text="Filter (patient / date / OD-OS)",
            manager=self.ui,
            object_id=ObjectID(object_id="#filter_label"),
        )
        y += 18
        self.filter_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((pad, y), (col_w, 30)),
            manager=self.ui,
            object_id=ObjectID(object_id="#filter_input"),
        )
        y += 38

        # List
        list_top = y
        list_h = h - list_top - pad
        self.entry_list = pygame_gui.elements.UISelectionList(
            relative_rect=pygame.Rect((pad, list_top), (col_w, list_h)),
            item_list=[],
            manager=self.ui,
        )
        self._list_top = list_top
        self._list_col_w = col_w

    def _toggle_label(self, text, on):
        return f"{'[X]' if on else '[ ]'}  {text}"

    def _update_toggle_style(self, btn, on):
        btn.change_object_id(ObjectID(object_id="#toggle_on" if on else "#toggle_off"))

    def _relayout_sidebar(self):
        w, h = self.screen.get_size()
        new_h = h - self._list_top - SIDEBAR_PAD
        if new_h > 50:
            self.entry_list.set_dimensions((self._list_col_w, new_h))

    # -- Data ops ------------------------------------------------------------

    def refresh_entries(self):
        self.input_root = self.input_entry.get_text()
        self.output_root = self.output_entry.get_text()
        self.entries = find_masks(Path(self.output_root).expanduser())
        if self.selected_rel not in self.entries:
            self.selected_rel = None
        self._rebuild_list()

    def _format_entry(self, rel, info):
        if info["mp4"]:
            tag = "[mp4]"
        elif info["bin"]:
            tag = "[bin]"
        else:
            tag = "[ - ]"
        return f"{tag}  {rel}"

    def _rebuild_list(self):
        inp = Path(self.input_root).expanduser()
        inp_compressed = Path(ROOT_COMPRESSED_VIDEO).expanduser()
        self._entry_info = {
            rel: probe_cube(inp, inp_compressed, rel) for rel in self.entries
        }

        query = self.filter_entry.get_text().strip().lower()
        terms = query.split()
        if terms:
            filtered = [
                rel for rel in self.entries if all(t in rel.lower() for t in terms)
            ]
        else:
            filtered = list(self.entries)

        self._filtered_entries = filtered
        display_items = [
            self._format_entry(rel, self._entry_info[rel]) for rel in filtered
        ]
        self.entry_list.set_item_list(display_items)

        if terms:
            self.count_label.set_text(f"{len(filtered)} / {len(self.entries)} masks")
        else:
            self.count_label.set_text(f"{len(self.entries)} masks")

    def load_selected(self):
        if self.selected_rel is None:
            return
        rel = self.selected_rel

        # Compressed mp4 lives under ROOT_COMPRESSED_VIDEO; raw cube.bin under
        # the (editable) input root. RegisteredVideo handles loading + the rigid
        # displacement registration, and (optionally) caches the result.
        if self.use_compressed:
            root_data = Path(ROOT_COMPRESSED_VIDEO).expanduser()
        else:
            root_data = Path(self.input_root).expanduser()

        cache_dir = self.cache_root if self.use_cache else None
        try:
            self.status_msg = f"Loading + registering: {rel}"
            self._flush_status()
            rv = VideoRegistrator(
                video=Path(rel),
                root_masks=Path(self.output_root).expanduser(),
                root_data=root_data,
                skip_first_n_frames=10,
                drop_last_n_frames=10,
                flatten_rpe=False,
                correct_transversal=True,
                use_encoded_video=self.use_compressed,
                cache_dir=cache_dir,
                verbose=False,
            )
            masks = rv.registered_masks
            video = rv.registered_frames
        except Exception as e:
            self.status_msg = f"Failed to load/register {rel}: {e}"
            return

        if video.shape != masks.shape:
            self.status_msg = f"Shape mismatch: {video.shape} vs {masks.shape}"
            return

        self._raw_masks = masks
        self._raw_video = video
        self._rebuild_display()

        self.idx = 0
        self.playing = True
        src = "mp4" if self.use_compressed else "bin"
        self.status_msg = f"Loaded {rel}  [{src}{', cached' if self.use_cache else ''}]"

    def _rebuild_display(self):
        """Re-derive self.masks/self.video + chart features from _raw_* + current smoothing settings."""
        if self._raw_masks is None:
            return

        self.status_msg = "Processing..."
        self._flush_status()

        if self.apply_smoothing:
            bm, csi = extract_boundaries_gpu(self._raw_masks)
            csi_smooth = smooth_boundary_2d(
                csi, sigma_time=self.sigma_time, sigma_col=self.sigma_col
            )
            self.masks = rebuild_mask(bm, csi_smooth, H=self._raw_masks.shape[1])
        else:
            self.masks = self._raw_masks

        self.video = self._raw_video

        # Compute features for charts. Thickness is the BM→CSI span (fast,
        # GPU boundary extraction) rather than a per-frame distance transform.
        bm_f, csi_f = extract_boundaries_gpu(self.masks)
        self._features_cache = {
            "delta_y": compute_deltaY_masks(self.masks),
            "thickness": compute_deltaY_boundaries(bm_f, csi_f),
            "area": self.masks.sum(axis=(1, 2)).astype(np.float32),
        }

        # Invalidate render caches; frames are composed lazily in draw_viewer.
        self._render_version += 1
        self._frame_key = None
        self._chart_surface = None

    def _flush_status(self):
        self.screen.fill(BG)
        self._draw_sidebar_bg()
        self.draw_viewer()
        self.draw_controls()
        self.ui.draw_ui(self.screen)
        pygame.display.flip()
        pygame.event.pump()

    # -- Drawing -------------------------------------------------------------

    def _draw_sidebar_bg(self):
        w, h = self.screen.get_size()
        pygame.draw.rect(self.screen, PANEL, pygame.Rect(0, 0, SIDEBAR_W, h))
        pygame.draw.line(
            self.screen, PANEL_BORDER, (SIDEBAR_W - 1, 0), (SIDEBAR_W - 1, h), 1
        )

    def viewer_rect(self) -> pygame.Rect:
        w, h = self.screen.get_size()
        chart_w = CHART_W if self._features_cache is not None else 0
        return pygame.Rect(SIDEBAR_W, 0, w - SIDEBAR_W - chart_w, h - CONTROL_H)

    def chart_rect(self) -> pygame.Rect:
        w, h = self.screen.get_size()
        return pygame.Rect(w - CHART_W, 0, CHART_W, h - CONTROL_H)

    def control_rect(self) -> pygame.Rect:
        w, h = self.screen.get_size()
        return pygame.Rect(SIDEBAR_W, h - CONTROL_H, w - SIDEBAR_W, CONTROL_H)

    @property
    def n_frames(self) -> int:
        return 0 if self.video is None else self.video.shape[0]

    def fit_frame_rect(self) -> pygame.Rect:
        area = self.viewer_rect()
        if self.video is None:
            return area
        H, W = self.video.shape[1], self.video.shape[2]
        aspect = W / H
        area_aspect = area.w / area.h
        if area_aspect > aspect:
            h = area.h - 20
            w = int(h * aspect)
        else:
            w = area.w - 20
            h = int(w / aspect)
        x = area.x + (area.w - w) // 2
        y = area.y + (area.h - h) // 2
        return pygame.Rect(x, y, w, h)

    def draw_viewer(self):
        area = self.viewer_rect()
        pygame.draw.rect(self.screen, BG, area)

        if self.video is None:
            msg = "Select a mask from the sidebar"
            surf = self.font_lg.render(msg, True, TEXT_FAINT)
            self.screen.blit(
                surf,
                (
                    area.centerx - surf.get_width() // 2,
                    area.centery - surf.get_height() // 2,
                ),
            )
            return

        target = self.fit_frame_rect()
        # Recompose + rescale only when the frame or display params change.
        key = (
            self.idx,
            target.w,
            target.h,
            self.show_mask,
            round(self.mask_alpha, 2),
            self._render_version,
        )
        if key != self._frame_key or self._frame_surf is None:
            mask = None if self.masks is None else self.masks[self.idx]
            rgb = compose_overlay_frame(
                self.video[self.idx], mask, self.show_mask, self.mask_alpha
            )
            surf = np_to_surface(rgb)
            self._frame_surf = pygame.transform.smoothscale(surf, (target.w, target.h))
            self._frame_key = key
        self.screen.blit(self._frame_surf, target.topleft)

    def draw_charts(self):
        if self._features_cache is None:
            return

        area = self.chart_rect()
        pygame.draw.rect(self.screen, BG, area)
        pygame.draw.line(self.screen, PANEL_BORDER, (area.x, 0), (area.x, area.h), 1)

        # Re-render the (expensive) matplotlib chart only when data/size change.
        if self._chart_surface is None:
            self._chart_surface, self._chart_xmap = render_charts(
                self._features_cache["delta_y"],
                self._features_cache["thickness"],
                self._features_cache["area"],
                width_px=area.w,
                height_px=area.h // 3,
            )

        self.screen.blit(self._chart_surface, area.topleft)

        # Playhead: a cheap vertical line at the current frame (no re-render).
        n = self.n_frames
        if n > 1 and self._chart_xmap is not None:
            x0, x1 = self._chart_xmap
            px = int(area.x + x0 + (self.idx / (n - 1)) * (x1 - x0))
            pygame.draw.line(
                self.screen, TEXT, (px, area.y + 4), (px, area.y + area.h - 4), 1
            )

    def draw_controls(self):
        area = self.control_rect()
        pygame.draw.rect(self.screen, PANEL, area)
        pygame.draw.line(
            self.screen, PANEL_BORDER, (area.x, area.y), (area.right, area.y), 1
        )

        pad = 20
        bar_h = 6
        bar_rect = pygame.Rect(area.x + pad, area.y + 14, area.w - 2 * pad, bar_h)
        pygame.draw.rect(self.screen, PANEL_BORDER, bar_rect, border_radius=3)

        if self.video is not None:
            n = self.n_frames
            progress = self.idx / max(1, n - 1)
            fill_w = int(bar_rect.w * progress)
            if fill_w > 0:
                fill = pygame.Rect(bar_rect.x, bar_rect.y, fill_w, bar_h)
                pygame.draw.rect(self.screen, ACCENT, fill, border_radius=3)
            knob = (bar_rect.x + fill_w, bar_rect.y + bar_h // 2)
            pygame.draw.circle(self.screen, TEXT, knob, 7)
            pygame.draw.circle(self.screen, ACCENT, knob, 5)

        self._bar_rect = bar_rect

        y = area.y + 36
        if self.video is not None:
            n = self.n_frames
            status = "PLAYING" if self.playing else "PAUSED"
            left = f"Frame {self.idx} / {n - 1}"
            smooth_info = (
                f"  σt={self.sigma_time:.1f}  σx={self.sigma_col:.1f}"
                if self.apply_smoothing
                else "  (no smoothing)"
            )
            center = f"{status}   -   {self.fps} fps{smooth_info}"
            right = ""
            # Add opacity info and if mask is off
            if self.show_mask:
                right += f"Mask α={self.mask_alpha:.2f}  "
        else:
            left, center, right = "", self.status_msg, ""

        if left:
            self.screen.blit(self.font.render(left, True, TEXT), (area.x + pad, y))
        csurf = self.font.render(center, True, TEXT_DIM)
        self.screen.blit(csurf, (area.centerx - csurf.get_width() // 2, y))
        if right:
            rsurf = self.font.render(right, True, TEXT_DIM)
            self.screen.blit(rsurf, (area.right - rsurf.get_width() - pad, y))

    # -- Events --------------------------------------------------------------

    def handle_ui_event(self, event):
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element is self.refresh_btn:
                self.refresh_entries()
            elif event.ui_element is self.compressed_btn:
                self.use_compressed = not self.use_compressed
                self.compressed_btn.set_text(
                    self._toggle_label("Prefer compressed (mp4)", self.use_compressed)
                )
                self._update_toggle_style(self.compressed_btn, self.use_compressed)
            elif event.ui_element is self.cache_btn:
                self.use_cache = not self.use_cache
                self.cache_btn.set_text(
                    self._toggle_label("Use registration cache", self.use_cache)
                )
                self._update_toggle_style(self.cache_btn, self.use_cache)
            elif event.ui_element is self.smooth_btn:
                self.apply_smoothing = not self.apply_smoothing
                self.smooth_btn.set_text(
                    self._toggle_label("Apply smoothing", self.apply_smoothing)
                )
                self._update_toggle_style(self.smooth_btn, self.apply_smoothing)
                self._rebuild_display()

        elif event.type == pygame_gui.UI_SELECTION_LIST_NEW_SELECTION:
            if event.ui_element is self.entry_list:
                for rel in self._filtered_entries:
                    if self._format_entry(rel, self._entry_info[rel]) == event.text:
                        self.selected_rel = rel
                        self.load_selected()
                        break

        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element is self.sigma_time_slider:
                self.sigma_time = float(event.value)
                self.sigma_time_label.set_text(f"sigma_time: {self.sigma_time:.1f}")
                if self.apply_smoothing:
                    self._rebuild_display()
            elif event.ui_element is self.sigma_col_slider:
                self.sigma_col = float(event.value)
                self.sigma_col_label.set_text(f"sigma_col: {self.sigma_col:.1f}")
                if self.apply_smoothing:
                    self._rebuild_display()

        elif event.type == pygame_gui.UI_TEXT_ENTRY_CHANGED:
            if event.ui_element is self.filter_entry:
                self._rebuild_list()

        elif event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
            if event.ui_element in (self.input_entry, self.output_entry):
                self.refresh_entries()
            elif event.ui_element is self.filter_entry:
                self._rebuild_list()

    def handle_custom_event(self, event) -> bool:
        if event.type == pygame.QUIT:
            return False

        elif event.type == pygame.WINDOWSIZECHANGED:
            self.ui.set_window_resolution(self.screen.get_size())
            self._relayout_sidebar()
            # Cached surfaces are size-specific; force a re-render at new size.
            self._chart_surface = None
            self._frame_key = None

        elif event.type == pygame.KEYDOWN:
            if (
                self.input_entry.is_focused
                or self.output_entry.is_focused
                or self.filter_entry.is_focused
            ):
                return True

            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                return False
            elif event.key == pygame.K_SPACE and self.video is not None:
                self.playing = not self.playing
            elif event.key == pygame.K_RIGHT and self.video is not None:
                self.idx = min(self.idx + 1, self.n_frames - 1)
                self.playing = False
            elif event.key == pygame.K_LEFT and self.video is not None:
                self.idx = max(self.idx - 1, 0)
                self.playing = False
            elif event.key == pygame.K_UP:
                self.fps = min(self.fps + 5, 120)
            elif event.key == pygame.K_DOWN:
                self.fps = max(self.fps - 5, 1)
            elif event.key == pygame.K_r and self.video is not None:
                self.idx = 0
            elif event.key == pygame.K_F5:
                self.refresh_entries()
            elif event.key == pygame.K_o:
                self.show_mask = not self.show_mask
            elif event.key == pygame.K_PLUS:
                self.mask_alpha = min(self.mask_alpha + 0.05, 1.0)
            elif event.key == pygame.K_MINUS:
                self.mask_alpha = max(self.mask_alpha - 0.05, 0.0)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if (
                self._bar_rect
                and self._bar_rect.collidepoint(event.pos)
                and self.video is not None
            ):
                rel = (event.pos[0] - self._bar_rect.x) / self._bar_rect.w
                self.idx = int(np.clip(rel, 0, 1) * (self.n_frames - 1))
                self.playing = False

        return True

    # -- Main loop -----------------------------------------------------------

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(120) / 1000.0
            events = pygame.event.get()

            for event in events:
                if not self.handle_custom_event(event):
                    running = False
                    break
                self.handle_ui_event(event)
                self.ui.process_events(event)

            if not running:
                break

            now = pygame.time.get_ticks()
            if (
                self.playing
                and self.video is not None
                and now - self.last_tick >= 1000 / self.fps
            ):
                self.idx = (self.idx + 1) % self.n_frames
                self.last_tick = now

            self.ui.update(dt)

            self.screen.fill(BG)
            self._draw_sidebar_bg()
            self.draw_viewer()
            self.draw_charts()
            self.draw_controls()
            self.ui.draw_ui(self.screen)

            pygame.display.flip()

        pygame.quit()


# --- CLI --------------------------------------------------------------------


def main():
    default_input = os.environ.get("OCT_INPUT_ROOT", ROOT_DATA_MNT)
    default_output = os.environ.get("OCT_OUTPUT_ROOT", str(ROOT_MASKS))

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=default_input)
    parser.add_argument("--output-root", default=default_output)
    args = parser.parse_args()

    App(args.input_root, args.output_root).run()


if __name__ == "__main__":
    main()
