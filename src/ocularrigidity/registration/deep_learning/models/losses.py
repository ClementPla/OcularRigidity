"""Direct-optimisation (unsupervised) objectives for the registration regressor.

Regressing the classical pipeline's ``transform.npz`` treats it as truth, which
it is not: ``dx`` is a band-passed phase-correlation peak and ``dy`` is
``BM(moving) - BM(fixed)`` read off a segmentation mask, so the target carries
every failure of the estimator it came from. Here the network is scored on *what
registration is for* instead: after warping the moving frame by the predicted
``(dx, dy)``, does it line up with the fixed frame?

Three complementary similarity terms, all differentiable w.r.t. ``(dx, dy)``:

``feature_similarity_loss``
    Cosine similarity of the frozen MiT pyramid, one term per scale, the moving
    features warped by the prediction. Coarse scales have a wide basin of
    attraction (stride 32 => one feature pixel is 32 image pixels, so a 50 px
    error is under two cells); fine scales localise. This is the multiscale
    encoder doing what it is good at, used as the metric rather than the input.

``photometric_loss``
    Masked local NCC on a band-passed image pyramid. Feature maps at stride >= 4
    cannot see a 1-2 px lateral shift, and OCT layers are locally horizontal, so
    ``dx`` is only observable in the speckle -- which lives at full resolution
    and in the high frequencies. This is the term that makes ``dx`` identifiable
    (same reasoning as the band-pass in ``lateral/correlation.py``).

``bm_alignment_loss``
    Soft Bruch's-membrane row per column, from the frozen segmentation head, on
    fixed vs warped-moving. This *is* the cohort QC metric: ``bm_jitter_p95`` in
    ``flag_misregistration.py`` is the frame-to-frame spread of exactly this
    curve. Optional (it costs a decoder pass), but it is the closest thing to
    optimising the acceptance criterion directly.

Plus two priors that stop the field from cheating: smoothness along W, and a
coverage penalty, since any similarity averaged over the valid region alone can
be lowered by pushing most of the frame out of view.

Warp convention -- identical to ``registration/rigid.py``, which applies the
lateral shift first and then the per-column vertical one::

    out(x, y) = moving(x - dx, y + dy(x))

so a positive ``dy`` samples from lower down (content moves up), and ``dy`` is
indexed by the *output* column. Displacements are always expressed in full-
resolution pixels; :func:`warp` rescales them to whatever map it is given.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "warp",
    "masked_lncc",
    "photometric_loss",
    "lateral_profile_loss",
    "feature_similarity_loss",
    "soft_bm_rows",
    "bm_alignment_loss",
    "smoothness_loss",
    "cycle_consistency_loss",
    "shift_equivariance_loss",
    "UnsupervisedRegistrationLoss",
]


# --------------------------------------------------------------------------- #
# Warping
# --------------------------------------------------------------------------- #
def _resample_dy(dy: torch.Tensor, width: int) -> torch.Tensor:
    """Resample a per-column displacement ``(B, W)`` onto ``width`` columns."""
    if dy.shape[-1] == width:
        return dy
    return F.interpolate(
        dy.unsqueeze(1), size=width, mode="linear", align_corners=False
    ).squeeze(1)


def warp(
    moving: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    ref_shape: Tuple[int, int],
    *,
    padding_mode: str = "zeros",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply ``out(x, y) = moving(x - dx, y + dy(x))`` to ``moving`` (B, C, h, w).

    ``dx`` (B,) and ``dy`` (B, W) are in pixels of the full-resolution grid
    ``ref_shape = (H, W)``; they are rescaled to the resolution of ``moving``,
    so the same prediction warps the image and every level of the pyramid.

    Returns the warped map and a validity mask (B, 1, h, w) — 1 where the sample
    came from inside ``moving``, fractional on the border. Every similarity term
    below is averaged over that mask: pixels sampled from the zero padding say
    nothing about alignment, and scoring them would reward shifting content out.
    """
    B, _, h, w = moving.shape
    H, W = ref_shape
    sy, sx = h / H, w / W
    device, dtype = moving.device, moving.dtype

    dyw = (_resample_dy(dy, w) * sy).to(dtype).view(B, 1, w)  # rows of this map
    dxw = (dx * sx).to(dtype).view(B, 1, 1)

    ys = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)
    xs = torch.arange(w, device=device, dtype=dtype).view(1, 1, w)

    gx = (xs - dxw).expand(B, h, w)
    gy = (ys + dyw).expand(B, h, w)
    grid = torch.stack([gx / max(w - 1, 1) * 2 - 1, gy / max(h - 1, 1) * 2 - 1], dim=-1)

    out = F.grid_sample(
        moving, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True
    )
    ones = torch.ones((B, 1, h, w), device=device, dtype=dtype)
    valid = F.grid_sample(
        ones, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return out, valid


# --------------------------------------------------------------------------- #
# Image-space similarity
# --------------------------------------------------------------------------- #
def _box(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)


def _bandpass(x: torch.Tensor, k: int = 15) -> torch.Tensor:
    """Remove the local mean: kills the slow intensity gradient that dominates a
    B-scan and leaves the speckle/edge content the shift is measurable in."""
    return x - _box(x, k)


def masked_lncc(
    a: torch.Tensor,
    b: torch.Tensor,
    valid: torch.Tensor,
    k: int = 9,
    eps: float = 1e-4,
    contrast_weighted: bool = True,
) -> torch.Tensor:
    """1 - local NCC between ``a`` and ``b`` (B, 1, h, w), weighted by ``valid``.

    Local (window ``k``) rather than global: OCT brightness varies across the
    frame, and a global NCC is dominated by the bright RPE band whichever way
    the frame is shifted.

    ``contrast_weighted`` weights each window by ``sqrt(var_a var_b)`` instead of
    counting every non-flat window equally. Most of a B-scan is vitreous — dark,
    textureless, and perfectly correlated with itself at *any* shift — so a flat
    average buries the signal.
    """
    w = valid
    n = _box(w, k).clamp_min(eps)
    mu_a = _box(w * a, k) / n
    mu_b = _box(w * b, k) / n
    var_a = (_box(w * a * a, k) / n - mu_a**2).clamp_min(0)
    var_b = (_box(w * b * b, k) / n - mu_b**2).clamp_min(0)
    cov = _box(w * a * b, k) / n - mu_a * mu_b
    ncc = cov / torch.sqrt(var_a * var_b + eps)
    if contrast_weighted:
        # detached: this chooses *where* to look, and letting the warp lower the
        # loss by steering the weights onto agreeable windows is another way of
        # cheating.
        weight = w * torch.sqrt(var_a * var_b + 1e-12).detach()
    else:
        # A constant window has an undefined correlation; there are many.
        weight = w * (var_a > eps).to(a.dtype) * (var_b > eps).to(a.dtype)
    return 1.0 - (ncc * weight).sum() / weight.sum().clamp_min(1e-6)


def photometric_loss(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    *,
    levels: int = 3,
    window: int = 9,
    bandpass: bool = True,
) -> torch.Tensor:
    """Masked LNCC on a ``levels``-deep average-pooled pyramid of the images."""
    H, W = fixed.shape[-2:]
    total = fixed.new_zeros(())
    a_l, m_l = fixed, moving
    for lvl in range(levels):
        warped, valid = warp(m_l, dx, dy, (H, W))
        a = _bandpass(a_l) if bandpass else a_l
        b = _bandpass(warped) if bandpass else warped
        total = total + masked_lncc(a, b, valid, k=window)
        if lvl < levels - 1:
            a_l = F.avg_pool2d(a_l, 2)
            m_l = F.avg_pool2d(m_l, 2)
    return total / levels


def lateral_profile_loss(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    *,
    smooth: int = 31,
    drop_edges: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """1 - NCC between the vertical-mean intensity profiles, after the warp."""
    warped, valid = warp(moving, dx, dy, fixed.shape[-2:])
    # Validity-weighted column means: a column half-filled with zero padding
    # must not be compared against a full one.
    prof_f = fixed.mean(dim=2)  # (B, 1, W)
    prof_w = (warped * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(eps)

    if smooth > 1:  # remove the slow envelope, keep the lateral structure
        pad = smooth // 2
        prof_f = prof_f - F.avg_pool1d(prof_f, smooth, 1, pad, count_include_pad=False)
        prof_w = prof_w - F.avg_pool1d(prof_w, smooth, 1, pad, count_include_pad=False)
    if drop_edges > 0:
        prof_f = prof_f[..., drop_edges:-drop_edges]
        prof_w = prof_w[..., drop_edges:-drop_edges]

    a = prof_f - prof_f.mean(dim=-1, keepdim=True)
    b = prof_w - prof_w.mean(dim=-1, keepdim=True)
    ncc = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)
    return 1.0 - ncc.mean()


# --------------------------------------------------------------------------- #
# Feature-space similarity (the frozen encoder as the metric)
# --------------------------------------------------------------------------- #
def feature_similarity_loss(
    fixed_feats: Sequence[torch.Tensor],
    moving_feats: Sequence[torch.Tensor],
    dx: torch.Tensor,
    dy: torch.Tensor,
    ref_shape: Tuple[int, int],
    *,
    weights: Optional[Sequence[float]] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """1 - cosine similarity per scale, on channel-centred features.

    Warping the *features* rather than re-encoding the warped image is the
    PWC-Net / RAFT trade: a convolutional pyramid is close enough to
    translation-equivariant that a shifted feature map matches the feature map
    of the shifted image, and it costs one ``grid_sample`` instead of a second
    encoder pass. Centring each channel over the map first matters — raw MiT
    features have a large DC component and cosine similarity on it is ~1
    everywhere, whatever the alignment.
    """
    n = len(fixed_feats)
    if weights is None:
        weights = [1.0] * n
    total = fixed_feats[0].new_zeros(())
    per_scale: List[float] = []
    for f, m, wgt in zip(fixed_feats, moving_feats, weights):
        warped, valid = warp(m, dx, dy, ref_shape)
        f = f - f.mean(dim=(2, 3), keepdim=True)
        warped = warped - warped.mean(dim=(2, 3), keepdim=True)
        sim = (F.normalize(f, dim=1) * F.normalize(warped, dim=1)).sum(
            dim=1, keepdim=True
        )
        term = 1.0 - (sim * valid).sum() / valid.sum().clamp_min(1.0)
        per_scale.append(float(term.detach()))
        total = total + wgt * term
    return total / max(sum(weights), 1e-6), per_scale


def soft_bm_rows(prob: torch.Tensor) -> torch.Tensor:
    """Expected row of the first mask entry per column, from a soft mask.

    ``extract_boundaries_fast`` takes the first ``True`` going down each column.
    Its differentiable analogue: with ``c = clamp(cumsum(p), max=1)`` a step from
    0 to 1 at row ``r``, ``sum(1 - c) == r`` exactly, and it degrades gracefully
    for a soft ``p``. ``prob`` is (B, 1, H, W); returns (B, W) in rows.
    """
    c = torch.cumsum(prob, dim=2).clamp(max=1.0)
    return (1.0 - c).sum(dim=2).squeeze(1)


def bm_alignment_loss(
    prob_fixed: torch.Tensor,
    prob_warped: torch.Tensor,
    valid_cols: torch.Tensor,
    *,
    delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Huber on the soft-BM row difference, over columns present in both masks.

    Returns ``(loss, mean |ΔBM| in px)`` — the second value is directly
    comparable to ``BM_JITTER_P95_PX`` in ``flag_misregistration.py``.
    """
    bm_f = soft_bm_rows(prob_fixed)
    bm_w = soft_bm_rows(prob_warped)
    diff = (bm_f - bm_w) * valid_cols
    n = valid_cols.sum().clamp_min(1.0)
    abs_d = diff.abs()
    huber = torch.where(abs_d < delta, 0.5 * diff**2 / delta, abs_d - 0.5 * delta)
    return huber.sum() / n, (abs_d.sum() / n).detach()


# --------------------------------------------------------------------------- #
# Priors
# --------------------------------------------------------------------------- #
def smoothness_loss(dy: torch.Tensor, *, order: int = 1, eps: float = 1e-3):
    """Charbonnier penalty on the ``order``-th difference of ``dy`` along W.

    The true field is ``BM(moving) - BM(fixed)``: a smooth, low-order curve. Left
    free, a per-column displacement will happily carve out a jagged field that
    matches speckle noise column by column.
    """
    d = dy
    for _ in range(order):
        d = d[:, 1:] - d[:, :-1]
    return torch.sqrt(d**2 + eps**2).mean()


def coverage_loss(valid: torch.Tensor, ref_h: float = 1536.0) -> torch.Tensor:
    """Penalise pushing the frame out of view.

    Every similarity above is a mean over the valid region, so a large shift that
    leaves one well-matched sliver on screen scores well.

    ``ref_h`` keeps the penalty proportional to the displacement in *pixels*
    rather than to the fraction of frame it happens to occupy. Un-normalised,
    the invalid fraction is ``|dy| / H``, so the same physical shift costs twice
    as much on a 768-row crop as on the full 1536-row frame, biasing dy toward
    zero twice as hard purely because of the crop. A no-op at full resolution.
    """
    return (1.0 - valid.mean()) * (valid.shape[-2] / ref_h)


def cycle_consistency_loss(
    dx_ij: torch.Tensor, dx_jk: torch.Tensor, dx_ik: torch.Tensor
) -> torch.Tensor:
    """Penalise ``(dx_ij + dx_jk) - dx_ik`` for a closed triplet ``(i, j, k)``
    of frames from one video, each ``(B,)``.

    A lateral shift is a pure translation, so for any three frames of the same
    eye it composes exactly regardless of what happens in between --
    ``dx(i, k) == dx(i, j) + dx(j, k)`` -- whatever the true displacement is.
    Checking that identity needs no ground truth, only three ordinary forward
    passes on frames the pair sampler already visits, which is what makes it
    cheap to add on top of the direct-optimisation objective above.

    Only ``dx``: unlike a lateral shift, ``dy`` composition would have to
    account for the column re-indexing a nonzero ``dx`` induces (the same
    caveat ``FramePairDataset`` documents for its own ground-truth difference),
    which is not worth the complexity for a term aimed at the one quantity
    (``dx``) that direct optimisation leaves chronically underdetermined --
    see ``lateral_profile_loss``.
    """
    return F.smooth_l1_loss(dx_ij + dx_jk, dx_ik)


def shift_equivariance_loss(
    dx_ref: torch.Tensor,
    dx_shifted: torch.Tensor,
    delta: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Require dx to track a *known* synthetic lateral shift, one-for-one.

    Shifting the moving frame's content right by ``delta`` columns changes the
    displacement that aligns it by exactly ``-delta``:  the warp reads
    ``out(x) = moving(x - dx)``, so with ``moving'(u) = moving(u - delta)``,
    ``moving'(x - dx') == moving(x - dx)`` requires ``dx' = dx - delta``.
    Hence ``dx_ref - dx_shifted == delta``, exactly, for whatever the true
    underlying displacement happens to be.

    This is the one signal for dx that needs no estimate of the truth: the
    shift is applied by us, in whole columns, so it is exact and free of the
    classical estimator entirely. Crucially it also has no degenerate
    solution -- a collapsed head (``dx`` constant, in particular ``dx == 0``)
    gives ``dx_ref - dx_shifted == 0 != delta`` and is penalised hard, which
    is exactly what :func:`cycle_consistency_loss` on its own cannot do.

    The two are complementary and neither replaces the other:

    * this term fixes the *gain* -- dx must move 1:1 with real pixels -- but
      is blind to a constant offset, since ``c`` cancels in the difference;
    * ``cycle_consistency_loss`` removes that offset, because adding ``c`` to
      every prediction turns ``d_ij + d_jk == d_ik`` into ``d_ik + 2c``, which
      only holds for ``c == 0``; but it is blind to the gain, and is minimised
      perfectly by the collapse this term forbids.
    """
    # beta=1 (the default) on purpose. What this term corrects early on is a
    # *gain* -- the head tracking only a fraction of the applied displacement --
    # and an error of several pixels then sits in smooth_l1's linear regime,
    # where the gradient is a constant 1. That is the desirable behaviour here:
    # it keeps pushing at full strength until the gain is nearly right, and the
    # reported value stays in pixels, directly readable as the mean miss. A
    # wider beta would make the gradient proportional to the error, which sounds
    # better conditioned but is uniformly *weaker* over this range (0.84 vs 1.0
    # at gain 0.16) and shrinks the term's share of the total ~4x.
    return F.smooth_l1_loss(dx_ref - dx_shifted, delta, beta=beta)


# --------------------------------------------------------------------------- #
# Assembled objective
# --------------------------------------------------------------------------- #
class UnsupervisedRegistrationLoss(nn.Module):
    """Weighted sum of the terms above; returns ``(loss, components)``.

    ``segmenter`` is the frozen segmentation module (``ChoroidSegmentationModule``).
    When given and ``w_bm > 0``, the BM term is enabled: it needs one decoder
    pass over the fixed frame and one over the warped moving frame, with grad
    flowing through the (frozen) network back to the warp.

    The BM term is on by default because it is by far the most discriminative
    of the three.

    ``w_lateral`` carries dx on its own.

    The BM term is also the memory-hungry one: a MiT encoder pass over a 1536x1024 frame
    with its activations held for the backward. ``bm_scale`` segments a
    downscaled pair instead (0.5 cuts it fourfold); BM localisation at half
    resolution is still well inside the 3 px cohort threshold.

    ``coarse_to_fine`` ramps the per-scale feature weights: early in training
    only the coarse scales count (wide basin, no local minima to fall into),
    and the fine scales are phased in as the prediction gets close. Call
    :meth:`set_progress` once per epoch with a value in [0, 1].
    """

    def __init__(
        self,
        *,
        n_scales: int = 4,
        w_feature: float = 1.0,
        w_photometric: float = 1.0,
        w_lateral: float = 1.0,
        w_bm: float = 0.1,
        w_smooth: float = 0.05,
        w_curvature: float = 0.02,
        w_coverage: float = 0.5,
        photometric_levels: int = 3,
        photometric_window: int = 9,
        coarse_to_fine: bool = True,
        segmenter: Optional[nn.Module] = None,
        bm_scale: float = 1.0,
        bm_on_features: bool = False,
        bm_max_shift_px: float = 154.0,
    ):
        super().__init__()
        self.n_scales = n_scales
        self.w_feature = w_feature
        self.w_photometric = w_photometric
        self.w_lateral = w_lateral
        self.w_bm = w_bm
        self.w_smooth = w_smooth
        self.w_curvature = w_curvature
        self.w_coverage = w_coverage
        self.photometric_levels = photometric_levels
        self.photometric_window = photometric_window
        self.coarse_to_fine = coarse_to_fine
        self.segmenter = segmenter
        self.bm_scale = bm_scale
        self.bm_on_features = bm_on_features
        self.bm_max_shift_px = bm_max_shift_px
        self._progress = 1.0

    def _decode_bm(self, feats4, img: torch.Tensor) -> torch.Tensor:
        """Segmentation logits from a *pyramid*, skipping the encoder.

        ``segmentation_head(decoder(encoder(x)))`` reproduces ``model(x)``
        exactly (asserted by the tests behind ``fused.py``), and ``encoder(x)``
        is ``[x, empty, *forward_features(x)]``. The Segformer decoder reads
        only the four real scales -- swapping the first two entries for
        anything at all leaves its output bit-identical -- so the pyramid the
        regressor already holds is everything it needs. ``img`` and an empty
        stub stand in for those two ignored entries, purely to satisfy
        indexing.
        """
        seg = self.segmenter.model
        b, _, h, w = img.shape
        stub = img.new_zeros(b, 0, h // 2, w // 2)
        return seg.segmentation_head(seg.decoder([img, stub] + list(feats4)))

    def set_progress(self, progress: float) -> None:
        self._progress = float(min(max(progress, 0.0), 1.0))

    def scale_weights(self) -> List[float]:
        """Fine->coarse weights. Scale 0 is finest; it is the last to switch on."""
        if not self.coarse_to_fine:
            return [1.0] * self.n_scales
        # Fine scale i is fully on once progress passes i / n_scales.
        out = []
        for i in range(self.n_scales):
            onset = (self.n_scales - 1 - i) / self.n_scales
            out.append(
                float(min(max((self._progress - onset) * self.n_scales, 0.0), 1.0))
            )
        # The coarsest scale is always on, so the objective is never empty.
        out[-1] = 1.0
        return out

    def forward(
        self,
        dx: torch.Tensor,
        dy: torch.Tensor,
        *,
        fixed_img: torch.Tensor,
        moving_img: torch.Tensor,
        fixed_feats: Sequence[torch.Tensor],
        moving_feats: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        H, W = fixed_img.shape[-2:]
        comp: Dict[str, float] = {}
        loss = dx.new_zeros(())

        if self.w_feature > 0:
            l_feat, per_scale = feature_similarity_loss(
                fixed_feats, moving_feats, dx, dy, (H, W), weights=self.scale_weights()
            )
            loss = loss + self.w_feature * l_feat
            comp["feat"] = float(l_feat.detach())
            for i, v in enumerate(per_scale):
                comp[f"feat{i}"] = v

        if self.w_photometric > 0:
            l_photo = photometric_loss(
                fixed_img,
                moving_img,
                dx,
                dy,
                levels=self.photometric_levels,
                window=self.photometric_window,
            )
            loss = loss + self.w_photometric * l_photo
            comp["photo"] = float(l_photo.detach())

        if self.w_lateral > 0:
            l_lat = lateral_profile_loss(fixed_img, moving_img, dx, dy)
            loss = loss + self.w_lateral * l_lat
            comp["lat"] = float(l_lat.detach())

        # The image warp is only needed for the BM term; coverage needs the
        # validity mask alone, which is one channel instead of the frame.
        use_bm = self.w_bm > 0 and self.segmenter is not None
        feature_bm = use_bm and self.bm_on_features
        valid = None
        if use_bm and not feature_bm:
            warped_img, valid = warp(moving_img, dx, dy, (H, W))
        elif use_bm or self.w_coverage > 0:
            # The feature path never needs the warped *frame*, only the single
            # channel of validity -- a grid_sample and nothing more.
            _, valid = warp(torch.ones_like(moving_img[:, :1]), dx, dy, (H, W))

        if use_bm:
            if feature_bm:
                # OFF BY DEFAULT -- measured to hurt. Warping the pyramid and
                # decoding it saves the encoder pass (~92% of the segmenter's
                # cost) and agrees with the image path on the *value* of bm_px
                # to ~0.2 px near convergence. But dy is driven by the
                # gradient, and swept around the correct dy this path gives
                # |d l_bm / d dy| ~1.1e-3 flat, against 3.3e-3 rising to 8.5e-3
                # at +-120 px for the image path: 3-7x weaker, and with none of
                # the growth with distance that drags large offsets in. The
                # gradient of a stride-4..32 feature map is simply smoother
                # than that of the full-resolution frame. Enable only where
                # bm_px is wanted as a cheap *measurement*, not as a driver.
                warped_feats = [warp(f, dx, dy, (H, W))[0] for f in moving_feats]
                v_in = valid
                with torch.no_grad():
                    prob_f = torch.sigmoid(self._decode_bm(fixed_feats, fixed_img))
                prob_w = torch.sigmoid(self._decode_bm(warped_feats, moving_img))
            else:
                f_in, w_in, v_in = fixed_img, warped_img, valid
                if self.bm_scale != 1.0:
                    kw = dict(
                        scale_factor=self.bm_scale, mode="bilinear", align_corners=False
                    )
                    f_in = F.interpolate(f_in, **kw)
                    w_in = F.interpolate(w_in, **kw)
                    v_in = F.interpolate(v_in, **kw)
                # The fixed frame does not depend on (dx, dy), so its
                # segmentation carries no gradient.
                with torch.no_grad():
                    prob_f = torch.sigmoid(self.segmenter(f_in))
                prob_w = torch.sigmoid(self.segmenter(w_in))

            # A column is scorable where both masks exist and the warp did not
            # pull most of it in from outside the moving frame. The test is on
            # the *fraction* of the column in view, not its worst row: any
            # vertical shift invalidates the top or bottom rows, so requiring
            # every row to be valid scores only the columns whose displacement
            # is ~0 -- silently zeroing the term for exactly the predictions it
            # is meant to penalise.
            # How much of a column may have been pulled in from outside before
            # it stops being scorable. State it in *pixels*: as a fraction it
            # silently tightens with the frame height, and a crop changes that
            # height. The old 0.9 was calibrated on the full 1536-row frame,
            # where it drops a column past |dy| = 154 px; reused verbatim on a
            # 768-row crop it drops one past 77 px, which kills the term
            # outright for the majority of pairs (median max|dy| is 87 px) --
            # switching off the most discriminative signal exactly where the
            # displacement is largest, and leaving dy with nothing driving it.
            h_rows = v_in.shape[2]
            min_valid = 1.0 - min(self.bm_max_shift_px / max(h_rows, 1), 0.5)
            cols = (
                (
                    (prob_f.amax(dim=2) > 0.5)
                    & (prob_w.amax(dim=2) > 0.5)
                    & (v_in.mean(dim=2) > min_valid)
                )
                .squeeze(1)
                .to(prob_f.dtype)
            )
            l_bm, bm_px = bm_alignment_loss(prob_f, prob_w, cols)
            # Rows are counted in whatever grid was segmented; report and
            # weight in full-resolution pixels so bm_px stays comparable to the
            # cohort threshold. The feature path decodes the pyramid at its own
            # full resolution, so its scale is 1.
            s_bm = 1.0 if feature_bm else self.bm_scale
            l_bm, bm_px = l_bm / s_bm, bm_px / s_bm
            loss = loss + self.w_bm * l_bm
            comp["bm"] = float(l_bm.detach())
            comp["bm_px"] = float(bm_px)

        if self.w_smooth > 0:
            l_s = smoothness_loss(dy, order=1)
            loss = loss + self.w_smooth * l_s
            comp["smooth"] = float(l_s.detach())
        if self.w_curvature > 0:
            l_c = smoothness_loss(dy, order=2)
            loss = loss + self.w_curvature * l_c
            comp["curv"] = float(l_c.detach())
        if self.w_coverage > 0:
            l_cov = coverage_loss(valid)
            loss = loss + self.w_coverage * l_cov
            comp["cov"] = float(l_cov.detach())

        comp["total"] = float(loss.detach())
        return loss, comp
