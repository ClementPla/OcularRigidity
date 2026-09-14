import torch
import numpy as np
import torch.nn.functional as F

from ocularrigidity.registration.layout import to_gray

DEBUG = False


def profile_correlation_dx(
    curve: torch.Tensor,
    ref_curve: torch.Tensor,
    max_shift: int = None,
    drop_edges: int = 75,
    subpixel: bool = True,
) -> torch.Tensor:
    """
    curve: T x W, ref_curve: W
    Returns dx: T, the sub-pixel lateral shift that best aligns
    curve[t] to ref_curve via 1D cross-correlation and parabolic interpolation.
    When ``subpixel`` is False the parabolic fit is skipped and the integer-pixel
    cross-correlation peak is returned directly.
    """
    T, W = curve.shape
    if max_shift is None:
        max_shift = W // 4

    curve = curve[:, drop_edges : W - drop_edges]
    ref_curve = ref_curve[drop_edges : W - drop_edges]
    T, W = curve.shape
    c = curve - curve.mean(dim=1, keepdim=True)
    r = ref_curve - ref_curve.mean()

    shifts = torch.arange(-max_shift, max_shift + 1, device=curve.device)
    num_shifts = len(shifts)
    scores = torch.zeros(T, num_shifts, device=curve.device)

    # 1. Compute integer cross-correlation scores
    for i, s in enumerate(shifts):
        s = int(s.item())
        if s >= 0:
            c_seg = c[:, : W - s] if s > 0 else c
            r_seg = r[s:] if s > 0 else r
        else:
            c_seg = c[:, -s:]
            r_seg = r[: W + s]
        scores[:, i] = (c_seg * r_seg).sum(dim=1)

    # 2. Find the best integer shift
    best_idx = scores.argmax(dim=1)
    dx = shifts[best_idx].float()

    # 3. Sub-pixel Interpolation (Parabolic Fit)
    # Ensure we only interpolate if the peak isn't touching the boundary edges
    valid_mask = (best_idx > 0) & (best_idx < num_shifts - 1)

    if subpixel and valid_mask.any():
        idx_valid = best_idx[valid_mask]

        # Get scores at the peak (y0) and its neighbors (y_minus_1, y_plus_1)
        y_minus_1 = scores[valid_mask, idx_valid - 1]
        y_0 = scores[valid_mask, idx_valid]
        y_plus_1 = scores[valid_mask, idx_valid + 1]

        # Parabolic interpolation formula
        denominator = 2 * (y_minus_1 - 2 * y_0 + y_plus_1)

        # Protect against division by zero (e.g., flat regions)
        nonzero_mask = denominator != 0

        offset = torch.zeros_like(y_0)
        offset[nonzero_mask] = (
            y_minus_1[nonzero_mask] - y_plus_1[nonzero_mask]
        ) / denominator[nonzero_mask]

        # Add the fractional offset to the integer shift
        dx[valid_mask] += offset

    return dx


def frame_correlation_dx(
    frames: torch.Tensor,
    ref: torch.Tensor = None,
    scale_factor=1.0,
    crop_factor=0.66,
    max_shift: int = 16,
    max_vshift: int = 512,
    batch_size: int = 256,
    device: str = "cuda",
    bandpass: tuple[float, float] = (0.02, 0.5),
    return_confidence: bool = False,
    subpixel: bool = True,
) -> torch.Tensor:
    # Colour in, luminance out: the phase correlation, the FFT windows and the
    # peak search below are all 2-D, and a lateral shift is a property of the
    # scene rather than of the channel it is measured in.
    frames = to_gray(frames)
    T, H, W = frames.shape

    h, w = int(H * scale_factor), int(W * scale_factor)
    center = w // 2
    crop_w = int(W * crop_factor)
    crop_x_start = (W - crop_w) // 2

    # Columns are the last axis, so this crops W for a (T, H, W) stack and for a
    # single (H, W) reference alike.
    frames = frames[..., crop_x_start : crop_x_start + crop_w]
    if ref is not None:
        ref = to_gray(torch.as_tensor(ref)[None])[0]
        ref = ref[..., crop_x_start : crop_x_start + crop_w]

    # Hann window kills FFT wrap-around edge artifacts.
    win = (
        torch.hann_window(h, periodic=False, device=device)[:, None]
        * torch.hann_window(w, periodic=False, device=device)[None, :]
    )

    # Reference: temporal median template (computed on a subsample for speed).
    if ref is None:
        idx = torch.linspace(0, T - 1, min(T, 64)).round().long()
        ref = frames[idx].to(device).float().median(dim=0).values
    else:
        ref = ref.to(device).float()
    r = F.interpolate(ref[None, None], size=(h, w), mode="area")[0, 0]

    r = (r - r.mean()) * win
    Fr_conj = torch.fft.rfft2(r)[None].conj()  # (1, h, w//2+1)

    # Restrict the peak search to +-max_shift (converted to downsampled pixels).
    max_shift_down = max(1, int(round(max_shift * w / crop_w)))
    positions = torch.arange(w, device=device) - center  # zero shift at `center`
    in_window = positions.abs() <= max_shift_down

    f_lo, f_hi = bandpass
    fy = torch.fft.fftfreq(h, device=device)[:, None]
    fx = torch.fft.rfftfreq(w, device=device)[None, :]
    fr = torch.sqrt(fy**2 + fx**2)
    band = ((1.0 - torch.exp(-((fr / f_lo) ** 2))) * torch.exp(-((fr / f_hi) ** 2)))[
        None
    ]
    y_center = h // 2
    yw = max(1, int(round(max_vshift * h / H)))
    y_lo, y_hi = max(0, y_center - yw), min(h, y_center + yw + 1)

    dx = torch.empty(T, device=device, dtype=torch.float32)
    conf = torch.empty(T, device=device, dtype=torch.float32)
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        x = frames[start:end].to(device).float()
        x = F.interpolate(x.unsqueeze(1), size=(h, w), mode="area").squeeze(1)
        x = (x - x.mean(dim=(-2, -1), keepdim=True)) * win

        cps = torch.fft.rfft2(x) * Fr_conj
        cps = cps / (cps.abs() + 1e-8)  # phase only
        cps = cps * band
        corr = torch.fft.irfft2(cps, s=(h, w))
        corr = torch.fft.fftshift(corr, dim=(-2, -1))  # zero shift at center

        # Localize the 2D correlation peak (within the vertical search band and
        # the lateral window), then take the x-line through it. Summing over y
        # would dilute a compact phase-correlation peak with many noisy rows
        # (~2-4x worse under speckle); the peak row keeps full SNR and absorbs any
        # residual vertical shift for free.
        nb = end - start
        region = corr[:, y_lo:y_hi, :].clone()
        region[:, :, ~in_window] = -float("inf")
        y_star = region.reshape(nb, -1).argmax(dim=1).div(w, rounding_mode="floor")
        y_star = y_star + y_lo  # (b,) row of the 2D peak
        raw_corr_x = corr[torch.arange(nb, device=device), y_star, :]  # (b, w)

        # Light 1D smoothing suppresses isolated speckle spikes, but it broadens
        # the sharp phase-correlation peak and biases its apex. So it is used ONLY
        # for a robust *coarse* integer pick; the sub-pixel parabola is then fitted
        # on the RAW profile after re-finding the true maximum next to that pick.
        corr_x = F.avg_pool1d(raw_corr_x.unsqueeze(1), 5, stride=1, padding=2).squeeze(
            1
        )
        corr_x = corr_x.masked_fill(~in_window[None], -float("inf"))
        peak0 = corr_x.argmax(dim=1).clamp(1, w - 2)  # coarse robust pick (b,)

        # Re-find the true maximum on the RAW profile within +-R of the coarse pick.
        R = 2
        cand = (peak0[:, None] + torch.arange(-R, R + 1, device=device)[None, :]).clamp(
            1, w - 2
        )
        peak = cand.gather(
            1, raw_corr_x.gather(1, cand).argmax(dim=1, keepdim=True)
        ).squeeze(1)

        ym1 = raw_corr_x.gather(1, (peak - 1)[:, None]).squeeze(1)
        y0 = raw_corr_x.gather(1, peak[:, None]).squeeze(1)
        yp1 = raw_corr_x.gather(1, (peak + 1)[:, None]).squeeze(1)
        if subpixel:
            denom = ym1 - 2 * y0 + yp1
            offset = torch.where(
                torch.isfinite(denom) & (denom != 0),
                0.5 * (ym1 - yp1) / denom,
                torch.zeros_like(y0),
            ).clamp(-1.0, 1.0)
            peak_sub = peak.float() + offset
        else:
            peak_sub = peak.float()

        # Confidence: prominence of the (smoothed) coarse peak over the window.
        win_vals = corr_x[:, in_window]
        y0_sm = corr_x.gather(1, peak0[:, None]).squeeze(1)
        conf[start:end] = (y0_sm - win_vals.mean(dim=1)) / (win_vals.std(dim=1) + 1e-8)

        dx[start:end] = (center - peak_sub) * (crop_w / w)
        if start == 0 and DEBUG:
            index = 10
            # Bokeh rather than plotly/ipympl: it embeds via BokehJS (no ipywidgets
            # model, so no "model not found" in VSCode), renders on canvas (fast),
            # and gives linked pan/zoom. Run once in the notebook if nothing shows:
            #     from bokeh.io import output_notebook; output_notebook()
            from bokeh.io import output_notebook, show
            from bokeh.layouts import gridplot
            from bokeh.models import LinearColorMapper, Span
            from bokeh.plotting import figure

            output_notebook(hide_banner=True)

            def _norm(t):
                """Contrast-stretch a frame to [0, 1] for display."""
                a = t.detach().cpu().numpy().astype(np.float32)
                lo, hi = np.percentile(a, [1, 99])
                return np.clip((a - lo) / (hi - lo + 1e-8), 0.0, 1.0)

            def _rgba(r_ch, g_ch, b_ch):
                """Three HxW [0,1] channels -> HxW uint32 RGBA for image_rgba.
                Row 0 is flipped to the top (Bokeh's y origin is at the bottom)."""
                a = np.dstack(
                    [
                        np.clip(r_ch, 0, 1) * 255,
                        np.clip(g_ch, 0, 1) * 255,
                        np.clip(b_ch, 0, 1) * 255,
                        np.full(r_ch.shape, 255),
                    ]
                ).astype(np.uint8)
                return np.flipud(a).copy().view(np.uint32).reshape(a.shape[:2])

            def _img_fig(title, rgba_img, link=None):
                kw = dict(
                    title=title,
                    match_aspect=True,
                    tools="pan,wheel_zoom,box_zoom,reset",
                    active_scroll="wheel_zoom",
                    height=300,
                    width=380,
                )
                if link is not None:  # share pan/zoom with the reference panel
                    kw["x_range"], kw["y_range"] = link.x_range, link.y_range
                p = figure(**kw)
                h, w = rgba_img.shape
                p.image_rgba(image=[rgba_img], x=0, y=0, dw=w, dh=h)
                p.toolbar.logo = None
                p.axis.visible = p.grid.visible = False
                return p

            # Reference as fed to the correlation (downsampled, mean-subtracted)
            # and the same reference after the band-pass, to show which image
            # structures the correlation actually uses. The `band` mask lives in
            # the frequency domain, so we apply it via FFT -> mask -> iFFT. The
            # Hann window is omitted here for legibility (it only apodises edges).
            r_ds = F.interpolate(ref[None, None].float(), size=(h, w), mode="area")[
                0, 0
            ]
            r_ms = r_ds - r_ds.mean()
            r_band = torch.fft.irfft2(torch.fft.rfft2(r_ms) * band[0], s=(h, w))
            g_ref, g_band = _norm(r_ms), _norm(r_band)

            p_ref = _img_fig("Reference", _rgba(g_ref, g_ref, g_ref))
            p_ref_filt = _img_fig(
                "Reference filtered (band-pass)",
                _rgba(g_band, g_band, g_band),
                link=p_ref,
            )

            corr_map = corr[index].cpu().numpy()
            p_corr = figure(
                title="Cross-correlation map",
                match_aspect=True,
                tools="pan,wheel_zoom,box_zoom,reset",
                active_scroll="wheel_zoom",
                height=300,
                width=380,
            )
            p_corr.image(
                image=[np.flipud(corr_map)],
                x=0,
                y=0,
                dw=corr_map.shape[1],
                dh=corr_map.shape[0],
                color_mapper=LinearColorMapper(
                    palette="Magma256",
                    low=float(corr_map.min()),
                    high=float(corr_map.max()),
                ),
            )
            p_corr.toolbar.logo = None
            p_corr.axis.visible = p_corr.grid.visible = False

            # Profile panel: raw vs smoothed profile and the parabola actually
            # fitted (on the raw samples). Only the ±max_shift window is meaningful.
            lo_x = center - max_shift_down - 2
            hi_x = center + max_shift_down + 2
            sl = slice(max(0, lo_x), min(w, hi_x + 1))
            xs_w = np.arange(w)[sl]
            raw_w = raw_corr_x[index].cpu().numpy()[sl]
            filt_w = corr_x[index].cpu().numpy()[sl]
            filt_w = np.where(np.isfinite(filt_w), filt_w, np.nan)  # -inf -> gap

            # Parabola through the 3 raw samples at the chosen peak (vertex = dx).
            pk_i = int(peak[index].item())
            ym1_i = float(raw_corr_x[index, pk_i - 1])
            y0_i = float(raw_corr_x[index, pk_i])
            yp1_i = float(raw_corr_x[index, pk_i + 1])
            qa = 0.5 * (ym1_i - 2 * y0_i + yp1_i)
            qb = 0.5 * (yp1_i - ym1_i)
            xx = np.linspace(pk_i - 2.0, pk_i + 2.0, 80)
            yy = qa * (xx - pk_i) ** 2 + qb * (xx - pk_i) + y0_i

            p_prof = figure(
                title="Correlation profile (summed over y)",
                x_range=(lo_x, hi_x),
                tools="pan,wheel_zoom,box_zoom,reset",
                active_scroll="wheel_zoom",
                height=300,
                width=380,
            )
            p_prof.line(xs_w, raw_w, line_color="royalblue", legend_label="raw")
            p_prof.line(
                xs_w,
                filt_w,
                line_color="orange",
                line_dash="dashed",
                legend_label="filtered (smoothed)",
            )
            p_prof.line(
                xx, yy, line_color="green", line_width=2, legend_label="parabola fit"
            )
            p_prof.add_layout(
                Span(
                    location=center,
                    dimension="height",
                    line_color="black",
                    line_dash="dotted",
                )
            )
            p_prof.add_layout(
                Span(
                    location=float(peak_sub[index].item()),
                    dimension="height",
                    line_color="red",
                    line_dash="dashed",
                )
            )
            p_prof.legend.location = "top_left"
            p_prof.legend.label_text_font_size = "8pt"
            p_prof.toolbar.logo = None

            show(
                gridplot(
                    [[p_ref, p_ref_filt], [p_corr, p_prof]],
                    toolbar_location="right",
                )
            )

    # Pixel-precise request: snap to whole ORIGINAL-resolution pixels. The peak is
    # already integer on the downsampled grid, but the back-projection
    # (× crop_w / w) turns it into a fraction; rounding here makes the returned
    # shift an integer number of pixels in the original image (not the
    # downsampled one). With ``subpixel`` the fractional peak offset is kept.
    if not subpixel:
        dx = dx.round()

    if return_confidence:
        return dx, conf

    return dx
