"""SiNC losses (Speth et al., "Non-Contrastive Unsupervised Learning of
Physiological Signals from Video", CVPR 2023), on irregular samples.

The paper takes an FFT of a uniformly sampled clip. Here the spectrum is a
direct DFT at each sample's own timestamps, restricted to the valid samples, so
dropped frames and speed augmentation need no resampling of the output: gaps
are simply absent from the sum, and a sped-up clip just carries shorter times.

All three losses read the same power spectrum on a frequency grid shared by
the batch (the variance loss averages spectra across samples):

* bandwidth — power outside ``[f_lo, f_hi]`` over total power;
* sparsity  — in-band power outside ``peak ± delta`` over in-band power;
* variance  — squared 1-D Wasserstein distance between the batch-averaged
  in-band spectrum and a prior over rates. This is what stops every clip from
  collapsing onto one frequency, i.e. what replaces an expected BPM.

The paper's prior is uniform over the band. On this cohort (HR 52–90 bpm, 5th–
95th percentile) a uniform 30–180 bpm prior wants half the power above 105 bpm,
and the network learns to supply it by locking clips onto the 2nd harmonic:
across epochs the variance loss and the harmonic rate correlated at −0.84.
:meth:`SiNCLoss.set_prior` replaces it with the population's rate distribution
(see :func:`rate_prior`); only the aggregate distribution is used, never a
clip's own rate.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SiNCLossConfig:
    f_lo: float = 0.5  # Hz (30 bpm)
    f_hi: float = 3.0  # Hz (180 bpm)
    # Half-width of the peak window in the sparsity loss (paper: 6 bpm).
    delta_hz: float = 0.1
    # Grid spacing and extent. f_max should sit just below the lowest Nyquist
    # frequency in the cohort (frame clocks run 82–89 Hz): higher, and aliased
    # bins double-count; much lower, and power above it goes unpenalised.
    df: float = 0.025
    f_max: float = 40.0
    w_bandwidth: float = 1.0
    w_sparsity: float = 1.0
    w_variance: float = 1.0


def masked_power_spectrum(
    x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Normalised power ``(B, F)`` of ``x`` at ``freqs``, over valid samples only.

    ``x``, ``t``, ``mask`` are ``(B, T)``. The mean over valid samples is removed
    first: the gap pattern leaks roughly 10 % of any DC offset into the cardiac
    band. Each spectrum is normalised to unit sum.
    """
    x = x.float()
    t = t.float()
    w = mask.float()
    n = w.sum(-1, keepdim=True).clamp_min(1.0)
    x = (x - (x * w).sum(-1, keepdim=True) / n) * w

    phase = 2 * torch.pi * t[..., None] * freqs  # (B, T, F)
    re = torch.einsum("bt,btf->bf", x, torch.cos(phase))
    im = torch.einsum("bt,btf->bf", x, torch.sin(phase))
    power = re.square() + im.square()
    return power / power.sum(-1, keepdim=True).clamp_min(1e-12)


class SiNCLoss(torch.nn.Module):
    def __init__(self, config: SiNCLossConfig | None = None):
        super().__init__()
        self.config = cfg = config or SiNCLossConfig()
        freqs = torch.arange(0.0, cfg.f_max + cfg.df / 2, cfg.df)
        self.register_buffer("freqs", freqs, persistent=False)
        self.register_buffer(
            "in_band", (freqs >= cfg.f_lo) & (freqs <= cfg.f_hi), persistent=False
        )
        # Persistent, so a checkpoint carries the prior it was trained with.
        n_band = int(self.in_band.sum())
        self.register_buffer("prior", torch.full((n_band,), 1.0 / n_band))

    @property
    def band_freqs(self) -> torch.Tensor:
        return self.freqs[self.in_band]

    def set_prior(self, prior: np.ndarray | torch.Tensor) -> None:
        """Prior over the in-band bins (normalised here)."""
        prior = torch.as_tensor(prior, dtype=torch.float32, device=self.prior.device)
        if prior.shape != self.prior.shape:
            raise ValueError(
                f"prior has {prior.numel()} bins, band has {self.prior.numel()}"
            )
        self.prior.copy_(prior / prior.sum())

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        with torch.autocast(device_type=x.device.type, enabled=False):
            psd = masked_power_spectrum(x, t, mask, self.freqs)

            # DC is removed by construction; keep it out of the denominator too.
            psd_nodc = psd[:, 1:]
            in_band = self.in_band[1:]
            bandwidth = psd_nodc[:, ~in_band].sum(-1) / psd_nodc.sum(-1).clamp_min(
                1e-12
            )

            band = psd[:, self.in_band]
            band = band / band.sum(-1, keepdim=True).clamp_min(1e-12)
            f_band = self.freqs[self.in_band]
            peak = f_band[band.argmax(-1)]
            near_peak = (f_band[None] - peak[:, None]).abs() <= cfg.delta_hz
            sparsity = (band * ~near_peak).sum(-1)

            mean_band = band.mean(0)
            variance = (mean_band.cumsum(0) - self.prior.cumsum(0)).square().mean()

            bandwidth, sparsity = bandwidth.mean(), sparsity.mean()
            loss = (
                cfg.w_bandwidth * bandwidth
                + cfg.w_sparsity * sparsity
                + cfg.w_variance * variance
            )
        return dict(
            loss=loss,
            bandwidth=bandwidth,
            sparsity=sparsity,
            variance=variance,
            peak_hz=peak.detach(),
        )


def rate_prior(
    band_freqs: np.ndarray,
    rates_bpm: np.ndarray,
    speed_range: tuple[float, float] = (1.0, 1.0),
    smooth_hz: float = 0.05,
    floor: float = 0.02,
    n_samples: int = 200_000,
    seed: int = 0,
) -> np.ndarray:
    """Density of clip rates over ``band_freqs``: population rates × speed.

    Speed augmentation multiplies each clip's rate by a uniform draw from
    ``speed_range``, so the prior is the rate histogram convolved with that.
    Smoothed with a Gaussian of ``smooth_hz``, then mixed with a uniform floor
    (``floor`` of the mass) so no in-band rate is ruled out entirely.
    """
    rng = np.random.default_rng(seed)
    rates = np.asarray(rates_bpm, float)
    rates = rates[np.isfinite(rates) & (rates > 0)]
    f = rng.choice(rates, n_samples) / 60.0 * rng.uniform(*speed_range, n_samples)
    df = float(band_freqs[1] - band_freqs[0])
    edges = np.r_[band_freqs - df / 2, band_freqs[-1] + df / 2]
    hist = np.histogram(f, edges)[0].astype(float)
    k = np.arange(-4 * smooth_hz, 4 * smooth_hz + df / 2, df)
    kernel = np.exp(-0.5 * (k / smooth_hz) ** 2)
    hist = np.convolve(hist, kernel / kernel.sum(), mode="same")
    hist /= hist.sum()
    return (1 - floor) * hist + floor / len(hist)


@torch.no_grad()
def peak_frequency(
    x: torch.Tensor,
    t: torch.Tensor,
    mask: torch.Tensor,
    f_lo: float = 0.5,
    f_hi: float = 3.0,
    df: float = 0.002,
    chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """In-band peak of a long waveform, for evaluation: ``(peak_hz, freqs, psd)``.

    Frequencies are processed in chunks, since a whole video times a fine grid
    would not fit as one ``(B, T, F)`` phase tensor.
    """
    freqs = torch.arange(f_lo, f_hi + df / 2, df, device=x.device)
    x, t, w = x.float(), t.float(), mask.float()
    n = w.sum(-1, keepdim=True).clamp_min(1.0)
    x = (x - (x * w).sum(-1, keepdim=True) / n) * w
    parts = []
    for f in freqs.split(chunk):
        phase = 2 * torch.pi * t[..., None] * f
        re = torch.einsum("bt,btf->bf", x, torch.cos(phase))
        im = torch.einsum("bt,btf->bf", x, torch.sin(phase))
        parts.append(re.square() + im.square())
    psd = torch.cat(parts, -1)
    return freqs[psd.argmax(-1)], freqs, psd
