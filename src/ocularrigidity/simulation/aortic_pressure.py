import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import welch


fs = 1000.0
T = 300.0  # 5 minutes

t = np.arange(0, T, 1 / fs)

mean_HR = 60.0 / 60  # 60 bpm

# RSA:
RSA_freq = 15 / 60  # 15 bpm
RSA_amplitude = 0.1 * mean_HR  # 10% variation in heart rate due to RSA

# Mayer reflex
Mayer_freq = 6 / 60  # 6 bpm
Mayer_amplitude = 0.05 * mean_HR  # 5% variation in heart rate due to Mayer reflex


modulation = (
    mean_HR
    + RSA_amplitude * np.sin(2 * np.pi * RSA_freq * t)
    + Mayer_amplitude * np.sin(2 * np.pi * Mayer_freq * t)
)


phi = np.cumsum(modulation) / fs  # Integral of the modulation to get the phase

beat_idx = (
    np.where(np.diff(np.floor(phi)) > 0)[0] + 1
)  # Indices of the beats, where the phase crosses an integer
beat_times = t[beat_idx]


plt.plot(t, modulation)
plt.plot(t, phi % 1, alpha=0.5, label="Phase (mod 1)")

# Mark beat times
for tb in beat_times:
    plt.axvline(tb, color="k", ls=":", alpha=0.3)

plt.xlabel("t (s)")
plt.ylabel("HR (Hz)")
plt.title("Heart rate modulation")
plt.xlim(0, 15)
plt.savefig("heart_rate_modulation.png", dpi=300)
plt.show()
RR = np.diff(beat_times)


T_sys, Q_peak = 0.3, 350.0  # SV ~ 75 mL with this shape
nsys = int(round(T_sys * fs)) + 1
x = np.arange(nsys) / (fs * T_sys)
pulse = Q_peak * x * np.exp(1.0 - x)  # Poisson-like, peaks at x=1


n_taper = max(1, int(0.5 * nsys))
pulse[-n_taper:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(n_taper) / n_taper))
plt.plot(np.arange(nsys) / fs, pulse)

plt.xlabel("t (s)")
plt.ylabel("Q (mL/s)")
plt.title("Unit pulse shape (Poisson-like)")
plt.savefig("unit_pulse_shape.png", dpi=300)
plt.show()

Q = np.zeros_like(t)
for tb in beat_times:
    i0 = int(round(tb * fs))
    i1 = min(i0 + nsys, len(t))
    Q[i0:i1] = pulse[: i1 - i0]


plt.plot(t, Q)
# Mark beat times
for tb in beat_times:
    plt.axvline(tb, color="k", ls=":", alpha=0.3)
plt.xlabel("t (s)")
plt.ylabel("Q (mL/s)")
plt.title("Inflow Q(t) — Poisson-shaped pulses, IPFM-timed")
plt.xlim(0, 15)
plt.savefig("inflow_Q_t.png", dpi=300)
plt.show()

R, Rd, C, L = 0.05, 1.00, 1.50, 0.005  # mmHg*s/mL ; mmHg*s/mL ; mL/mmHg ; mmHg*s^2/mL


def rhs(ti, y):
    return [(Q_of_t(ti) - y[0] / Rd) / C]


Q_of_t = interp1d(
    t, Q, kind="linear", bounds_error=False, fill_value=0.0, assume_sorted=True
)
print("Simulating aortic pressure response to pulsatile flow...")
sol = solve_ivp(
    rhs,
    (t[0], t[-1]),
    [80.0],
    t_eval=t,
    method="LSODA",
    rtol=1e-6,
    atol=1e-6,
    max_step=5e-3,
)
print("Simulation complete.")

P_wk = sol.y[0]

dQdt = np.gradient(Q, 1 / fs)
P = L * dQdt + R * Q + P_wk
i0 = int(20 * fs)
tt, PP, QQ = t[i0:], P[i0:], Q[i0:]

ed_idx = (beat_times[beat_times > 20.0] * fs).astype(int) - 1
EDP = P[ed_idx]
# True systolic peak per beat
SBP_per_beat = []
for k in range(len(beat_times) - 1):
    tb, tn = beat_times[k], beat_times[k + 1]
    if tb < 20.0:
        continue
    seg = P[int(tb * fs) : int(tn * fs)]
    if len(seg):
        SBP_per_beat.append(seg.max())
SBP_per_beat = np.array(SBP_per_beat)

nperseg = int(60 * fs)
f_psd, Sxx = welch(
    PP - PP.mean(), fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant"
)

fig, ax = plt.subplots(4, 1, figsize=(10, 12))

w = slice(0, int(15 * fs))


ax[0].plot(tt[w] - tt[w][0], QQ[w], lw=0.9, color="tab:orange")
ax[0].set(
    xlabel="t (s)",
    ylabel="Q (mL/s)",
    title="Inflow Q(t) — Poisson-shaped pulses, IPFM-timed",
)

ax[1].plot(tt[w] - tt[w][0], PP[w], lw=0.9)
ax[1].set(xlabel="t (s)", ylabel="P (mmHg)", title="Aortic pressure (15 s window)")

ax[2].semilogy(f_psd, Sxx, lw=0.8)
ax[2].set(
    xlim=(0, 6),
    xlabel="f (Hz)",
    ylabel="PSD",
    title="P(t) spectrum: f0, harmonics, HRV sidebands",
)
for k in range(1, 6):
    ax[2].axvline(k * mean_HR, color="k", ls=":", alpha=0.4)
# Mark Mayer and RSA frequencies
ax[2].axvline(
    Mayer_freq, color="C0", ls="--", alpha=0.5, label=f"Mayer {Mayer_freq} Hz"
)
ax[2].axvline(RSA_freq, color="C3", ls="--", alpha=0.5, label=f"RSA {RSA_freq} Hz")
for s in (-1, 1):
    ax[2].axvline(mean_HR + s * Mayer_freq, color="C0", ls="--", alpha=0.5)
    ax[2].axvline(mean_HR + s * RSA_freq, color="C3", ls="--", alpha=0.5)
ax[2].legend()
ax[3].semilogy(f_psd, Sxx, lw=0.8)
ax[3].set(
    xlim=(0, 0.5),
    xlabel="f (Hz)",
    ylabel="PSD",
    title="LF / HF band — raw HRV in P(t) envelope",
)
ax[3].axvline(Mayer_freq, color="C0", ls="--", alpha=0.6, label=f"LF {Mayer_freq} Hz")
ax[3].axvline(RSA_freq, color="C3", ls="--", alpha=0.6, label=f"HF {RSA_freq} Hz")
ax[3].legend()

plt.tight_layout()
plt.savefig("aortic_pressure_simulation.png", dpi=300)

plt.show()
