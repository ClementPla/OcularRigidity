"""
Two-group choroidal pulsation: artery-vein phase shift demo.
============================================================

Linearity trick
---------------
Eigenstrain RHS is linear in alpha:
    f_total(t) = alpha_A(t) * f_A^unit + alpha_B(t) * f_B^unit
By linearity of K u = f:
    u(t)       = alpha_A(t) * u_A^unit + alpha_B(t) * u_B^unit

where u_A^unit, u_B^unit are the static FEM solutions with each group at
alpha = 1 inside the disks. Two FEM solves total. After that, sweeping
delays / amplitudes / waveform shapes is pure arithmetic, basically free.
"""

import numpy as np
import matplotlib.pyplot as plt
from skfem import (
    MeshTri,
    Basis,
    ElementVector,
    ElementTriP1,
    BilinearForm,
    LinearForm,
    asm,
    condense,
    solve,
)
from skfem.helpers import sym_grad, trace, ddot


# ----------------------------------------------------------------------
# Geometry & material  (same as before)
# ----------------------------------------------------------------------
WIDTH, HEIGHT = 3000.0, 300.0
E, NU = 5.0e3, 0.45
mu = E / (2.0 * (1.0 + NU))
lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))

# Each vessel now carries a 'group' tag in addition to (cx, cy, r).
# I placed both groups at comparable depths and sizes so neither group
# dominates by leverage alone -- the phase-shift effect is then the
# leading source of waveform shape change.
VESSELS = [
    # (cx,     cy,    r,    group)
    (1000.0, 200.0, 80.0, "arterial"),
    (2200.0, 220.0, 90.0, "arterial"),
    (600.0, 180.0, 70.0, "venous"),
    (1700.0, 190.0, 75.0, "venous"),
    (2700.0, 170.0, 65.0, "venous"),
]

ALPHA_PEAK_ART = 0.08  # arterial radial pulsation amplitude
ALPHA_PEAK_VEN = 0.05  # venous; smaller because capacitive smoothing
N_FRAMES = 60


# ----------------------------------------------------------------------
# Mesh, basis, Dirichlet on BM, stiffness  (unchanged)
# ----------------------------------------------------------------------
mesh = MeshTri.init_tensor(np.linspace(0.0, WIDTH, 241), np.linspace(0.0, HEIGHT, 33))
basis = Basis(mesh, ElementVector(ElementTriP1()))
D = basis.get_dofs(lambda x: x[1] < 1e-9)


@BilinearForm
def stiffness(u, v, w):
    return 2.0 * mu * ddot(sym_grad(u), sym_grad(v)) + lam * trace(sym_grad(u)) * trace(
        sym_grad(v)
    )


K = asm(stiffness, basis)


# ----------------------------------------------------------------------
# Per-group unit-amplitude RHS assembly
# ----------------------------------------------------------------------
# Closure pattern: assemble_unit_rhs('arterial') returns the load vector
# you'd get if every arterial vessel had alpha = 1 and every venous vessel
# had alpha = 0. The time variation is reintroduced later by simple scalar
# multiplication -- no need to reassemble at every frame.
def assemble_unit_rhs(group):
    @LinearForm
    def rhs(v, w):
        x, y = w.x[0], w.x[1]
        a = np.zeros_like(x)
        for cx, cy, r, g in VESSELS:
            if g == group:
                a += (x - cx) ** 2 + (y - cy) ** 2 <= r**2
        return 2.0 * (lam + mu) * a * trace(sym_grad(v))

    return asm(rhs, basis)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# THE TWO FEM SOLVES.  Everything after this is bookkeeping + plotting.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
u_unit_art = solve(*condense(K, assemble_unit_rhs("arterial"), D=D))
u_unit_ven = solve(*condense(K, assemble_unit_rhs("venous"), D=D))


# ----------------------------------------------------------------------
# CSI bookkeeping: pull the unit u_y response at top nodes once
# ----------------------------------------------------------------------
top_nodes = np.where(np.isclose(mesh.p[1], HEIGHT))[0]
top_nodes = top_nodes[np.argsort(mesh.p[0, top_nodes])]
top_x = mesh.p[0, top_nodes]
mid_mask = (top_x > 0.2 * WIDTH) & (top_x < 0.8 * WIDTH)

uy_art_unit_csi = u_unit_art.reshape(-1, 2)[top_nodes, 1]
uy_ven_unit_csi = u_unit_ven.reshape(-1, 2)[top_nodes, 1]


# ----------------------------------------------------------------------
# Cardiac waveforms (raised cosines; trivially swap in something sharper)
# ----------------------------------------------------------------------
phases = np.linspace(0.0, 2.0 * np.pi, N_FRAMES, endpoint=False)


def alpha_arterial(phi):
    return 0.5 * ALPHA_PEAK_ART * (1.0 - np.cos(phi))


def alpha_venous(phi, delay_cycles):
    # venous(phi) = arterial-like waveform evaluated at phi - 2*pi*delay,
    # i.e. delayed in time by `delay_cycles` cardiac cycles.
    return 0.5 * ALPHA_PEAK_VEN * (1.0 - np.cos(phi - 2.0 * np.pi * delay_cycles))


def csi_waveform(delay_cycles):
    """
    The whole point of the linearity trick lives here:

        u_csi(t) = alpha_A(t) * uy_A^unit + alpha_V(t) * uy_V^unit

    No FEM solve inside this function. Just two coefficients per frame
    multiplying two precomputed spatial responses. We then average over
    the central 60 % of the strip to mimic measuring well away from the
    free-edge artefacts.
    """
    a_a = alpha_arterial(phases)  # (N_FRAMES,)
    a_v = alpha_venous(phases, delay_cycles)  # (N_FRAMES,)
    # outer-product superposition over time x nodes
    csi_t = (
        a_a[:, None] * uy_art_unit_csi[None, :]
        + a_v[:, None] * uy_ven_unit_csi[None, :]
    )
    return csi_t[:, mid_mask].mean(axis=1)


# ----------------------------------------------------------------------
# Sweep delays. Each call is a few multiplies + a mean -- millisecond cheap.
# ----------------------------------------------------------------------
delays = [0.0, 0.10, 0.25, 0.50]
waveforms = {d: csi_waveform(d) for d in delays}


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)

# Panel 1: geometry, color by group
ax = axes[0]
ax.set_aspect("equal")
ax.add_patch(plt.Rectangle((0, 0), WIDTH, HEIGHT, fill=False, ec="black"))
gcol = {"arterial": "#e74c3c", "venous": "#3498db"}
for cx, cy, r, g in VESSELS:
    ax.add_patch(
        plt.Circle((cx, cy), r, fc=gcol[g], ec="black", lw=0.5, alpha=0.65, label=g)
    )
ax.text(WIDTH / 2, -22, "BM (fixed)", ha="center")
ax.text(WIDTH / 2, HEIGHT + 12, "CSI (free, observed)", ha="center")
ax.set_xlim(-50, WIDTH + 50)
ax.set_ylim(-45, HEIGHT + 35)
ax.set_xlabel("x [µm]")
ax.set_ylabel("y [µm]")
ax.set_title("Two-group geometry (red = arterial, blue = venous)")

# Panel 2: input vessel waveforms for one chosen delay
ax = axes[1]
ax.plot(
    phases / (2 * np.pi),
    alpha_arterial(phases),
    color="#e74c3c",
    lw=2,
    label="alpha_arterial(t)",
)
ax.plot(
    phases / (2 * np.pi),
    alpha_venous(phases, 0.25),
    color="#3498db",
    lw=2,
    label="alpha_venous(t), delay = 0.25 cyc",
)
ax.set_xlabel("cardiac phase [cycles]")
ax.set_ylabel("isotropic vessel strain  alpha")
ax.set_title("Input vessel pulsation waveforms (one delay shown)")
ax.legend()
ax.grid(alpha=0.3)

# Panel 3: CSI response across delays  -- the money plot
ax = axes[2]
for d, wfm in waveforms.items():
    ax.plot(phases / (2 * np.pi), wfm, lw=2, label=f"delay = {d:.2f} cyc")
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("cardiac phase [cycles]")
ax.set_ylabel("mean CSI u_y [µm]")
ax.set_title("Predicted CSI waveform vs venous phase delay")
ax.legend()
ax.grid(alpha=0.3)

plt.savefig("two_group_output.png", dpi=130)

plt.show()


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
print("-" * 64)
for d, wfm in waveforms.items():
    ptp = wfm.max() - wfm.min()
    pk = phases[np.argmax(wfm)] / (2 * np.pi)
    print(f"delay = {d:.2f} cyc  |  PTP = {ptp:5.2f} µm  |  peak phase = {pk:.2f}")
print("-" * 64)
print("delay 0.00 : groups in phase, single broad maximum, largest PTP.")
print("delay 0.25 : peak shifts later, waveform broadens/skews.")
print("delay 0.50 : groups oppose, partial cancellation -> smaller PTP,")
print("             possibly biphasic if amplitudes were closer.")
