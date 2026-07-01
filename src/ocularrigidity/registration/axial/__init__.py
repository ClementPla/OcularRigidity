"""Recalage axial (2e passe) : alignement de chaque A-scan sur la mediane du volume.

Reproduit l'identification de la RPE du projet MATLAB (compensation d'ombres +
Laplacien-d'une-Gaussienne + recalage par A-scan sur une reference mediane).
"""

from ocularrigidity.registration.axial.shadow import correct_shadow
from ocularrigidity.registration.axial.log_filter import (
    fspecial_log,
    laplacian_of_gaussian,
)
from ocularrigidity.registration.axial.median_registration import (
    estimate_ascan_vshift_to_median,
    register_ascans_to_median,
)

__all__ = [
    "correct_shadow",
    "fspecial_log",
    "laplacian_of_gaussian",
    "estimate_ascan_vshift_to_median",
    "register_ascans_to_median",
]
