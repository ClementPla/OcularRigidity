from typing import Literal

import numpy as np
from sklearn.decomposition import PCA, FastICA, MiniBatchSparsePCA, SparsePCA


def project_into_separable_components(
    signal,
    method: Literal["pca", "ica", "svd"],
    n_components: int | None = None,
    random_state: int = 1234,
    max_iter: int = 1000,
    whiten: Literal["unit-variance", "arbitrary-variance"] | bool = "unit-variance",
    tol: float = 1e-4,
    fun="cube",
):
    """
    Projects a 2D signal (TxW) into separable components using PCA, ICA or SVD.

    Time is the sample axis and the W spatial points are the features, so the
    decomposition reads ``signal[t, w] ~= sum_k projected_signal[t, k] * mixings[w, k]``
    (up to the per-point temporal mean, which "pca" and "ica" remove internally).

    Args:
        signal (np.ndarray): TxW array of values to project.
        method (Literal["pca", "ica", "svd"]): Method to use for projection.
            "svd" is the *uncentered* decomposition: unlike "pca" and "ica" it keeps
            each point's temporal mean, so the leading component typically captures the
            static baseline shape rather than a dynamic mode. Run on centered input it
            is equivalent to "pca".
        n_components (Optional[int]): Number of components to keep. If None, all components are kept.

    Returns:
        projected_signal (np.ndarray): T x n_components array of temporal sources.
        mixings (np.ndarray): W x n_components spatial pattern; column k weights how
            much each of the W points expresses component k. Note the methods do not
            scale this the same way: "pca" and "svd" columns are unit-norm (amplitude
            stays in the scores), whereas "ica" with ``whiten="unit-variance"`` yields
            unit-variance sources, so the mixing columns carry the amplitude in the
            original signal units.
    """
    if method == "pca":
        if isinstance(whiten, str):
            whiten = whiten == "unit-variance"
        model = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
        projected_signal = model.fit_transform(signal)
        mixings = model.components_.T
    elif method == "sparse-pca":
        model = SparsePCA(
            n_components=n_components,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
        )
        projected_signal = model.fit_transform(signal)
        mixings = model.components_.T

    elif method == "ica":
        model = FastICA(
            n_components=n_components,
            random_state=random_state,
            max_iter=max_iter,
            whiten=whiten,
            fun=fun,
            tol=tol,
        )
        projected_signal = model.fit_transform(signal)
        mixings = model.mixing_
    elif method == "svd":
        u, s, vt = np.linalg.svd(np.asarray(signal, dtype=float), full_matrices=False)
        k = s.size if n_components is None else min(n_components, s.size)
        projected_signal = u[:, :k] * s[:k]
        mixings = vt[:k].T

    else:
        raise ValueError(f"Unknown method: {method}")

    return projected_signal, mixings
