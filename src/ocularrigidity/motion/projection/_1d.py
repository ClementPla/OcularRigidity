from typing import Literal, Optional
from sklearn.decomposition import PCA, FastICA


def project_into_separable_components(
    signal,
    method: Literal["pca", "ica"],
    n_components: Optional[int] = None,
    normalize: bool = True,
    random_state: int = 1234,
    max_iter: int = 1000,
    whiten: bool = "unit-variance",
    tol: float = 1e-4,
    fun="cube",
):
    """
    Projects a 2D signal (TxW) into separable components using PCA or ICA.

    Args:
        signal (np.ndarray): TxW array of values to project.
        method (Literal["pca", "ica"]): Method to use for projection. Either "pca" or "ica".
        n_components (Optional[int]): Number of components to keep. If None, all components are kept.

    Returns:
        projected_signal (np.ndarray): TxW array of projected values.
        components (np.ndarray): n_components x W array of component vectors.
    """
    if normalize:
        signal = signal - signal.mean(axis=0, keepdims=True)

    if method == "pca":
        model = PCA(n_components=n_components)
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

    else:
        raise ValueError(f"Unknown method: {method}")

    return projected_signal, mixings
