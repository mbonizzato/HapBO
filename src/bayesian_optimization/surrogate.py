"""A small Gaussian-process surrogate used by the BO loop."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def _matern52(
    X1: np.ndarray,
    X2: np.ndarray,
    length_scale: np.ndarray,
    signal_variance: float,
) -> np.ndarray:
    """Matérn 5/2 kernel ``s2 * (1 + sqrt(5) r + 5 r^2 / 3) * exp(-sqrt(5) r)``."""

    scaled_difference = (X1[:, None, :] - X2[None, :, :]) / length_scale[None, None, :]
    distance = np.sqrt(np.sum(scaled_difference**2, axis=2))
    sqrt_five_distance = np.sqrt(5.0) * distance
    return signal_variance * (
        1.0 + sqrt_five_distance + (5.0 / 3.0) * distance**2
    ) * np.exp(-sqrt_five_distance)


def _scaled_distance(X1: np.ndarray, X2: np.ndarray, length_scale: np.ndarray) -> np.ndarray:
    """Lengthscale-scaled Euclidean distance r, the same expression as in ``_matern52``."""

    scaled_difference = (X1[:, None, :] - X2[None, :, :]) / length_scale[None, None, :]
    return np.sqrt(np.sum(scaled_difference**2, axis=2))


def _rbf(
    X1: np.ndarray,
    X2: np.ndarray,
    length_scale: np.ndarray,
    signal_variance: float,
) -> np.ndarray:
    """Squared-exponential (RBF) kernel ``s2 * exp(-r^2 / 2)``."""

    distance = _scaled_distance(X1, X2, length_scale)
    return signal_variance * np.exp(-0.5 * distance**2)


def _matern12(
    X1: np.ndarray,
    X2: np.ndarray,
    length_scale: np.ndarray,
    signal_variance: float,
) -> np.ndarray:
    """Exponential (Matérn 1/2) kernel ``s2 * exp(-r)``."""

    distance = _scaled_distance(X1, X2, length_scale)
    return signal_variance * np.exp(-distance)


# GP covariance functions selectable by name.  Each is a function of the
# lengthscale-scaled distance r of ``_matern52`` and equals the signal variance
# at r = 0 (the prior variance ``predict`` uses).
DEFAULT_KERNEL = "matern52"
KERNELS = ("matern52", "rbf", "matern12")


def validate_kernel(name: str) -> str:
    """Return ``name`` if it is one of ``KERNELS``; raise ``ValueError`` otherwise."""

    if not isinstance(name, str) or name not in KERNELS:
        raise ValueError(f"unknown GP kernel {name!r}; expected one of {list(KERNELS)}")
    return name


def _covariance_function(name: str) -> Callable[..., np.ndarray]:
    """The covariance function of a kernel name."""

    if name == "matern52":
        return _matern52
    if name == "rbf":
        return _rbf
    if name == "matern12":
        return _matern12
    raise ValueError(f"unknown GP kernel {name!r}; expected one of {list(KERNELS)}")


class GaussianProcessSurrogate:
    """Gaussian process with a Matérn 5/2 (default), RBF or Matérn 1/2 kernel.

    The implementation is intentionally compact and depends only on NumPy.  It
    uses a Cholesky factorization rather than explicitly inverting the kernel
    matrix.

    Parameters
    ----------
    length_scale:
        Positive scalar lengthscale shared by every input dimension (isotropic
        kernel).
    prior_mean:
        Optional callable ``m_z(X) -> array`` returning a per-row GP prior mean
        in *standardized* target units (``z``-space).  When supplied, the GP is
        fit to the residual ``z_y - m_z(X_train)`` and predictions add ``m_z``
        back before de-standardizing -- i.e. a proper nonzero-mean GP, not a
        post-hoc offset.  Requires ``normalize_y=True`` so ``m_z`` and the
        residual share one ``z``-space.
    kernel:
        Covariance function: ``"matern52"`` (default), ``"rbf"`` (``s2 *
        exp(-r^2 / 2)``) or ``"matern12"`` (``s2 * exp(-r)``), with r the
        lengthscale-scaled Euclidean distance.  Unknown names raise.  The fit
        and the predictions both use it.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 1e-6,
        normalize_y: bool = False,
        prior_mean: Callable[[np.ndarray], np.ndarray | float] | None = None,
        kernel: str = DEFAULT_KERNEL,
    ) -> None:
        self.kernel_name = validate_kernel(kernel)
        if np.ndim(length_scale) != 0:
            raise ValueError("length_scale must be a positive scalar")
        length_scale = float(length_scale)
        if not (np.isfinite(length_scale) and length_scale > 0.0):
            raise ValueError("length_scale must be positive and finite")
        if signal_variance <= 0:
            raise ValueError("signal_variance must be positive")
        if noise_variance < 0:
            raise ValueError("noise_variance must be non-negative")
        if prior_mean is not None and not normalize_y:
            raise ValueError("prior_mean requires normalize_y=True (m_z lives in z-space)")

        self.length_scale = length_scale
        self.signal_variance = float(signal_variance)
        self.noise_variance = float(noise_variance)
        self.normalize_y = bool(normalize_y)
        self.prior_mean = prior_mean
        self.is_fitted = False

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def _evaluate_prior_mean(
        self,
        callback: Callable[[np.ndarray], np.ndarray | float],
        X: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(callback(X.copy()), dtype=float)
        if values.ndim == 0:
            values = np.full(X.shape[0], float(values))
        values = values.reshape(-1)
        if values.size != X.shape[0] or not np.all(np.isfinite(values)):
            raise ValueError("prior_mean must return one finite value per row")
        return values

    def kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the covariance (``kernel_name``) between two sets of points."""
        X1 = self._as_2d(X1)
        X2 = self._as_2d(X2)
        if X1.shape[1] != X2.shape[1]:
            raise ValueError("X1 and X2 must have the same number of features")

        length_scale = np.full(X1.shape[1], self.length_scale)
        covariance = _covariance_function(self.kernel_name)
        return covariance(X1, X2, length_scale, self.signal_variance)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessSurrogate":
        """Fit the GP to observations."""
        X = self._as_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.size:
            raise ValueError("X and y must contain the same number of observations")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")

        self.X_train_ = X.copy()
        self.y_mean_ = float(np.mean(y)) if self.normalize_y else 0.0
        centered_y = y - self.y_mean_
        if self.normalize_y:
            raw_scale = float(np.std(centered_y))
            self.y_scale_ = raw_scale if raw_scale > 1e-12 else 1.0
            centered_y = centered_y / self.y_scale_
        else:
            self.y_scale_ = 1.0

        # Proper nonzero GP prior mean: fit the residual z_y - m_z(X_train).
        if self.prior_mean is not None:
            residual = centered_y - self._evaluate_prior_mean(self.prior_mean, X)
        else:
            residual = centered_y

        covariance = self.kernel(X, X)
        identity = np.eye(X.shape[0])
        # Increase jitter only if floating-point issues prevent factorization.
        jitter = 1e-12
        for _ in range(8):
            try:
                self.cholesky_ = np.linalg.cholesky(
                    covariance + (self.noise_variance + jitter) * identity
                )
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            raise np.linalg.LinAlgError("Unable to factorize the GP covariance matrix")

        self.alpha_ = np.linalg.solve(
            self.cholesky_.T,
            np.linalg.solve(self.cholesky_, residual),
        )
        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Predict posterior means and standard deviations."""
        if not self.is_fitted:
            raise RuntimeError("The surrogate must be fitted before prediction")
        X = self._as_2d(X)
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError("X has a different number of features than the training data")

        cross_covariance = self.kernel(X, self.X_train_)
        mean_z = cross_covariance @ self.alpha_
        # Add the GP prior mean back (in z-space) before de-standardizing.
        if self.prior_mean is not None:
            mean_z = mean_z + self._evaluate_prior_mean(self.prior_mean, X)
        mean = mean_z * self.y_scale_ + self.y_mean_

        if not return_std:
            return mean

        solved = np.linalg.solve(self.cholesky_, cross_covariance.T)
        prior_variance = np.full(X.shape[0], self.signal_variance)
        variance = np.maximum(prior_variance - np.sum(solved**2, axis=0), 0.0)
        std = np.sqrt(variance) * self.y_scale_
        return mean, std
