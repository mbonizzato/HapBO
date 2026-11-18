"""Valid GP prior-mean (residual fit in standardized z-space)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bayesian_optimization import GaussianProcessSurrogate  # noqa: E402


def _data(seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(10, 2))
    y = 2.0 * np.sin(4.0 * X[:, 0]) + X[:, 1] + 5.0
    return X, y


def test_prior_mean_requires_normalize_y() -> None:
    try:
        GaussianProcessSurrogate(prior_mean=lambda X: np.zeros(len(X)))
    except ValueError as exc:
        assert "normalize_y" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("prior_mean without normalize_y must raise")


def test_flat_prior_mean_matches_zero_mean_gp() -> None:
    X, y = _data()
    Xtest = np.random.default_rng(2).uniform(0.0, 1.0, size=(6, 2))
    plain = GaussianProcessSurrogate(length_scale=0.4, normalize_y=True).fit(X, y)
    flat = GaussianProcessSurrogate(
        length_scale=0.4, normalize_y=True, prior_mean=lambda Z: np.zeros(len(Z))
    ).fit(X, y)
    np.testing.assert_allclose(plain.predict(Xtest)[0], flat.predict(Xtest)[0], atol=1e-12)
    np.testing.assert_allclose(plain.predict(Xtest)[1], flat.predict(Xtest)[1], atol=1e-12)


def test_far_field_reverts_to_prior_mean() -> None:
    X, y = _data()
    # Constant m_z = 1.5 in z-space; far from data the residual GP -> 0 so the
    # posterior mean -> y_mean + y_scale * m_z.
    gp = GaussianProcessSurrogate(
        length_scale=0.15, normalize_y=True, prior_mean=lambda Z: 1.5 * np.ones(len(Z))
    ).fit(X, y)
    far = np.array([[50.0, 50.0]])
    mean, _ = gp.predict(far)
    expected = gp.y_mean_ + gp.y_scale_ * 1.5
    np.testing.assert_allclose(mean.item(), expected, rtol=1e-6)


def test_noiseless_interpolates_but_noisy_smooths() -> None:
    X, y = _data()
    m_z = lambda Z: 0.3 * Z[:, 0]  # noqa: E731 (nontrivial prior mean)
    noiseless = GaussianProcessSurrogate(
        length_scale=0.4, normalize_y=True, noise_variance=0.0, prior_mean=m_z
    ).fit(X, y)
    mean_nf, _ = noiseless.predict(X)
    np.testing.assert_allclose(mean_nf, y, atol=1e-4)  # exact interpolation at data

    noisy = GaussianProcessSurrogate(
        length_scale=0.4, normalize_y=True, noise_variance=0.2, prior_mean=m_z
    ).fit(X, y)
    mean_noisy, std_noisy = noisy.predict(X)
    assert not np.allclose(mean_noisy, y, atol=1e-4)  # smoothing, not interpolation
    assert np.all(std_noisy > 0.0)


def test_prior_mean_is_invariant_under_affine_transform_of_y() -> None:
    X, y = _data()
    Xtest = np.random.default_rng(3).uniform(0.0, 1.0, size=(7, 2))
    m_z = lambda Z: 0.8 * Z[:, 0] - 0.2 * Z[:, 1]  # noqa: E731
    a, b = 3.5, -4.0
    gp1 = GaussianProcessSurrogate(length_scale=0.5, normalize_y=True, prior_mean=m_z).fit(X, y)
    gp2 = GaussianProcessSurrogate(length_scale=0.5, normalize_y=True, prior_mean=m_z).fit(
        X, a * y + b
    )
    m1, s1 = gp1.predict(Xtest)
    m2, s2 = gp2.predict(Xtest)
    # Predictions transform affinely; std scales by |a|; selection is unchanged.
    np.testing.assert_allclose(m2, a * m1 + b, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(s2, abs(a) * s1, rtol=1e-8, atol=1e-8)
    assert int(np.argmax(m1)) == int(np.argmax(m2))
