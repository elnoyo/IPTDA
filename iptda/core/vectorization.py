"""Representaciones vectoriales de diagramas de persistencia."""

from __future__ import annotations

import numpy as np
from scipy.special import erf


def finite_part(D: np.ndarray, cap: float | None = None) -> np.ndarray:
    """Puntos finitos; si ``cap`` no es ``None``, las muertes infinitas se
    sustituyen por ``cap`` en lugar de descartarse."""
    D = np.asarray(D, dtype=float).reshape(-1, 2)
    if cap is None:
        return D[np.isfinite(D[:, 1])]
    D = D.copy()
    D[~np.isfinite(D[:, 1]), 1] = cap
    return D


def betti_curve(D: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """``β(t) = #{(b, d) : b <= t < d}``."""
    D = np.asarray(D, dtype=float).reshape(-1, 2)
    ts = np.asarray(ts, dtype=float)
    return np.sum((D[:, 0][None, :] <= ts[:, None]) & (ts[:, None] < D[:, 1][None, :]), axis=1)


def landscapes(D: np.ndarray, ts: np.ndarray, k: int = 3) -> np.ndarray:
    """Paisajes de persistencia (Bubenik): ``λ_j(t)`` es el ``j``-ésimo mayor
    valor de ``max(0, min(t − b, d − t))``. Devuelve un array (k, len(ts))."""
    D = np.asarray(D, dtype=float).reshape(-1, 2)
    ts = np.asarray(ts, dtype=float)
    if len(D) == 0:
        return np.zeros((k, len(ts)))
    tents = np.maximum(0.0, np.minimum(ts[None, :] - D[:, 0][:, None], D[:, 1][:, None] - ts[None, :]))
    tents = -np.sort(-tents, axis=0)
    out = np.zeros((k, len(ts)))
    m = min(k, len(D))
    out[:m] = tents[:m]
    return out


def persistence_image(D: np.ndarray, birth_range: tuple[float, float],
                      pers_range: tuple[float, float], resolution: int = 20,
                      sigma: float = 0.05, weight: str = "lineal") -> np.ndarray:
    """Imagen de persistencia (Adams et al., 2017).

    Cada punto ``(b, d)`` se lleva a coordenadas ``(b, p = d − b)`` y se
    sustituye por una gaussiana de desviación ``sigma`` ponderada por
    ``w(p)`` (``"lineal"``: ``p / p_max``; ``"constante"``: 1). El valor de
    cada píxel es la integral exacta de la suma de gaussianas sobre él.
    Devuelve una matriz (resolución_p, resolución_b) con la persistencia en
    las filas (fila 0 = persistencia mínima).
    """
    D = finite_part(D)
    bx = np.linspace(birth_range[0], birth_range[1], resolution + 1)
    py = np.linspace(pers_range[0], pers_range[1], resolution + 1)
    img = np.zeros((resolution, resolution))
    if len(D) == 0:
        return img
    b = D[:, 0]
    p = D[:, 1] - D[:, 0]
    if weight == "lineal":
        pmax = p.max() if p.max() > 0 else 1.0
        w = p / pmax
    else:
        w = np.ones_like(p)
    s = sigma * np.sqrt(2)
    Ix = 0.5 * (erf((bx[None, 1:] - b[:, None]) / s) - erf((bx[None, :-1] - b[:, None]) / s))
    Iy = 0.5 * (erf((py[None, 1:] - p[:, None]) / s) - erf((py[None, :-1] - p[:, None]) / s))
    img = np.einsum("n,ni,nj->ji", w, Ix, Iy)
    return img


def persistent_entropy(D: np.ndarray) -> float:
    """Entropía de persistencia ``E = −Σ p_i log p_i`` con
    ``p_i = ℓ_i / Σ ℓ`` (solo barras finitas de longitud positiva)."""
    D = finite_part(D)
    L = D[:, 1] - D[:, 0]
    L = L[L > 0]
    if len(L) == 0:
        return 0.0
    P = L / L.sum()
    return float(-np.sum(P * np.log(P)))
