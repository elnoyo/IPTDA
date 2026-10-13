"""Utilidades geométricas: distancias, esferas circunscritas, bolas mínimas
envolventes y submuestreo maxmin."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.spatial.distance import cdist

_EPS = 1e-10


def pairwise_distances(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return cdist(X, X if Y is None else np.asarray(Y, dtype=float))


def circumspheres(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Esfera circunscrita «mínima» de cada símplice, con centro en la
    envolvente afín del símplice (válido en cualquier dimensión ambiente).

    ``P`` tiene forma ``(m, k+1, d)``: ``m`` símplices de ``k+1`` vértices.
    Devuelve ``(centros (m, d), radios (m,))``. Los símplices degenerados
    (vértices afínmente dependientes) reciben radio ``inf``.
    """
    P = np.asarray(P, dtype=float)
    m, kp1, d = P.shape
    if kp1 == 1:
        return P[:, 0, :].copy(), np.zeros(m)
    p0 = P[:, 0, :]
    U = P[:, 1:, :] - p0[:, None, :]                 # (m, k, d)
    G = np.einsum("mid,mjd->mij", U, U)              # matrices de Gram (m, k, k)
    rhs = 0.5 * np.einsum("mii->mi", G)              # (m, k)
    centers = np.full((m, d), np.nan)
    radii = np.full(m, np.inf)
    # Detección de degenerados: determinante de Gram relativo
    scale = np.maximum(np.einsum("mii->m", G), _EPS) / max(kp1 - 1, 1)
    det = np.linalg.det(G) / scale ** (kp1 - 1)
    ok = np.abs(det) > 1e-12
    if np.any(ok):
        lam = np.linalg.solve(G[ok], rhs[ok][..., None])[..., 0]
        c = p0[ok] + np.einsum("mi,mid->md", lam, U[ok])
        centers[ok] = c
        radii[ok] = np.linalg.norm(c - p0[ok], axis=1)
    return centers, radii


def min_enclosing_ball(P: np.ndarray) -> tuple[np.ndarray, float]:
    """Bola mínima envolvente de un conjunto pequeño de puntos ``P`` (k, d).

    Método exacto por candidatos: la bola mínima es la bola circunscrita
    mínima de algún subconjunto soporte; se toma la menor de las que
    contienen a todos los puntos.
    """
    P = np.asarray(P, dtype=float)
    k = len(P)
    if k == 1:
        return P[0].copy(), 0.0
    best_c, best_r = None, np.inf
    for size in range(2, k + 1):
        for S in combinations(range(k), size):
            c, r = circumspheres(P[list(S)][None])
            c, r = c[0], r[0]
            if not np.isfinite(r) or r >= best_r:
                continue
            if np.all(np.linalg.norm(P - c, axis=1) <= r * (1 + 1e-9) + 1e-12):
                best_c, best_r = c, r
    return best_c, float(best_r)


def min_enclosing_radii(X: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Radio de la bola mínima envolvente de cada símplice (vectorizado).

    ``simplices`` es un array ``(m, k+1)`` de índices de vértices.
    """
    simplices = np.asarray(simplices, dtype=int)
    m, kp1 = simplices.shape
    if m == 0:
        return np.zeros(0)
    P = X[simplices]                                  # (m, k+1, d)
    if kp1 == 1:
        return np.zeros(m)
    if kp1 == 2:
        return 0.5 * np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
    # Candidatos: bolas circunscritas mínimas de todos los subconjuntos de
    # tamaño >= 2. Para cada símplice, la menor que contiene a todos.
    best = np.full(m, np.inf)
    for size in range(2, kp1 + 1):
        for S in combinations(range(kp1), size):
            c, r = circumspheres(P[:, list(S), :])
            finite = np.isfinite(r)
            if not np.any(finite):
                continue
            dist = np.linalg.norm(P - np.where(finite[:, None], c, 0.0)[:, None, :], axis=2)
            inside = np.all(dist <= (r * (1 + 1e-9) + 1e-12)[:, None], axis=1) & finite
            best = np.where(inside & (r < best), r, best)
    return best


def maxmin_subsample(X: np.ndarray, k: int, first: int = 0) -> np.ndarray:
    """Submuestreo por el punto más lejano (maxmin). Devuelve índices.
    Se detiene antes si los puntos restantes repiten puntos ya elegidos."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    k = min(k, n)
    idx = [first]
    dmin = np.linalg.norm(X - X[first], axis=1)
    for _ in range(1, k):
        j = int(np.argmax(dmin))
        if dmin[j] == 0:
            break
        idx.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(X - X[j], axis=1))
    return np.array(idx, dtype=int)


def hausdorff_distance(X: np.ndarray, Y: np.ndarray) -> float:
    D = pairwise_distances(X, Y)
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))
