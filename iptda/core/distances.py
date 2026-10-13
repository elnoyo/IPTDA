"""Distancias entre diagramas de persistencia.

Ambas distancias usan la norma del máximo (L∞) en el plano y permiten
emparejar cualquier punto con su proyección sobre la diagonal, a distancia
``(d − b) / 2``.

* Cuello de botella: ``d_B = ínf_γ sup_x ‖x − γ(x)‖∞`` (exacta: búsqueda
  binaria sobre los costes candidatos + emparejamiento bipartito perfecto).
* Wasserstein-p: ``W_p = (ínf_γ Σ_x ‖x − γ(x)‖∞^p)^{1/p}`` (exacta:
  algoritmo húngaro sobre la matriz aumentada).

Los puntos con muerte infinita solo pueden emparejarse entre sí (por la
coordenada de nacimiento); si su número difiere, la distancia es infinita.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


@dataclass
class Matching:
    value: float
    # pares (i, j): i índice en A o -1 (diagonal), j índice en B o -1 (diagonal)
    pairs: list[tuple[int, int]]


def _split(D):
    D = np.asarray(D, dtype=float).reshape(-1, 2)
    fin = np.isfinite(D[:, 1])
    return np.nonzero(fin)[0], np.nonzero(~fin)[0], D


def _augmented_cost(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matriz (m+k)×(m+k): filas = A ∪ proyecciones de B; columnas = B ∪
    proyecciones de A."""
    m, k = len(A), len(B)
    C = np.full((m + k, m + k), np.inf)
    if m and k:
        C[:m, :k] = np.max(np.abs(A[:, None, :] - B[None, :, :]), axis=2)
    if m:
        C[np.arange(m), k + np.arange(m)] = (A[:, 1] - A[:, 0]) / 2
    if k:
        C[m + np.arange(k), np.arange(k)] = (B[:, 1] - B[:, 0]) / 2
    C[m:, k:] = 0.0
    return C


def _essential_part(DA, ia, DB, ib, p=None):
    """Empareja los puntos esenciales ordenando los nacimientos."""
    if len(ia) != len(ib):
        return np.inf, []
    oa = ia[np.argsort(DA[ia, 0])]
    ob = ib[np.argsort(DB[ib, 0])]
    diffs = np.abs(DA[oa, 0] - DB[ob, 0])
    pairs = [(int(a), int(b)) for a, b in zip(oa, ob)]
    if p is None:
        return (float(diffs.max()) if len(diffs) else 0.0), pairs
    return float(np.sum(diffs ** p)), pairs


def bottleneck(DA, DB) -> Matching:
    fa, ia, DA = _split(DA)
    fb, ib, DB = _split(DB)
    ess, ess_pairs = _essential_part(DA, ia, DB, ib)
    if not np.isfinite(ess):
        return Matching(np.inf, [])
    A, B = DA[fa], DB[fb]
    m, k = len(A), len(B)
    if m + k == 0:
        return Matching(ess, ess_pairs)
    C = _augmented_cost(A, B)
    cand = np.unique(C[np.isfinite(C)])
    lo, hi = 0, len(cand) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        G = csr_matrix((C <= cand[mid]).astype(np.int8))
        match = maximum_bipartite_matching(G, perm_type="column")
        if np.all(match >= 0):
            best = (cand[mid], match)
            hi = mid - 1
        else:
            lo = mid + 1
    value, match = best
    pairs = list(ess_pairs)
    for row, col in enumerate(match):
        i = int(fa[row]) if row < m else -1
        j = int(fb[col]) if col < k else -1
        if i == -1 and j == -1:
            continue
        pairs.append((i, j))
    return Matching(float(max(value, ess)), pairs)


def wasserstein(DA, DB, p: float = 2.0) -> Matching:
    fa, ia, DA = _split(DA)
    fb, ib, DB = _split(DB)
    ess, ess_pairs = _essential_part(DA, ia, DB, ib, p)
    if not np.isfinite(ess):
        return Matching(np.inf, [])
    A, B = DA[fa], DB[fb]
    m, k = len(A), len(B)
    total = ess
    pairs = list(ess_pairs)
    if m + k:
        C = _augmented_cost(A, B)
        Cp = np.where(np.isfinite(C), C ** p, 1e300)
        rows, cols = linear_sum_assignment(Cp)
        total += float(Cp[rows, cols].sum())
        for r, c in zip(rows, cols):
            i = int(fa[r]) if r < m else -1
            j = int(fb[c]) if c < k else -1
            if i == -1 and j == -1:
                continue
            pairs.append((i, j))
    return Matching(total ** (1.0 / p), pairs)
