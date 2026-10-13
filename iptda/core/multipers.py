"""Persistencia biparamétrica: bifiltración Rips × codensidad.

Cada símplice ``σ`` tiene un bigrado ``(r_σ, c_σ)``:

* ``r_σ``: valor de Rips (diámetro / 2);
* ``c_σ``: máximo de la codensidad de sus vértices (cuanto mayor, más
  aislado está el punto).

El complejo en ``(r, c)`` es ``K(r, c) = {σ : r_σ <= r, c_σ <= c}``; si
``(r, c) <= (r', c')`` coordenada a coordenada, ``K(r, c) ⊆ K(r', c')``.
Es una bifiltración «uniformemente crítica» (un bigrado por símplice).

Invariantes calculados:

* Función de Hilbert: ``HF_k(r, c) = dim H_k(K(r, c))``.
* Código de barras fibrado: restricción a una recta de pendiente
  no negativa ``L(t) = a + t·u``; el símplice entra en
  ``t_σ = max_i (σ_i − a_i) / u_i`` (si ``u_i = 0`` se exige ``σ_i <= a_i``).
* Invariante de rango: ``rk(a → b) = rango(H_k(K(a)) → H_k(K(b)))`` para
  ``a <= b``; es el número de barras del código fibrado de la recta que pasa
  por ``a`` y ``b`` que contienen al intervalo ``[t_a, t_b]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .filtration import Filtration, rips_filtration
from .geometry import pairwise_distances
from .persistence import (PersistenceResult, coboundary_lists, cohomology_pairs,
                          compute_persistence)

CODENSITIES = ["Distancia al k-ésimo vecino", "Densidad gaussiana (invertida)"]


def codensity(X: np.ndarray, kind: str, k: int = 5, sigma: float = 0.3) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if kind == "Distancia al k-ésimo vecino":
        d, _ = cKDTree(X).query(X, k=k + 1)      # el primero es el propio punto
        return d[:, -1]
    if kind == "Densidad gaussiana (invertida)":
        D = pairwise_distances(X)
        dens = np.exp(-(D / sigma) ** 2 / 2).sum(axis=1)
        return 1.0 - dens / dens.max()
    raise ValueError(kind)


@dataclass
class Bifiltration:
    simplices: list[tuple[int, ...]]
    r: np.ndarray
    c: np.ndarray
    reliable_dim: int
    vertex_codensity: np.ndarray

    def complex_at(self, r: float, c: float) -> list[tuple[int, ...]]:
        mask = (self.r <= r) & (self.c <= c)
        return [s for s, m in zip(self.simplices, mask) if m]

    def restrict(self, values: np.ndarray, mask: np.ndarray) -> Filtration:
        idx = np.nonzero(mask)[0]
        return Filtration([self.simplices[i] for i in idx], values[idx],
                          reliable_dim=self.reliable_dim)


def rips_codensity_bifiltration(X: np.ndarray, cod: np.ndarray, max_dim: int = 2,
                                max_radius: float = np.inf) -> Bifiltration:
    """Bifiltración Rips × codensidad; los símplices quedan en el orden de
    la filtración de Rips (por r y después por dimensión)."""
    F = rips_filtration(X, max_dim, max_radius)
    cod = np.asarray(cod, dtype=float)
    c = np.array([cod[list(s)].max() for s in F.simplices])
    return Bifiltration(list(F.simplices), F.values.copy(), c, F.reliable_dim, cod)


def hilbert_function(B: Bifiltration, r_grid: np.ndarray, c_grid: np.ndarray,
                     dims=(0, 1), progress=None) -> dict[int, np.ndarray]:
    """``HF[k][j, i] = dim H_k(K(r_grid[i], c_grid[j]))``.

    Para cada ``c_j`` se calcula una sola vez la persistencia de la
    filtración de Rips de ``{σ : c_σ <= c_j}`` y se evalúa en todos los
    ``r_i``."""
    r_grid = np.asarray(r_grid, dtype=float)
    c_grid = np.asarray(c_grid, dtype=float)
    out = {k: np.zeros((len(c_grid), len(r_grid)), dtype=int) for k in dims}
    # Los símplices de B están ordenados por r (y dimensión): cada
    # subcomplejo {c_σ <= c_j} hereda ese orden, así que la matriz cofrontera
    # global se reutiliza restringida a una máscara.
    top = max(dims)
    F = Filtration(list(B.simplices), B.r.copy(), reliable_dim=B.reliable_dim)
    if F.simplices != B.simplices:      # no debería ocurrir: B ya está ordenada
        raise RuntimeError("La bifiltración no está en orden de filtración de Rips.")
    dim_arr = F.dims
    cob = coboundary_lists(dim_arr, F.boundary(), top)
    for j, cj in enumerate(c_grid):
        mask = B.c <= cj
        if np.any(mask):
            for d, a, b in cohomology_pairs(dim_arr, cob, top, active=mask):
                if d not in out:
                    continue
                birth = B.r[a]
                death = np.inf if b is None else B.r[b]
                if death > birth:
                    out[d][j] += (birth <= r_grid) & (r_grid < death)
        if progress is not None:
            progress(j + 1, len(c_grid))
    return out


def line_entry_times(B: Bifiltration, a: tuple[float, float], u: tuple[float, float]) -> np.ndarray:
    """Valor ``t_σ`` en que cada símplice entra en ``K(a + t·u)``
    (``inf`` si nunca entra)."""
    ar, ac = a
    ur, uc = u
    if ur < 0 or uc < 0 or (ur == 0 and uc == 0):
        raise ValueError("La dirección debe tener componentes no negativas y no ser nula.")
    t = np.full(len(B.simplices), -np.inf)
    if ur > 0:
        t = np.maximum(t, (B.r - ar) / ur)
    else:
        t = np.where(B.r <= ar, t, np.inf)
    if uc > 0:
        t = np.maximum(t, (B.c - ac) / uc)
    else:
        t = np.where(B.c <= ac, t, np.inf)
    return t


def fibered_barcode(B: Bifiltration, a: tuple[float, float], u: tuple[float, float],
                    max_dim: int = 1) -> PersistenceResult:
    """Persistencia de la filtración 1-paramétrica ``t ↦ K(a + t·u)``."""
    t = line_entry_times(B, a, u)
    mask = np.isfinite(t)
    F = B.restrict(t, mask)
    return compute_persistence(F, max_dim=max_dim, cycles=False)


def rank_invariant(B: Bifiltration, a, b, dim: int) -> int:
    """``rango(H_dim(K(a)) → H_dim(K(b)))`` para ``a <= b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.any(b < a):
        raise ValueError("Se necesita a <= b coordenada a coordenada.")
    u = b - a
    if np.all(u == 0):
        u = np.array([1.0, 0.0])
        tb = 0.0
    else:
        tb = 1.0
    P = fibered_barcode(B, tuple(a), tuple(u), max_dim=dim)
    return sum(1 for p in P.pairs if p.dim == dim and p.birth <= 0.0 and tb < p.death)
