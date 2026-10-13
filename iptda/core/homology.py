"""Homología simplicial de complejos finitos.

* Matrices borde orientadas: ``∂[σ_0..σ_k] = Σ_i (−1)^i [σ_0..σ̂_i..σ_k]``.
* Rango sobre Z/p (p primo) por eliminación gaussiana modular y sobre Q de
  forma exacta (eliminación sin fracciones con enteros de Python).
* Números de Betti: ``β_k = dim C_k − rango ∂_k − rango ∂_{k+1}``.
* Generadores (con coeficientes en Z/2): clases esenciales de la reducción
  de la matriz borde del complejo completo.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np

from .filtration import Filtration, static_complex
from .persistence import compute_persistence

FIELDS = {"Z/2": 2, "Z/3": 3, "Z/5": 5, "Q": 0}


def simplices_by_dim(F: Filtration) -> list[list[tuple[int, ...]]]:
    out: list[list] = [[] for _ in range(F.max_dim + 1)]
    for s in F.simplices:
        out[len(s) - 1].append(s)
    for L in out:
        L.sort()
    return out


def boundary_matrix(rows: list[tuple], cols: list[tuple]) -> np.ndarray:
    """Matriz del operador borde ``∂_k : C_k → C_{k−1}`` (enteros con signo).

    ``rows``: (k−1)-símplices; ``cols``: k-símplices."""
    idx = {s: i for i, s in enumerate(rows)}
    M = np.zeros((len(rows), len(cols)), dtype=np.int64)
    for j, s in enumerate(cols):
        for i in range(len(s)):
            f = s[:i] + s[i + 1:]
            M[idx[f], j] = (-1) ** i
    return M


def rank_mod_p(M: np.ndarray, p: int) -> int:
    A = np.array(M, dtype=np.int64) % p
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        if r == rows:
            break
        piv = np.nonzero(A[r:, c])[0]
        if len(piv) == 0:
            continue
        pr = r + int(piv[0])
        if pr != r:
            A[[r, pr]] = A[[pr, r]]
        inv = pow(int(A[r, c]), -1, p)
        A[r] = (A[r] * inv) % p
        nz = np.nonzero(A[:, c])[0]
        for i in nz:
            if i != r:
                A[i] = (A[i] - A[i, c] * A[r]) % p
        r += 1
    return r


def rank_rational(M: np.ndarray) -> int:
    """Rango exacto sobre Q: eliminación sin fracciones con enteros de
    Python, dividiendo cada fila por el mcd de sus entradas."""
    A = [[int(x) for x in row] for row in np.asarray(M)]
    rows = len(A)
    cols = len(A[0]) if rows else 0
    r = 0
    for c in range(cols):
        pr = next((i for i in range(r, rows) if A[i][c] != 0), None)
        if pr is None:
            continue
        A[r], A[pr] = A[pr], A[r]
        a = A[r][c]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                b = A[i][c]
                row = [a * x - b * y for x, y in zip(A[i], A[r])]
                g = 0
                for x in row:
                    g = gcd(g, x)
                A[i] = [x // g for x in row] if g > 1 else row
        r += 1
        if r == rows:
            break
    return r


def rank(M: np.ndarray, p: int) -> int:
    if M.size == 0:
        return 0
    return rank_rational(M) if p == 0 else rank_mod_p(M, p)


@dataclass
class HomologyReport:
    counts: list[int]                 # c_k = número de k-símplices
    ranks: list[int]                  # ranks[k] = rango de ∂_k (∂_0 = 0)
    betti: list[int]
    euler: int
    field: str
    generators: list[list[list[tuple]]]   # generators[k] = lista de ciclos (Z/2)
    dd_zero: bool                     # comprobación ∂_k ∘ ∂_{k+1} = 0


def homology(F: Filtration, field: str = "Z/2", with_generators: bool = True) -> HomologyReport:
    p = FIELDS[field]
    S = simplices_by_dim(F)
    top = len(S) - 1
    counts = [len(L) for L in S]
    ranks = [0] * (top + 2)
    mats = {}
    for k in range(1, top + 1):
        M = boundary_matrix(S[k - 1], S[k])
        mats[k] = M
        ranks[k] = rank(M, p)
    betti = [counts[k] - ranks[k] - ranks[k + 1] for k in range(top + 1)]
    euler = sum((-1) ** k * c for k, c in enumerate(counts))
    dd = all(not np.any(mats[k] @ mats[k + 1]) for k in range(1, top))
    gens: list[list] = [[] for _ in range(top + 1)]
    if with_generators:
        G = Filtration(list(F.simplices), np.zeros(len(F)), reliable_dim=top)
        res = compute_persistence(G, max_dim=top, cycles=False, essential_cycles=True,
                                  cycle_dims=tuple(range(top + 1)))
        for pr in res.pairs:
            if pr.is_essential and pr.cycle is not None:
                gens[pr.dim].append(pr.cycle)
    return HomologyReport(counts, ranks[:top + 1], betti, euler, field, gens, dd)


def parse_complex(text: str) -> Filtration:
    """Interpreta un texto con símplices maximales separados por ``;`` o
    saltos de línea; los vértices (enteros >= 0) se separan por espacios o
    comas. Ejemplo: ``"0 1 2; 2 3; 3 4"``."""
    maximal = []
    for chunk in text.replace("\n", ";").split(";"):
        chunk = chunk.strip().strip("[](){}")
        if not chunk:
            continue
        verts = [int(v) for v in chunk.replace(",", " ").split()]
        if any(v < 0 for v in verts):
            raise ValueError("Los vértices deben ser enteros no negativos.")
        if len(set(verts)) != len(verts):
            raise ValueError(f"Vértice repetido en el símplice {verts}.")
        maximal.append(tuple(verts))
    if not maximal:
        raise ValueError("No se ha indicado ningún símplice.")
    return static_complex(maximal)
