"""Homología persistente con coeficientes en Z/2.

Algoritmo estándar de reducción de la matriz borde por columnas
(Edelsbrunner–Letscher–Zomorodian, Zomorodian–Carlsson) con la optimización
de «limpieza» (*clearing*, Chen–Kerber): se reducen primero las columnas de
mayor dimensión y cada pivote ``low(j) = i`` permite anular la columna ``i``
sin reducirla.

Cada columna se guarda como un conjunto de índices; sumar columnas en Z/2
es la diferencia simétrica.

Representantes:

* Barra finita ``[b, d)`` de dimensión ``k``: la columna reducida ``R_j``
  del símplice ``j`` que la mata es un ``k``-ciclo que nace en ``b``.
* Barra infinita: si se piden (``essential_cycles=True``) se acumula la
  matriz ``V`` (``R = ∂V``) y la columna ``V_i`` del símplice que la crea es
  un ciclo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .filtration import Filtration

#: Tolerancia (relativa) bajo la cual una barra se considera de longitud nula.
ZERO_TOL = 1e-9


@dataclass
class PersistencePair:
    dim: int
    birth: float
    death: float                      # np.inf para barras esenciales
    birth_index: int
    death_index: int | None
    cycle: list[tuple[int, ...]] | None = None   # representante (símplices de dim `dim`)
    coefficients: list[int] | None = None        # coeficientes del ciclo (solo si p > 2)

    @property
    def persistence(self) -> float:
        return self.death - self.birth

    @property
    def is_essential(self) -> bool:
        return self.death_index is None


def is_zero_length(p: PersistencePair) -> bool:
    return p.death - p.birth <= ZERO_TOL * max(1.0, abs(p.birth))


@dataclass
class PersistenceResult:
    pairs: list[PersistencePair]
    max_dim: int
    filtration: Filtration | None = field(default=None, repr=False)
    field: int = 2                    # característica del cuerpo de coeficientes Z/p

    def by_dim(self, dim: int, include_zero: bool = False) -> list[PersistencePair]:
        """Pares de dimensión ``dim``. Por defecto se omiten los de
        persistencia nula (o nula salvo error de redondeo)."""
        return [p for p in self.pairs if p.dim == dim and (include_zero or not is_zero_length(p))]

    def diagram(self, dim: int, include_zero: bool = False) -> np.ndarray:
        """Diagrama de persistencia de dimensión ``dim`` como array (k, 2)."""
        P = self.by_dim(dim, include_zero)
        if not P:
            return np.zeros((0, 2))
        return np.array([[p.birth, p.death] for p in P], dtype=float)

    def betti_at(self, t: float, dim: int) -> int:
        """Número de Betti del complejo ``K_t``: barras con ``b <= t < d``."""
        return sum(1 for p in self.pairs if p.dim == dim and p.birth <= t < p.death)

    def betti_numbers_at(self, t: float) -> list[int]:
        return [self.betti_at(t, k) for k in range(self.max_dim + 1)]


def compute_persistence(filt: Filtration, max_dim: int | None = None,
                        cycles: bool = True, essential_cycles: bool = False,
                        cycle_dims: tuple[int, ...] | None = None,
                        field: int = 2) -> PersistenceResult:
    """Calcula la persistencia de ``filt`` hasta la dimensión homológica
    ``max_dim`` (por defecto ``filt.reliable_dim``) con coeficientes en el
    cuerpo Z/p, ``p = field`` primo (por defecto Z/2).

    ``cycles``: guardar representantes de las barras finitas.
    ``essential_cycles``: guardar también representantes de las infinitas.
    ``cycle_dims``: dimensiones para las que se guardan representantes
    (por defecto todas las >= 1).

    Si no se piden representantes se usa la reducción de la matriz
    cofrontera (cohomología persistente), que da los mismos pares y es mucho
    más rápida en complejos de Rips (de Silva, Morozov, Vejdemo-Johansson,
    2011; Bauer, 2021).
    """
    if max_dim is None:
        max_dim = filt.reliable_dim
    max_dim = max(0, min(max_dim, filt.max_dim))
    if not _is_prime(field):
        raise ValueError(f"{field} no es primo: Z/{field} no es un cuerpo.")
    if field != 2:
        return _persistence_mod_p(filt, max_dim, field, cycles, essential_cycles, cycle_dims)
    if not cycles and not essential_cycles:
        return _cohomology_persistence(filt, max_dim)
    simplices = filt.simplices
    values = filt.values
    dims = filt.dims
    N = len(simplices)
    bnd = filt.boundary()
    want = (lambda k: k >= 1) if cycle_dims is None else (lambda k: k in cycle_dims)

    R: list[set | None] = [None] * N
    V: list[set | None] = [None] * N
    pivot_of: dict[int, int] = {}
    cleared = np.zeros(N, dtype=bool)

    # Se reducen las columnas de dimensión max_dim+1 (que matan clases de
    # dimensión max_dim) hasta 1. Las columnas de vértices son nulas.
    top = min(max_dim + 1, filt.max_dim)
    for d in range(top, 0, -1):
        track_v = essential_cycles and want(d)
        cols = np.nonzero(dims == d)[0]
        for j in cols:
            j = int(j)
            if cleared[j]:
                R[j] = set()
                continue
            col = set(bnd[j])
            v = {j} if track_v else None
            while col:
                low = max(col)
                k = pivot_of.get(low)
                if k is None:
                    break
                col ^= R[k]
                if track_v:
                    v ^= V[k]
            R[j] = col
            if track_v:
                V[j] = v
            if col:
                pivot_of[max(col)] = j
                cleared[max(col)] = True

    pairs: list[PersistencePair] = []
    death_of = {low: j for low, j in pivot_of.items()}
    for i in range(N):
        d = int(dims[i])
        if d > max_dim:
            continue
        if R[i]:            # columna negativa: mata una clase de dim d-1
            continue
        j = death_of.get(i)
        if j is None:
            cyc = None
            if essential_cycles and want(d):
                if d == 0:
                    cyc = [simplices[i]]
                elif V[i] is not None:
                    cyc = [simplices[a] for a in sorted(V[i])]
            pairs.append(PersistencePair(d, float(values[i]), np.inf, i, None, cyc))
        else:
            cyc = None
            if cycles and want(d) and d >= 1:
                cyc = [simplices[a] for a in sorted(R[j])]
            pairs.append(PersistencePair(d, float(values[i]), float(values[j]), i, j, cyc))
    return PersistenceResult(pairs, max_dim, filt)


def coboundary_lists(dims: np.ndarray, bnd: list[list[int]], max_dim: int) -> list[list[int]]:
    """Cocaras de codimensión 1 de cada símplice (hasta dimensión max_dim+1)."""
    cob: list[list[int]] = [[] for _ in range(len(bnd))]
    for j, b in enumerate(bnd):
        if dims[j] <= max_dim + 1:
            for i in b:
                cob[i].append(j)
    return cob


def cohomology_pairs(dims: np.ndarray, cob: list[list[int]], max_dim: int,
                     active: np.ndarray | None = None) -> list[tuple[int, int, int | None]]:
    """Reducción de la matriz cofrontera (cohomología persistente, Z/2).

    Las columnas (k-símplices) se procesan en orden de filtración inverso y
    el pivote de una columna es su cocara de menor índice. Una columna de un
    k-símplice ``i`` con pivote ``j`` da el par (i, j). Limpieza: los
    k-símplices que ya mataron una clase de dimensión k−1 se omiten. Las
    columnas que se anulan dan las clases esenciales.

    ``active`` (máscara booleana) permite trabajar con un subcomplejo
    (cerrado por caras) cuyo orden de filtración es el heredado.
    Devuelve tuplas ``(dim, i, j)`` con ``j = None`` para las esenciales.
    """
    N = len(cob)
    out = []
    killed = np.zeros(N, dtype=bool)
    for d in range(0, max_dim + 1):
        sel = dims == d
        if active is not None:
            sel &= active
        pivot_of: dict[int, set] = {}
        for i in np.nonzero(sel)[0][::-1]:
            i = int(i)
            if killed[i]:
                continue
            if active is None:
                col = set(cob[i])
            else:
                col = {j for j in cob[i] if active[j]}
            while col:
                low = min(col)
                other = pivot_of.get(low)
                if other is None:
                    break
                col ^= other
            if col:
                j = min(col)
                pivot_of[j] = col
                killed[j] = True
                out.append((d, i, j))
            else:
                out.append((d, i, None))
    return out


def _cohomology_persistence(filt: Filtration, max_dim: int) -> PersistenceResult:
    dims = filt.dims
    cob = coboundary_lists(dims, filt.boundary(), max_dim)
    v = filt.values
    pairs = [PersistencePair(d, float(v[i]), np.inf if j is None else float(v[j]), i, j)
             for d, i, j in cohomology_pairs(dims, cob, max_dim)]
    pairs.sort(key=lambda p: p.birth_index)
    return PersistenceResult(pairs, max_dim, filt)


# ---------------------------------------------------------------------------
# Coeficientes en Z/p (p primo impar)
# ---------------------------------------------------------------------------

def _is_prime(p: int) -> bool:
    p = int(p)
    return p >= 2 and all(p % q for q in range(2, int(p ** 0.5) + 1))


def _signed_boundary(filt: Filtration) -> list[dict[int, int]]:
    """Columnas de la matriz borde orientada:
    ∂[v0..vk] = Σ (−1)^i [v0..v̂i..vk]."""
    idx = filt.index
    out = []
    for s in filt.simplices:
        col = {}
        if len(s) > 1:
            for i in range(len(s)):
                col[idx[s[:i] + s[i + 1:]]] = -1 if i % 2 else 1
        out.append(col)
    return out


def _axpy(col: dict, c: int, other: dict, p: int):
    """col ← col − c·other  (mod p), eliminando las entradas nulas."""
    for r, v in other.items():
        nv = (col.get(r, 0) - c * v) % p
        if nv:
            col[r] = nv
        else:
            col.pop(r, None)


def _scale(col: dict, c: int, p: int) -> dict:
    return {r: (v * c) % p for r, v in col.items()}


def _persistence_mod_p(filt: Filtration, max_dim: int, p: int, cycles: bool,
                       essential_cycles: bool, cycle_dims) -> PersistenceResult:
    """Reducción por columnas de la matriz borde con coeficientes en Z/p.

    Igual que en Z/2, pero cada columna es un diccionario {fila: coeficiente}
    y para anular el pivote se resta el múltiplo adecuado de la columna
    pivote (normalizada para que su pivote valga 1)."""
    if not cycles and not essential_cycles:
        return _cohomology_mod_p(filt, max_dim, p)
    simplices = filt.simplices
    values = filt.values
    dims = filt.dims
    N = len(simplices)
    bnd = _signed_boundary(filt)
    want = (lambda k: k >= 1) if cycle_dims is None else (lambda k: k in cycle_dims)
    R: list[dict | None] = [None] * N
    V: list[dict | None] = [None] * N
    pivot_of: dict[int, int] = {}
    cleared = np.zeros(N, dtype=bool)
    top = min(max_dim + 1, filt.max_dim)
    for d in range(top, 0, -1):
        track_v = essential_cycles and want(d)
        for j in np.nonzero(dims == d)[0]:
            j = int(j)
            if cleared[j]:
                R[j] = {}
                continue
            col = {r: v % p for r, v in bnd[j].items()}
            v = {j: 1} if track_v else None
            while col:
                low = max(col)
                k = pivot_of.get(low)
                if k is None:
                    break
                c = col[low]
                _axpy(col, c, R[k], p)
                if track_v:
                    _axpy(v, c, V[k], p)
            if col:
                inv = pow(col[max(col)], -1, p)       # normaliza: pivote = 1
                col = _scale(col, inv, p)
                if track_v:
                    v = _scale(v, inv, p)
                pivot_of[max(col)] = j
                cleared[max(col)] = True
            R[j] = col
            if track_v:
                V[j] = v

    def as_cycle(chain: dict):
        keys = sorted(chain)
        return [simplices[a] for a in keys], [int(chain[a]) for a in keys]

    pairs: list[PersistencePair] = []
    for i in range(N):
        d = int(dims[i])
        if d > max_dim or R[i]:
            continue
        j = pivot_of.get(i)
        if j is None:
            cyc = coef = None
            if essential_cycles and want(d):
                if d == 0:
                    cyc, coef = [simplices[i]], [1]
                elif V[i] is not None:
                    cyc, coef = as_cycle(V[i])
            pairs.append(PersistencePair(d, float(values[i]), np.inf, i, None, cyc, coef))
        else:
            cyc = coef = None
            if cycles and want(d) and d >= 1:
                cyc, coef = as_cycle(R[j])
            pairs.append(PersistencePair(d, float(values[i]), float(values[j]), i, j, cyc, coef))
    return PersistenceResult(pairs, max_dim, filt, field=p)


def _cohomology_mod_p(filt: Filtration, max_dim: int, p: int) -> PersistenceResult:
    """Reducción de la matriz cofrontera con coeficientes en Z/p (mismos
    pares que la reducción de la matriz borde; ver ``cohomology_pairs``)."""
    dims = filt.dims
    values = filt.values
    N = len(filt.simplices)
    cob: list[dict[int, int]] = [{} for _ in range(N)]
    for j, col in enumerate(_signed_boundary(filt)):
        if dims[j] <= max_dim + 1:
            for i, sgn in col.items():
                cob[i][j] = sgn % p
    pairs = []
    killed = np.zeros(N, dtype=bool)
    for d in range(0, max_dim + 1):
        pivot_of: dict[int, dict] = {}
        for i in np.nonzero(dims == d)[0][::-1]:
            i = int(i)
            if killed[i]:
                continue
            col = dict(cob[i])
            while col:
                low = min(col)
                other = pivot_of.get(low)
                if other is None:
                    break
                _axpy(col, col[low], other, p)
            if col:
                j = min(col)
                pivot_of[j] = _scale(col, pow(col[j], -1, p), p)
                killed[j] = True
                pairs.append(PersistencePair(d, float(values[i]), float(values[j]), i, j))
            else:
                pairs.append(PersistencePair(d, float(values[i]), np.inf, i, None))
    pairs.sort(key=lambda q: q.birth_index)
    return PersistenceResult(pairs, max_dim, filt, field=p)
