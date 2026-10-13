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
                        cycle_dims: tuple[int, ...] | None = None) -> PersistenceResult:
    """Calcula la persistencia de ``filt`` hasta la dimensión homológica
    ``max_dim`` (por defecto ``filt.reliable_dim``).

    ``cycles``: guardar representantes de las barras finitas.
    ``essential_cycles``: guardar también representantes de las infinitas.
    ``cycle_dims``: dimensiones para las que se guardan representantes
    (por defecto todas las >= 1).
    """
    if max_dim is None:
        max_dim = filt.reliable_dim
    max_dim = max(0, min(max_dim, filt.max_dim))
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
