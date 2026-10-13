"""Complejos simpliciales filtrados y sus construcciones.

Convención de escala (común a toda la aplicación)
-------------------------------------------------
El parámetro de filtración ``r`` es el **radio** de las bolas centradas en
los puntos:

* Vietoris–Rips: un símplice entra en ``r = diámetro / 2`` (todas las
  distancias entre sus vértices son ``<= 2r``; equivale a que las bolas se
  corten **dos a dos**).
* Čech: un símplice entra en el radio de la bola mínima que contiene a sus
  vértices (las bolas tienen un punto **común**).
* Alfa: radio de la filtración alfa (bolas recortadas por las celdas de
  Voronoi), con los mismos diagramas de persistencia que Čech.

Con esta convención se cumple ``Čech(r) ⊆ Rips(r) ⊆ Čech(√2·r)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.spatial import Delaunay

from .geometry import circumspheres, min_enclosing_radii, maxmin_subsample, pairwise_distances


@dataclass
class Filtration:
    """Complejo simplicial filtrado.

    ``simplices[i]`` es una tupla ordenada de vértices y ``values[i]`` el
    valor en que aparece. Tras ``sort()`` las caras siempre preceden a sus
    cocaras (orden por valor, después por dimensión).

    ``reliable_dim`` es la mayor dimensión homológica calculada sin
    truncamiento: si se construyen símplices hasta dimensión ``D``, la
    homología de dimensión ``D`` no es fiable (faltan los ``D+1``-símplices
    que la matarían), por lo que ``reliable_dim = D - 1``.
    """

    simplices: list[tuple[int, ...]]
    values: np.ndarray
    reliable_dim: int
    name: str = ""
    _index: dict | None = field(default=None, repr=False)

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.sort()

    def __len__(self):
        return len(self.simplices)

    def sort(self):
        dims = np.fromiter((len(s) for s in self.simplices), dtype=int, count=len(self.simplices))
        order = np.lexsort((dims, self.values))
        self.simplices = [self.simplices[i] for i in order]
        self.values = self.values[order]
        self._index = None

    @property
    def dims(self) -> np.ndarray:
        return np.fromiter((len(s) - 1 for s in self.simplices), dtype=int, count=len(self.simplices))

    @property
    def max_dim(self) -> int:
        return max((len(s) - 1 for s in self.simplices), default=-1)

    @property
    def index(self) -> dict:
        if self._index is None:
            self._index = {s: i for i, s in enumerate(self.simplices)}
        return self._index

    def boundary(self) -> list[list[int]]:
        """Índices de las caras de codimensión 1 de cada símplice."""
        idx = self.index
        out = []
        for s in self.simplices:
            if len(s) == 1:
                out.append([])
            else:
                out.append([idx[s[:i] + s[i + 1:]] for i in range(len(s))])
        return out

    def simplices_at(self, t: float, dim: int | None = None) -> list[tuple[int, ...]]:
        """Símplices presentes en el parámetro ``t`` (valor <= t)."""
        k = int(np.searchsorted(self.values, t, side="right"))
        S = self.simplices[:k]
        if dim is not None:
            S = [s for s in S if len(s) == dim + 1]
        return S

    def counts_at(self, t: float) -> list[int]:
        k = int(np.searchsorted(self.values, t, side="right"))
        c = [0] * (self.max_dim + 1)
        for s in self.simplices[:k]:
            c[len(s) - 1] += 1
        return c

    def euler_at(self, t: float) -> int:
        return sum((-1) ** d * c for d, c in enumerate(self.counts_at(t)))

    def check(self) -> bool:
        """Comprueba que es una filtración: las caras están presentes y su
        valor no supera el del símplice."""
        idx = self.index
        for s, v in zip(self.simplices, self.values):
            if len(s) > 1:
                for i in range(len(s)):
                    f = s[:i] + s[i + 1:]
                    j = idx.get(f)
                    if j is None or self.values[j] > v + 1e-12:
                        return False
        return True


# ---------------------------------------------------------------------------
# Complejos bandera (Rips, testigos)
# ---------------------------------------------------------------------------

def flag_filtration(E: np.ndarray, max_dim: int, max_value: float = np.inf,
                    vertex_values: np.ndarray | None = None,
                    vertex_labels: np.ndarray | None = None,
                    max_simplices: int = 2_000_000, name: str = "") -> Filtration:
    """Complejo bandera de la matriz simétrica de valores de arista ``E``.

    Un símplice entra en el máximo de los valores de sus aristas (y de sus
    vértices). Solo se incluyen símplices con valor ``<= max_value``.
    ``vertex_labels`` permite renombrar los vértices (p. ej. índices de
    los puntos originales de unos puntos de referencia).
    """
    E = np.asarray(E, dtype=float)
    n = len(E)
    vv = np.zeros(n) if vertex_values is None else np.asarray(vertex_values, dtype=float)
    labels = np.arange(n) if vertex_labels is None else np.asarray(vertex_labels)
    # Valor efectivo de una arista: max(valor de arista, valores de los vértices)
    Ev = np.maximum(E, np.maximum.outer(vv, vv))
    keep_v = vv <= max_value
    adj = (Ev <= max_value) & keep_v[:, None] & keep_v[None, :]
    np.fill_diagonal(adj, False)

    simplices: list[tuple] = []
    values: list[float] = []
    for i in range(n):
        if keep_v[i]:
            simplices.append((i,))
            values.append(vv[i])

    # vecinos "superiores" como máscaras de bits (enteros de Python)
    upper = [0] * n
    for i in range(n):
        js = np.nonzero(adj[i, i + 1:])[0] + i + 1
        m = 0
        for j in js:
            m |= 1 << int(j)
        upper[i] = m

    def bits(m):
        while m:
            low = m & -m
            yield low.bit_length() - 1
            m ^= low

    # Expansión incremental de cliques: (simplice, valor, candidatos comunes)
    level = []
    if max_dim >= 1:
        for i in range(n):
            for j in bits(upper[i]):
                v = Ev[i, j]
                simplices.append((i, j))
                values.append(v)
                level.append(((i, j), v, upper[i] & upper[j]))
    for d in range(2, max_dim + 1):
        nxt = []
        for s, v, cand in level:
            for k in bits(cand):
                w = v
                for a in s:
                    if Ev[a, k] > w:
                        w = Ev[a, k]
                t = s + (k,)
                simplices.append(t)
                values.append(w)
                if d < max_dim:
                    nxt.append((t, w, cand & upper[k]))
            if len(simplices) > max_simplices:
                raise MemoryError(
                    f"El complejo supera {max_simplices} símplices; reduzca el número "
                    "de puntos, el radio máximo o la dimensión.")
        level = nxt

    if vertex_labels is not None:
        simplices = [tuple(int(labels[a]) for a in s) for s in simplices]
        simplices = [tuple(sorted(s)) for s in simplices]
    return Filtration(simplices, np.array(values), reliable_dim=max_dim - 1, name=name)


def rips_filtration(X: np.ndarray, max_dim: int = 2, max_radius: float = np.inf,
                    max_simplices: int = 2_000_000) -> Filtration:
    """Filtración de Vietoris–Rips con la convención de radio (diámetro/2)."""
    D = pairwise_distances(X) / 2.0
    return flag_filtration(D, max_dim, max_radius, max_simplices=max_simplices,
                           name="Vietoris–Rips")


# ---------------------------------------------------------------------------
# Čech
# ---------------------------------------------------------------------------

def cech_filtration(X: np.ndarray, max_dim: int = 2, max_radius: float = np.inf,
                    max_simplices: int = 2_000_000) -> Filtration:
    """Filtración de Čech: el valor de un símplice es el radio de la bola
    mínima que contiene a sus vértices. Como ``Čech(r) ⊆ Rips(r)``, los
    candidatos se obtienen del complejo de Rips con el mismo radio máximo."""
    X = np.asarray(X, dtype=float)
    R = rips_filtration(X, max_dim, max_radius, max_simplices)
    by_dim: dict[int, list[tuple]] = {}
    for s in R.simplices:
        by_dim.setdefault(len(s), []).append(s)
    simplices, values = [], []
    for k, S in by_dim.items():
        arr = np.array(S, dtype=int)
        r = min_enclosing_radii(X, arr)
        keep = r <= max_radius
        simplices.extend(S[i] for i in np.nonzero(keep)[0])
        values.extend(r[keep])
    return Filtration(simplices, np.array(values), reliable_dim=max_dim - 1, name="Čech")


# ---------------------------------------------------------------------------
# Alfa
# ---------------------------------------------------------------------------

def alpha_filtration(X: np.ndarray) -> Filtration:
    """Filtración alfa a partir de la triangulación de Delaunay.

    Para cada símplice ``τ`` de Delaunay (de mayor a menor dimensión):

    * si ``τ`` es de Gabriel (su esfera circunscrita mínima no contiene
      estrictamente ningún otro punto), su valor es su radio circunscrito;
    * si no, su valor es el mínimo de los valores de sus cocaras.

    Basta comprobar los vértices opuestos de las cocaras de Delaunay.
    Los valores son radios (no radios al cuadrado).
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    tri = Delaunay(X)
    top = [tuple(sorted(int(v) for v in s)) for s in tri.simplices]
    # todas las caras
    faces: dict[int, set] = {k: set() for k in range(1, d + 2)}
    for s in top:
        for k in range(1, d + 2):
            for f in combinations(s, k):
                faces[k].add(f)
    value: dict[tuple, float] = {}
    cofaces: dict[tuple, list] = {}
    for k in range(d + 1, 1, -1):
        S = sorted(faces[k])
        arr = np.array(S, dtype=int)
        centers, radii = circumspheres(X[arr])
        for s, c, r in zip(S, centers, radii):
            cf = cofaces.get(s)
            if not cf:
                value[s] = float(r)
            else:
                gabriel = True
                for (sig, opp) in cf:
                    if np.linalg.norm(X[opp] - c) < r * (1 - 1e-12):
                        gabriel = False
                        break
                if gabriel:
                    value[s] = float(r)
                else:
                    value[s] = min(value[sig] for sig, _ in cf)
            for i in range(len(s)):
                f = s[:i] + s[i + 1:]
                if len(f) >= 2:
                    cofaces.setdefault(f, []).append((s, s[i]))
    for f in faces[1]:
        value[f] = 0.0
    simplices = list(value.keys())
    values = np.array([value[s] for s in simplices])
    return Filtration(simplices, values, reliable_dim=d, name="Alfa")


# ---------------------------------------------------------------------------
# Testigos
# ---------------------------------------------------------------------------

def witness_filtration(X: np.ndarray, n_landmarks: int = 30, nu: int = 1,
                       max_dim: int = 2, max_value: float = np.inf,
                       first_landmark: int = 0) -> tuple[Filtration, np.ndarray]:
    """Complejo de testigos «perezoso» (de Silva–Carlsson).

    Los puntos de referencia ``L`` se eligen por maxmin; todos los puntos
    actúan como testigos. Para ``ν ∈ {0, 1, 2}``, ``m_w`` es la distancia
    del testigo ``w`` a su ``ν``-ésimo punto de referencia más cercano
    (``m_w = 0`` si ``ν = 0``), y la arista ``[a, b]`` entra en

        min_w max(d(a, w), d(b, w)) − m_w.

    Los símplices superiores entran cuando lo hacen todas sus aristas
    (complejo bandera). Los vértices llevan las etiquetas de los puntos
    originales. Devuelve la filtración y los índices de referencia.
    """
    X = np.asarray(X, dtype=float)
    L = maxmin_subsample(X, n_landmarks, first_landmark)
    D = pairwise_distances(X[L], X)                    # (|L|, n)
    if nu == 0:
        m = np.zeros(D.shape[1])
    else:
        m = np.sort(D, axis=0)[nu - 1]
    k = len(L)
    E = np.zeros((k, k))
    for a in range(k):
        E[a] = np.min(np.maximum(D[a][None, :], D) - m[None, :], axis=1)
    E = np.maximum(E, E.T)
    np.fill_diagonal(E, 0.0)
    F = flag_filtration(E, max_dim, max_value, vertex_labels=L, name="Testigos (perezoso)")
    return F, L


# ---------------------------------------------------------------------------
# Filtraciones de subnivel
# ---------------------------------------------------------------------------

def lower_star_path(f: np.ndarray) -> Filtration:
    """Filtración de subnivel de una función sobre un camino (serie temporal):
    el vértice ``i`` entra en ``f[i]`` y la arista ``[i, i+1]`` en
    ``max(f[i], f[i+1])``."""
    f = np.asarray(f, dtype=float)
    n = len(f)
    simplices = [(i,) for i in range(n)] + [(i, i + 1) for i in range(n - 1)]
    values = np.concatenate([f, np.maximum(f[:-1], f[1:])])
    return Filtration(simplices, values, reliable_dim=1, name="Subnivel")


def static_complex(maximal: list[tuple], name: str = "") -> Filtration:
    """Clausura de una lista de símplices maximales, todos con valor 0."""
    S = set()
    for s in maximal:
        s = tuple(sorted(int(v) for v in s))
        if len(set(s)) != len(s):
            raise ValueError(f"Símplice con vértices repetidos: {s}")
        for k in range(1, len(s) + 1):
            S.update(combinations(s, k))
    simplices = sorted(S, key=lambda s: (len(s), s))
    top = max((len(s) for s in simplices), default=1) - 1
    return Filtration(simplices, np.zeros(len(simplices)), reliable_dim=top, name=name)


METHODS = ["Vietoris–Rips", "Čech", "Alfa", "Testigos (perezoso)"]


def build_filtration(method: str, X: np.ndarray, max_dim: int, max_radius: float,
                     n_landmarks: int = 30, nu: int = 1):
    """Construye la filtración pedida. Devuelve ``(filtración, landmarks)``;
    ``landmarks`` es ``None`` salvo en el complejo de testigos."""
    if method == "Vietoris–Rips":
        return rips_filtration(X, max_dim, max_radius), None
    if method == "Čech":
        return cech_filtration(X, max_dim, max_radius), None
    if method == "Alfa":
        F = alpha_filtration(X)
        return F, None
    if method == "Testigos (perezoso)":
        return witness_filtration(X, n_landmarks, nu, max_dim, max_radius)
    raise ValueError(method)
