"""Algoritmo Mapper (Singh, Mémoli, Carlsson, 2007).

1. Una función filtro (lente) ``f : X → R``.
2. Un recubrimiento de ``f(X)`` por intervalos solapados.
3. Para cada intervalo ``I``, agrupamiento (clustering) de ``f^{-1}(I)``.
4. El grafo Mapper tiene un nodo por grupo y una arista entre dos grupos
   que comparten puntos (es el 1-esqueleto del nervio del recubrimiento
   refinado por los grupos).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree

from .geometry import pairwise_distances

# ---------------------------------------------------------------------------
# Lentes
# ---------------------------------------------------------------------------

LENSES = ["Coordenada x", "Coordenada y", "Coordenada z", "Distancia al centroide",
          "Excentricidad", "Densidad (núcleo gaussiano)", "Primera componente principal"]


def lens_values(X: np.ndarray, lens: str, sigma: float = 0.3) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if lens == "Coordenada x":
        return X[:, 0].copy()
    if lens == "Coordenada y":
        return X[:, 1].copy()
    if lens == "Coordenada z":
        if X.shape[1] < 3:
            raise ValueError("La nube es 2D: no tiene coordenada z.")
        return X[:, 2].copy()
    if lens == "Distancia al centroide":
        return np.linalg.norm(X - X.mean(axis=0), axis=1)
    if lens == "Excentricidad":
        return pairwise_distances(X).mean(axis=1)
    if lens == "Densidad (núcleo gaussiano)":
        D = pairwise_distances(X)
        return np.exp(-(D / sigma) ** 2 / 2).sum(axis=1) / len(X)
    if lens == "Primera componente principal":
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[0]
    raise ValueError(lens)


# ---------------------------------------------------------------------------
# Recubrimiento
# ---------------------------------------------------------------------------

def uniform_cover(values: np.ndarray, n_intervals: int, overlap: float) -> list[tuple[float, float]]:
    """``n`` intervalos de igual longitud ``L`` que cubren ``[min, max]``,
    cada uno solapado una fracción ``overlap`` (0 <= g < 1) con el siguiente:
    ``L = (max − min) / (n − (n − 1) g)`` y el ``i``-ésimo empieza en
    ``min + i (1 − g) L``."""
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi == lo:
        hi = lo + 1e-9
    g = float(overlap)
    n = int(n_intervals)
    L = (hi - lo) / (n - (n - 1) * g)
    out = []
    for i in range(n):
        a = lo + i * (1 - g) * L
        out.append((a, a + L))
    # cierre numérico del último intervalo
    out[-1] = (out[-1][0], max(out[-1][1], hi))
    return out


# ---------------------------------------------------------------------------
# Agrupamiento
# ---------------------------------------------------------------------------

def single_linkage(X: np.ndarray, threshold: float) -> np.ndarray:
    """Componentes conexas del grafo de vecinos a distancia ``<= threshold``
    (enlace simple cortado a esa altura)."""
    if len(X) == 1:
        return np.zeros(1, dtype=int)
    Z = linkage(X, method="single")
    return fcluster(Z, t=threshold, criterion="distance") - 1


def dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """DBSCAN (Ester et al., 1996). Devuelve etiquetas; ``−1`` = ruido."""
    n = len(X)
    tree = cKDTree(X)
    neigh = tree.query_ball_point(X, r=eps)
    core = np.array([len(nb) >= min_samples for nb in neigh])
    labels = np.full(n, -1)
    c = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = c
        stack = [i]
        while stack:
            p = stack.pop()
            if not core[p]:
                continue
            for q in neigh[p]:
                if labels[q] == -1:
                    labels[q] = c
                    stack.append(q)
        c += 1
    return labels


# ---------------------------------------------------------------------------
# Grafo Mapper
# ---------------------------------------------------------------------------

@dataclass
class MapperGraph:
    nodes: list[np.ndarray]                 # índices de los puntos de cada nodo
    node_interval: list[int]                # intervalo del recubrimiento de cada nodo
    edges: list[tuple[int, int, int]]       # (u, v, número de puntos comunes)
    intervals: list[tuple[float, float]]
    lens: np.ndarray

    @property
    def n_components(self) -> int:
        parent = list(range(len(self.nodes)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for u, v, _ in self.edges:
            parent[find(u)] = find(v)
        return len({find(a) for a in range(len(self.nodes))})

    @property
    def betti(self) -> tuple[int, int]:
        """(β0, β1) del grafo: β1 = E − V + β0 (rango del grupo de ciclos)."""
        b0 = self.n_components
        return b0, len(self.edges) - len(self.nodes) + b0

    def node_lens_mean(self) -> np.ndarray:
        return np.array([self.lens[m].mean() for m in self.nodes])


def mapper(X: np.ndarray, lens: np.ndarray, n_intervals: int = 10, overlap: float = 0.3,
           clustering: str = "Enlace simple", threshold: float = 0.3,
           min_samples: int = 3) -> MapperGraph:
    X = np.asarray(X, dtype=float)
    lens = np.asarray(lens, dtype=float)
    intervals = uniform_cover(lens, n_intervals, overlap)
    nodes, node_int = [], []
    for k, (a, b) in enumerate(intervals):
        idx = np.nonzero((lens >= a) & (lens <= b))[0]
        if len(idx) == 0:
            continue
        if clustering == "Enlace simple":
            lab = single_linkage(X[idx], threshold)
        elif clustering == "DBSCAN":
            lab = dbscan(X[idx], threshold, min_samples)
        else:
            raise ValueError(clustering)
        for c in np.unique(lab):
            if c < 0:
                continue
            nodes.append(idx[lab == c])
            node_int.append(k)
    edges = []
    sets = [set(m.tolist()) for m in nodes]
    for u in range(len(nodes)):
        for v in range(u + 1, len(nodes)):
            w = len(sets[u] & sets[v])
            if w:
                edges.append((u, v, w))
    return MapperGraph(nodes, node_int, edges, intervals, lens)
