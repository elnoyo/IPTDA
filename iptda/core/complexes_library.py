"""Biblioteca de complejos simpliciales clásicos para el panel de homología.

Cada entrada tiene una representación de **dibujo** (coordenadas de
vértices de dibujo y símplices maximales de dibujo) y una **etiqueta**
abstracta por vértice de dibujo. Así, las superficies que no se pueden
sumergir en R³ (botella de Klein, plano proyectivo) se dibujan sobre su
polígono fundamental: varios vértices de dibujo representan el mismo
vértice abstracto (los lados se identifican).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .filtration import Filtration, static_complex


@dataclass
class LibraryComplex:
    name: str
    points: np.ndarray                     # coordenadas de dibujo (m, 2) o (m, 3)
    display_maximal: list[tuple[int, ...]]  # símplices maximales en índices de dibujo
    labels: np.ndarray                      # vértice de dibujo -> vértice abstracto
    description: str = ""
    expected: dict = field(default_factory=dict)   # cuerpo -> números de Betti esperados
    identified: bool = False                # True si se dibuja un polígono fundamental

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    def abstract_maximal(self) -> list[tuple[int, ...]]:
        return [tuple(sorted(int(self.labels[v]) for v in s)) for s in self.display_maximal]

    def filtration(self) -> Filtration:
        return static_complex(self.abstract_maximal(), name=self.name)

    def display_occurrences(self, simplex: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Copias de dibujo de un símplice abstracto (caras de los símplices
        de dibujo cuyas etiquetas coinciden con él)."""
        from itertools import combinations
        target = tuple(sorted(simplex))
        k = len(target)
        out = set()
        for s in self.display_maximal:
            if len(s) < k:
                continue
            for f in combinations(s, k):
                if tuple(sorted(int(self.labels[v]) for v in f)) == target:
                    out.add(tuple(sorted(f)))
        return sorted(out)


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def grid_surface(n: int, m: int, ident: str) -> tuple[np.ndarray, list, np.ndarray]:
    """Rejilla ``n × m`` sobre el cuadrado unidad, cada casilla dividida en
    dos triángulos, con identificaciones de los lados:

    * ``"toro"``: (0, y) ~ (1, y), (x, 0) ~ (x, 1)
    * ``"klein"``: (0, y) ~ (1, 1 − y), (x, 0) ~ (x, 1)
    * ``"mobius"``: (0, y) ~ (1, 1 − y)
    * ``"cilindro"``: (0, y) ~ (1, y)
    * ``"disco"``: sin identificaciones
    """
    def vid(i, j):
        return i * (m + 1) + j
    pts = np.array([[i / n, j / m] for i in range(n + 1) for j in range(m + 1)])
    uf = _UF(len(pts))
    if ident in ("toro", "cilindro"):
        for j in range(m + 1):
            uf.union(vid(0, j), vid(n, j))
    if ident in ("klein", "mobius"):
        for j in range(m + 1):
            uf.union(vid(0, j), vid(n, m - j))
    if ident in ("toro", "klein"):
        for i in range(n + 1):
            uf.union(vid(i, 0), vid(i, m))
    roots = {}
    labels = np.array([roots.setdefault(uf.find(v), len(roots)) for v in range(len(pts))])
    tris = []
    for i in range(n):
        for j in range(m):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tris.append((a, b, c))
            tris.append((a, c, d))
    return pts, tris, labels


def _torus_3d(n=7, m=4, R=2.0, r=0.8):
    pts, tris = [], []
    for i in range(n):
        for j in range(m):
            u, v = 2 * np.pi * i / n, 2 * np.pi * j / m
            pts.append(((R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)))

    def vid(i, j):
        return (i % n) * m + (j % m)
    for i in range(n):
        for j in range(m):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tris.append((a, b, c))
            tris.append((a, c, d))
    return np.array(pts), tris


def _identity(k):
    return np.arange(k)


#: tamaños de rejilla que dan complejos simpliciales válidos
#: (comprobado en tests/test_homology.py). La rejilla no sirve para el plano
#: proyectivo: produce triángulos degenerados.
_GRID_SIZES = {"toro": (3, 3), "klein": (3, 3), "mobius": (5, 1), "cilindro": (4, 1)}


def build_library() -> dict[str, LibraryComplex]:
    L: list[LibraryComplex] = []

    def add(name, pts, maximal, labels=None, desc="", expected=None, identified=False):
        pts = np.asarray(pts, dtype=float)
        L.append(LibraryComplex(name, pts, [tuple(s) for s in maximal],
                                _identity(len(pts)) if labels is None else np.asarray(labels),
                                desc, expected or {}, identified))

    add("Punto", [[0, 0]], [(0,)], desc="Un único vértice: contráctil.",
        expected={"Z/2": [1]})
    add("Dos puntos", [[0, 0], [1, 0]], [(0,), (1,)],
        desc="Dos componentes conexas: β0 = 2.", expected={"Z/2": [2]})
    add("Segmento", [[0, 0], [1, 0]], [(0, 1)], desc="Un 1-símplice: contráctil.",
        expected={"Z/2": [1, 0]})
    tri = [[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]]
    add("Triángulo lleno (disco)", tri, [(0, 1, 2)],
        desc="Un 2-símplice: contráctil. El ciclo del borde es el borde del triángulo.",
        expected={"Z/2": [1, 0, 0]})
    add("Borde del triángulo (circunferencia)", tri, [(0, 1), (1, 2), (0, 2)],
        desc="Tres aristas sin relleno: un 1-ciclo que no es borde, β1 = 1.",
        expected={"Z/2": [1, 1]})
    eight = [[0, 0], [-1, 0.6], [-1, -0.6], [1, 0.6], [1, -0.6]]
    add("Ocho (dos circunferencias unidas)", eight,
        [(0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (0, 4)],
        desc="Unión por un punto de dos circunferencias: β1 = 2.",
        expected={"Z/2": [1, 2]})
    # corona triangulada: dos triángulos concéntricos
    outer = [[np.cos(a), np.sin(a)] for a in np.pi / 2 + 2 * np.pi * np.arange(3) / 3]
    inner = [[0.4 * np.cos(a), 0.4 * np.sin(a)] for a in -np.pi / 2 + 2 * np.pi * np.arange(3) / 3]
    ann_t = [(0, 1, 5), (1, 2, 3), (2, 0, 4), (0, 4, 5), (1, 5, 3), (2, 3, 4)]
    add("Corona (disco con agujero)", outer + inner, ann_t,
        desc="Disco con un agujero: los triángulos rellenan todo salvo el centro, β1 = 1.",
        expected={"Z/2": [1, 1, 0]})
    tet = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    add("Tetraedro sólido (bola)", tet, [(0, 1, 2, 3)],
        desc="Un 3-símplice: contráctil.", expected={"Z/2": [1, 0, 0, 0]})
    add("Borde del tetraedro (esfera S²)", tet, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
        desc="Cuatro triángulos que encierran un hueco: β2 = 1.",
        expected={"Z/2": [1, 0, 1], "Q": [1, 0, 1]})
    oct_pts = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    oct_t = [(a, b, c) for a in (0, 1) for b in (2, 3) for c in (4, 5)]
    add("Octaedro (esfera S²)", oct_pts, oct_t,
        desc="Otra triangulación de la esfera: la homología no depende de la triangulación.",
        expected={"Z/2": [1, 0, 1]})
    tp, tt = _torus_3d()
    add("Toro (en R³)", tp, tt,
        desc="Triangulación del toro sumergida en R³: β = (1, 2, 1).",
        expected={"Z/2": [1, 2, 1], "Q": [1, 2, 1]})
    for key, name, desc, exp in [
        ("toro", "Toro (polígono fundamental)",
         "Cuadrado con lados opuestos identificados en el mismo sentido (aabb → aba⁻¹b⁻¹).",
         {"Z/2": [1, 2, 1], "Q": [1, 2, 1]}),
        ("klein", "Botella de Klein (polígono fundamental)",
         "Un par de lados se identifica invirtiendo el sentido. No es orientable: "
         "sobre Q, β = (1, 1, 0); sobre Z/2, β = (1, 2, 1).",
         {"Z/2": [1, 2, 1], "Q": [1, 1, 0], "Z/3": [1, 1, 0]}),
        ("mobius", "Banda de Möbius",
         "Rectángulo con un par de lados identificados con giro. Retrae por deformación a "
         "su circunferencia central: β = (1, 1, 0).",
         {"Z/2": [1, 1, 0], "Q": [1, 1, 0]}),
        ("cilindro", "Cilindro",
         "Rectángulo con un par de lados identificados sin giro: β = (1, 1, 0).",
         {"Z/2": [1, 1, 0], "Q": [1, 1, 0]}),
    ]:
        n, m = _GRID_SIZES[key]
        pts, tris, labels = grid_surface(n, m, key)
        add(name, pts, tris, labels, desc, exp, identified=True)
    # Plano proyectivo: triangulación mínima de 6 vértices (RP²₆) dibujada
    # sobre un hexágono con los puntos antipodales del borde identificados.
    hexa = [[np.cos(a), np.sin(a)] for a in np.pi / 3 * np.arange(6)]
    inner = [[0.45 * np.cos(a), 0.45 * np.sin(a)] for a in (np.pi / 3, np.pi, 5 * np.pi / 3)]
    H = list(range(6))
    Ia, Ib, Ic = 6, 7, 8
    rp_tris = [(H[0], H[1], Ia), (H[1], H[2], Ia), (Ia, Ib, H[2]),
               (H[2], H[3], Ib), (H[3], H[4], Ib), (Ib, Ic, H[4]),
               (H[4], H[5], Ic), (H[5], H[0], Ic), (Ic, Ia, H[0]), (Ia, Ib, Ic)]
    add("Plano proyectivo RP² (polígono fundamental)", hexa + inner, rp_tris,
        [0, 1, 2, 0, 1, 2, 3, 4, 5],
        "Triangulación mínima (6 vértices, 10 triángulos) sobre un hexágono con los puntos "
        "antipodales del borde identificados. Sobre Q es acíclico, β = (1, 0, 0); sobre Z/2, "
        "β = (1, 1, 1): la torsión de H1(RP²; Z) = Z/2 solo se ve con coeficientes en Z/2.",
        {"Z/2": [1, 1, 1], "Q": [1, 0, 0], "Z/3": [1, 0, 0]}, identified=True)
    return {c.name: c for c in L}


LIBRARY = build_library()
