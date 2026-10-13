"""Generadores de nubes de puntos 2D y 3D con topología conocida.

Cada generador recibe ``n`` (número de puntos), ``noise`` (desviación típica
del ruido gaussiano añadido) y ``rng`` (un ``numpy.random.Generator``), y
devuelve un array de forma ``(n, d)``.

El diccionario ``DATASETS`` registra los conjuntos disponibles junto con su
dimensión ambiente y los números de Betti del espacio subyacente (sin ruido),
que sirven como referencia en los paneles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _noise(X: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    if noise > 0:
        X = X + rng.normal(scale=noise, size=X.shape)
    return X


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------

def circle(n, noise, rng, radius=1.0):
    t = rng.uniform(0, 2 * np.pi, n)
    X = radius * np.column_stack([np.cos(t), np.sin(t)])
    return _noise(X, noise, rng)


def two_circles(n, noise, rng):
    """Dos circunferencias disjuntas de radios 1 y 0,6."""
    n1 = int(round(n * 0.6))
    n2 = n - n1
    A = circle(n1, 0.0, rng, 1.0) + np.array([-1.3, 0.0])
    B = circle(n2, 0.0, rng, 0.6) + np.array([1.2, 0.0])
    return _noise(np.vstack([A, B]), noise, rng)


def figure_eight(n, noise, rng):
    """Dos circunferencias tangentes (un ocho): b0 = 1, b1 = 2."""
    n1 = n // 2
    n2 = n - n1
    A = circle(n1, 0.0, rng) + np.array([-1.0, 0.0])
    B = circle(n2, 0.0, rng) + np.array([1.0, 0.0])
    return _noise(np.vstack([A, B]), noise, rng)


def annulus(n, noise, rng, r_in=0.6, r_out=1.0):
    """Corona circular con densidad uniforme en área."""
    u = rng.uniform(r_in ** 2, r_out ** 2, n)
    r = np.sqrt(u)
    t = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([r * np.cos(t), r * np.sin(t)])
    return _noise(X, noise, rng)


def clusters(n, noise, rng, k=3):
    """``k`` grupos gaussianos (b0 = k)."""
    centers = np.array([[np.cos(a), np.sin(a)] for a in
                        np.linspace(0, 2 * np.pi, k, endpoint=False)]) * 1.5
    labels = np.arange(n) % k
    X = centers[labels] + rng.normal(scale=0.15, size=(n, 2))
    return _noise(X, noise, rng)


def uniform_square(n, noise, rng):
    X = rng.uniform(-1, 1, size=(n, 2))
    return _noise(X, noise, rng)


def circle_with_outliers(n, noise, rng, frac_out=0.2):
    """Circunferencia más puntos atípicos uniformes en el cuadrado."""
    n_out = int(round(n * frac_out))
    A = circle(n - n_out, noise, rng)
    B = rng.uniform(-1.4, 1.4, size=(n_out, 2))
    return np.vstack([A, B])


def y_shape(n, noise, rng):
    """Una «Y»: tres segmentos que se unen en un punto (árbol)."""
    labels = np.arange(n) % 3
    angles = np.array([np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3])
    s = rng.uniform(0, 1, n)
    X = np.column_stack([s * np.cos(angles[labels]), s * np.sin(angles[labels])])
    return _noise(X, noise, rng)


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

def sphere(n, noise, rng, radius=1.0):
    X = rng.normal(size=(n, 3))
    X = radius * X / np.linalg.norm(X, axis=1, keepdims=True)
    return _noise(X, noise, rng)


def torus(n, noise, rng, R=2.0, r=0.8):
    """Toro de revolución muestreado uniformemente respecto al área
    (muestreo por rechazo)."""
    out = []
    while len(out) < n:
        u = rng.uniform(0, 2 * np.pi, 2 * n)
        v = rng.uniform(0, 2 * np.pi, 2 * n)
        w = rng.uniform(0, 1, 2 * n)
        keep = w < (R + r * np.cos(v)) / (R + r)
        for a, b in zip(u[keep], v[keep]):
            out.append(((R + r * np.cos(b)) * np.cos(a),
                        (R + r * np.cos(b)) * np.sin(a),
                        r * np.sin(b)))
    X = np.array(out[:n])
    return _noise(X, noise, rng)


def trefoil(n, noise, rng):
    """Nudo de trébol (topológicamente una circunferencia)."""
    t = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([np.sin(t) + 2 * np.sin(2 * t),
                         np.cos(t) - 2 * np.cos(2 * t),
                         -np.sin(3 * t)]) / 2.0
    return _noise(X, noise, rng)


def hopf_link(n, noise, rng):
    """Dos circunferencias enlazadas (b0 = 2, b1 = 2)."""
    n1 = n // 2
    n2 = n - n1
    t1 = rng.uniform(0, 2 * np.pi, n1)
    t2 = rng.uniform(0, 2 * np.pi, n2)
    A = np.column_stack([np.cos(t1), np.sin(t1), np.zeros(n1)])
    B = np.column_stack([1 + np.cos(t2), np.zeros(n2), np.sin(t2)])
    return _noise(np.vstack([A, B]), noise, rng)


def cylinder(n, noise, rng):
    t = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-1, 1, n)
    X = np.column_stack([np.cos(t), np.sin(t), z])
    return _noise(X, noise, rng)


def two_spheres(n, noise, rng):
    n1 = n // 2
    A = sphere(n1, 0.0, rng) + np.array([-1.3, 0, 0])
    B = sphere(n - n1, 0.0, rng, 0.7) + np.array([1.2, 0, 0])
    return _noise(np.vstack([A, B]), noise, rng)


def swiss_roll(n, noise, rng):
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n))
    h = rng.uniform(0, 1, n) * 3
    X = np.column_stack([t * np.cos(t), h, t * np.sin(t)]) / 5.0
    return _noise(X, noise, rng)


def uniform_cube(n, noise, rng):
    return _noise(rng.uniform(-1, 1, size=(n, 3)), noise, rng)


@dataclass(frozen=True)
class Dataset:
    name: str
    dim: int
    func: Callable
    betti: str  # números de Betti del espacio subyacente, como texto

    def sample(self, n: int, noise: float = 0.0, seed: int | None = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.asarray(self.func(int(n), float(noise), rng), dtype=float)


DATASETS: dict[str, Dataset] = {d.name: d for d in [
    Dataset("Circunferencia", 2, circle, "b0=1, b1=1"),
    Dataset("Dos circunferencias", 2, two_circles, "b0=2, b1=2"),
    Dataset("Ocho (dos círculos tangentes)", 2, figure_eight, "b0=1, b1=2"),
    Dataset("Corona circular", 2, annulus, "b0=1, b1=1"),
    Dataset("Tres grupos", 2, clusters, "b0=3, b1=0"),
    Dataset("Letra Y (árbol)", 2, y_shape, "b0=1, b1=0"),
    Dataset("Circunferencia con atípicos", 2, circle_with_outliers, "b0=1, b1=1 (+ruido)"),
    Dataset("Uniforme en el cuadrado", 2, uniform_square, "b0=1, b1=0"),
    Dataset("Esfera", 3, sphere, "b0=1, b1=0, b2=1"),
    Dataset("Toro", 3, torus, "b0=1, b1=2, b2=1"),
    Dataset("Nudo de trébol", 3, trefoil, "b0=1, b1=1"),
    Dataset("Enlace de Hopf", 3, hopf_link, "b0=2, b1=2"),
    Dataset("Cilindro", 3, cylinder, "b0=1, b1=1"),
    Dataset("Dos esferas", 3, two_spheres, "b0=2, b1=0, b2=2"),
    Dataset("Rollo suizo", 3, swiss_roll, "b0=1, b1=0"),
    Dataset("Uniforme en el cubo", 3, uniform_cube, "b0=1"),
]}


def datasets_of_dim(dim: int) -> list[str]:
    return [name for name, d in DATASETS.items() if d.dim == dim]
