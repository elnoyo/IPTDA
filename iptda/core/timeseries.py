"""Series temporales: señales de ejemplo, inmersión de Takens (ventanas
deslizantes) y ayudas para elegir el retardo."""

from __future__ import annotations

import numpy as np


def lorenz_x(n: int, dt: float = 0.01, transient: int = 1000) -> np.ndarray:
    """Coordenada x del sistema de Lorenz (σ=10, ρ=28, β=8/3), integrado
    con Runge–Kutta de orden 4."""
    def f(s):
        x, y, z = s
        return np.array([10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0 / 3.0 * z])
    s = np.array([1.0, 1.0, 1.0])
    out = np.empty(n)
    for i in range(n + transient):
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i >= transient:
            out[i - transient] = s[0]
    return out


def make_signal(kind: str, n: int = 600, noise: float = 0.0, seed: int = 0,
                period: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve ``(t, x)``. ``period`` está en número de muestras."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    w = 2 * np.pi / period
    if kind == "Seno":
        x = np.sin(w * t)
    elif kind == "Suma de dos senos (cuasiperiódica)":
        x = np.sin(w * t) + 0.8 * np.sin(np.sqrt(3) * w * t)
    elif kind == "Seno amortiguado":
        x = np.exp(-t / (n / 2.5)) * np.sin(w * t)
    elif kind == "Onda cuadrada":
        x = np.sign(np.sin(w * t))
    elif kind == "Chirp (frecuencia creciente)":
        x = np.sin(w * t * (0.5 + t / n))
    elif kind == "Lorenz (coordenada x)":
        x = lorenz_x(n) / 10.0
    elif kind == "Paseo aleatorio":
        x = np.cumsum(rng.normal(size=n)) / np.sqrt(n) * 3
    elif kind == "Ruido blanco":
        x = rng.normal(size=n)
    else:
        raise ValueError(kind)
    if noise > 0:
        x = x + rng.normal(scale=noise, size=n)
    return t, x


SIGNALS = ["Seno", "Suma de dos senos (cuasiperiódica)", "Seno amortiguado", "Onda cuadrada",
           "Chirp (frecuencia creciente)", "Lorenz (coordenada x)", "Paseo aleatorio",
           "Ruido blanco"]


def takens_embedding(x: np.ndarray, dim: int, tau: int, step: int = 1) -> np.ndarray:
    """Inmersión por retardos: ``v_i = (x_i, x_{i+τ}, ..., x_{i+(d−1)τ})``
    para ``i = 0, step, 2·step, ...``."""
    x = np.asarray(x, dtype=float)
    m = len(x) - (dim - 1) * tau
    if m <= 0:
        raise ValueError("La serie es demasiado corta para esa dimensión y retardo.")
    idx = np.arange(0, m, step)[:, None] + tau * np.arange(dim)[None, :]
    return x[idx]


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float) - np.mean(x)
    v = np.dot(x, x)
    if v == 0:
        return np.zeros(max_lag + 1)
    return np.array([np.dot(x[:len(x) - k], x[k:]) / v for k in range(max_lag + 1)])


def suggest_tau(x: np.ndarray, max_lag: int = 200) -> int:
    """Primer cruce por cero de la autocorrelación (o su primer mínimo
    local si no cruza)."""
    ac = autocorrelation(x, min(max_lag, len(x) // 2))
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            return k
    for k in range(1, len(ac) - 1):
        if ac[k] < ac[k - 1] and ac[k] <= ac[k + 1]:
            return k
    return 1


def pca_project(X: np.ndarray, k: int = 3) -> np.ndarray:
    """Proyección sobre las ``k`` primeras componentes principales."""
    Xc = X - X.mean(axis=0)
    if X.shape[1] <= k:
        return Xc
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T
