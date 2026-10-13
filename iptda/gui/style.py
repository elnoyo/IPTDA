"""Colores y estilo comunes de la interfaz y de las gráficas."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

#: Un color fijo por dimensión homológica (H0, H1, H2, H3). Las tres
#: primeras tonalidades forman una paleta categórica validada para
#: daltonismo; nunca se reasignan.
DIM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]

POINT_COLOR = "#1f1f1e"
EDGE_COLOR = "#52514e"
TRIANGLE_COLOR = "#2a78d6"
TETRA_COLOR = "#6250d6"
BALL_COLOR = "#2a78d6"
HIGHLIGHT = "#e34948"
MUTED = "#b5b4ae"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
SURFACE = "#fcfcfb"

#: Escala secuencial de una sola tonalidad (claro → oscuro).
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "iptda_blues", ["#dbe8f7", "#8fb8ea", "#2a78d6", "#174a8b", "#0b2647"])


def dim_color(k: int) -> str:
    return DIM_COLORS[k % len(DIM_COLORS)]


def apply_matplotlib_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT2,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
    })


QSS = """
QWidget { font-size: 10pt; }
QGroupBox { font-weight: bold; border: 1px solid #d6d5cf; border-radius: 6px;
            margin-top: 10px; padding-top: 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #174a8b; }
QListWidget#nav { background: #f2f1ec; border: none; font-size: 11pt; }
QListWidget#nav::item { padding: 9px 10px; border-radius: 6px; margin: 1px 4px; }
QListWidget#nav::item:selected { background: #2a78d6; color: white; }
QTextBrowser { background: #fcfcfb; border: 1px solid #e0dfd9; border-radius: 6px; }
QPushButton { padding: 5px 10px; }
QPushButton#primary { background: #2a78d6; color: white; border-radius: 5px; font-weight: bold; }
QPushButton#primary:disabled { background: #9dbbe2; }
QLabel#status { color: #52514e; }
QLabel#badge { background: #eef4fc; border: 1px solid #c9dcf3; border-radius: 5px; padding: 6px; }
"""
