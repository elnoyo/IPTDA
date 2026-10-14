"""Sistema de temas (claro / oscuro): colores de la interfaz, hoja de estilo
Qt y parámetros de Matplotlib.

Los colores de las gráficas se leen como atributos de este módulo
(``style.POINT_COLOR``…), que ``set_theme`` actualiza; así cada redibujado
usa el tema activo.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# ---------------------------------------------------------------------------
# Paletas
# ---------------------------------------------------------------------------

THEMES = {
    "light": {
        "bg": "#eef0f4",
        "surface": "#ffffff",
        "surface2": "#f6f7f9",
        "border": "#dfe2e8",
        "text": "#161a22",
        "text2": "#5a6170",
        "muted": "#9aa1ad",
        "grid": "#eceef2",
        "accent": "#2a78d6",
        "accent_hover": "#1f66bd",
        "accent_soft": "#e6effb",
        "accent_text": "#ffffff",
        "sidebar": "#111827",
        "sidebar_text": "#c9cfdb",
        "sidebar_muted": "#6b7383",
        "sidebar_hover": "#1c2536",
        "error": "#b3261e",
        # datos (paleta categórica validada para daltonismo; ver docs)
        "dims": ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"],
        "point": "#1f2430",
        "edge": "#5a6170",
        "triangle": "#2a78d6",
        "highlight": "#e34948",
        "seq": ["#e3edf9", "#9fc2ee", "#2a78d6", "#174a8b", "#0b2647"],
    },
    "dark": {
        "bg": "#0e1116",
        "surface": "#171b22",
        "surface2": "#1e232c",
        "border": "#2a303b",
        "text": "#e9ecf1",
        "text2": "#a9b0bd",
        "muted": "#5f6775",
        "grid": "#232832",
        "accent": "#3987e5",
        "accent_hover": "#5a9cec",
        "accent_soft": "#1b2a40",
        "accent_text": "#ffffff",
        "sidebar": "#0a0d12",
        "sidebar_text": "#c9cfdb",
        "sidebar_muted": "#5f6775",
        "sidebar_hover": "#161b24",
        "error": "#f2b8b5",
        "dims": ["#3987e5", "#d95926", "#199e70", "#9085e9"],
        "point": "#e9ecf1",
        "edge": "#8a93a3",
        "triangle": "#3987e5",
        "highlight": "#e66767",
        "seq": ["#1b2433", "#1f4f8c", "#3987e5", "#8fbcf2", "#e3eefc"],
    },
}

current = "light"
T: dict = dict(THEMES["light"])

# Atributos leídos por las funciones de dibujo (se actualizan en set_theme)
DIM_COLORS: list = []
POINT_COLOR = EDGE_COLOR = TRIANGLE_COLOR = BALL_COLOR = HIGHLIGHT = ""
MUTED = TEXT = TEXT2 = SURFACE = GRID = ""
SEQ_CMAP = None
SEQ_POINTS = None     # escala para colorear puntos: sin el tramo que se confunde con el fondo


def set_theme(name: str):
    """Activa el tema ``"light"`` o ``"dark"`` (colores y Matplotlib)."""
    global current, T, DIM_COLORS, POINT_COLOR, EDGE_COLOR, TRIANGLE_COLOR, BALL_COLOR
    global HIGHLIGHT, MUTED, TEXT, TEXT2, SURFACE, GRID, SEQ_CMAP, SEQ_POINTS
    current = name if name in THEMES else "light"
    T = dict(THEMES[current])
    DIM_COLORS = list(T["dims"])
    POINT_COLOR = T["point"]
    EDGE_COLOR = T["edge"]
    TRIANGLE_COLOR = BALL_COLOR = T["triangle"]
    HIGHLIGHT = T["highlight"]
    MUTED = T["muted"]
    TEXT = T["text"]
    TEXT2 = T["text2"]
    SURFACE = T["surface"]
    GRID = T["grid"]
    SEQ_CMAP = LinearSegmentedColormap.from_list(f"iptda_seq_{current}", T["seq"])
    SEQ_POINTS = ListedColormap(SEQ_CMAP(np.linspace(0.3, 1.0, 256)), name=f"iptda_pts_{current}")
    apply_matplotlib_style()


def dim_color(k: int) -> str:
    return DIM_COLORS[k % len(DIM_COLORS)]


_FONT = None


def _ui_font() -> str:
    """Primera fuente disponible de una lista de fuentes modernas."""
    global _FONT
    if _FONT is None:
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        _FONT = next((f for f in ("Inter", "Segoe UI", "Helvetica Neue", "Arial", "Roboto",
                                  "Noto Sans", "DejaVu Sans") if f in names), "DejaVu Sans")
    return _FONT


def apply_matplotlib_style():
    mpl.rcParams.update({
        "figure.facecolor": T["surface"],
        "savefig.facecolor": T["surface"],
        "axes.facecolor": T["surface"],
        "axes.edgecolor": T["border"],
        "axes.labelcolor": T["text2"],
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": T["text"],
        "axes.titlepad": 8,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": T["grid"],
        "grid.linewidth": 0.7,
        "xtick.color": T["text2"],
        "ytick.color": T["text2"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.labelcolor": T["text2"],
        "legend.facecolor": T["surface"],
        "legend.edgecolor": T["border"],
        "text.color": T["text"],
        "font.size": 9,
        "font.family": [_ui_font(), "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "image.cmap": "viridis",
    })


# ---------------------------------------------------------------------------
# Hoja de estilo de Qt
# ---------------------------------------------------------------------------

def qss() -> str:
    t = T
    return f"""
* {{ font-family: "Inter", "Segoe UI", "Helvetica Neue", "Arial", sans-serif; }}
QWidget {{ font-size: 10pt; color: {t['text']}; }}
QMainWindow, QWidget#central, QWidget#panelRoot {{ background: {t['bg']}; }}
QToolTip {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']};
            padding: 4px 6px; border-radius: 6px; }}

/* ---------- barra lateral ---------- */
QFrame#sidebar {{ background: {t['sidebar']}; border: none; }}
QLabel#brand {{ color: #ffffff; font-size: 17pt; font-weight: 700; letter-spacing: 1px; }}
QLabel#brandSub {{ color: {t['sidebar_muted']}; font-size: 8.5pt; }}
QListWidget#nav {{ background: transparent; border: none; outline: 0; color: {t['sidebar_text']};
                   font-size: 10.5pt; }}
QListWidget#nav::item {{ padding: 7px 8px; margin: 1px 8px; border-radius: 8px; }}
QListWidget#nav::item:hover {{ background: {t['sidebar_hover']}; }}
QListWidget#nav::item:selected {{ background: {t['accent']}; color: #ffffff; }}
QPushButton#sideButton {{ background: transparent; color: {t['sidebar_text']}; border: 1px solid #2a3344;
                          border-radius: 8px; padding: 7px 10px; text-align: left; }}
QPushButton#sideButton:hover {{ background: {t['sidebar_hover']}; }}

/* ---------- cabecera de panel ---------- */
QFrame#panelHeader {{ background: transparent; }}
QLabel#panelTitle {{ font-size: 17pt; font-weight: 700; color: {t['text']}; }}
QLabel#panelSubtitle {{ font-size: 10pt; color: {t['text2']}; }}
QProgressBar#busy {{ border: none; background: {t['accent_soft']}; border-radius: 2px; max-height: 4px; }}
QProgressBar#busy::chunk {{ background: {t['accent']}; border-radius: 2px; }}

/* ---------- tarjetas ---------- */
QFrame#card, QFrame#plotCard, QTextBrowser#theory {{
    background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 12px; }}
QGroupBox {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 12px;
             margin-top: 0px; padding: 30px 12px 10px 12px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: padding; subcontrol-position: top left; left: 14px; top: 10px;
                    color: {t['accent']}; font-size: 9pt; font-weight: 700; letter-spacing: 0.5px;
                    text-transform: uppercase; }}
QScrollArea#controlsScroll, QScrollArea#controlsScroll > QWidget > QWidget {{ background: transparent; border: none; }}
QLabel#badge {{ background: {t['surface']}; border: 1px solid {t['border']}; border-left: 4px solid {t['accent']};
                border-radius: 10px; padding: 9px 12px; color: {t['text']}; }}
QLabel#status {{ color: {t['text2']}; }}
QLabel#hint {{ color: {t['text2']}; font-size: 9pt; }}
QTextBrowser#theory {{ padding: 10px 14px; selection-background-color: {t['accent_soft']}; }}

/* ---------- controles ---------- */
QPushButton {{ background: {t['surface2']}; border: 1px solid {t['border']}; border-radius: 8px;
               padding: 7px 14px; color: {t['text']}; }}
QPushButton:hover {{ border-color: {t['accent']}; }}
QPushButton:checked {{ background: {t['accent_soft']}; border-color: {t['accent']}; color: {t['accent']}; }}
QPushButton#primary {{ background: {t['accent']}; color: {t['accent_text']}; border: none; font-weight: 600; }}
QPushButton#primary:hover {{ background: {t['accent_hover']}; }}
QPushButton#primary:disabled {{ background: {t['muted']}; }}
QPushButton#link {{ background: transparent; border: none; color: {t['accent']}; font-weight: 600;
                    padding: 4px 0px; text-align: left; }}
QPushButton#link:hover {{ text-decoration: underline; }}
QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QLineEdit {{
    background: {t['surface2']}; border: 1px solid {t['border']}; border-radius: 7px; padding: 4px 8px;
    color: {t['text']}; selection-background-color: {t['accent']}; min-height: 20px; }}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {t['accent']}; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {{ border-color: {t['accent']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {t['surface']}; border: 1px solid {t['border']};
    selection-background-color: {t['accent']}; selection-color: #ffffff; outline: 0; padding: 4px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    border: none; width: 16px; background: transparent; }}
QCheckBox {{ spacing: 8px; color: {t['text']}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid {t['muted']};
                        background: {t['surface2']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']}; }}
QSlider::groove:horizontal {{ height: 4px; background: {t['border']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {t['surface']}; border: 2px solid {t['accent']};
                              width: 14px; height: 14px; margin: -7px 0; border-radius: 9px; }}
QSlider::handle:horizontal:hover {{ background: {t['accent']}; }}
QLabel#sliderValue {{ color: {t['text']}; font-weight: 600; }}

/* ---------- pestañas, tablas, barras de desplazamiento ---------- */
QTabWidget::pane {{ border: 1px solid {t['border']}; border-radius: 12px; background: {t['surface']}; top: -1px; }}
QTabBar::tab {{ background: transparent; color: {t['text2']}; padding: 8px 16px; border: none;
                border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:selected {{ color: {t['accent']}; border-bottom: 2px solid {t['accent']}; }}
QTabBar::tab:hover {{ color: {t['text']}; }}
QTableWidget {{ background: {t['surface']}; gridline-color: {t['grid']}; border: 1px solid {t['border']};
                border-radius: 10px; color: {t['text']}; }}
QHeaderView::section {{ background: {t['surface2']}; color: {t['text2']}; border: none;
                        border-bottom: 1px solid {t['border']}; padding: 4px 6px; font-size: 8.5pt; }}
QTableCornerButton::section {{ background: {t['surface2']}; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {t['muted']}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t['border']}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: transparent; }}
QToolBar {{ background: {t['surface']}; border: none; spacing: 2px; }}
QToolButton {{ border: none; border-radius: 6px; padding: 3px; }}
QToolButton:hover {{ background: {t['accent_soft']}; }}
QToolButton:checked {{ background: {t['accent_soft']}; }}
QMenuBar {{ background: {t['bg']}; color: {t['text']}; }}
QMenuBar::item:selected {{ background: {t['accent_soft']}; border-radius: 4px; }}
QMenu {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 5px; }}
QMenu::item:selected {{ background: {t['accent']}; color: #ffffff; }}
QStatusBar {{ background: {t['bg']}; color: {t['text2']}; }}
"""


def palette():
    """Paleta de Qt coherente con el tema (para widgets nativos)."""
    from PySide6.QtGui import QColor, QPalette
    t = T
    p = QPalette()
    p.setColor(QPalette.Window, QColor(t["bg"]))
    p.setColor(QPalette.WindowText, QColor(t["text"]))
    p.setColor(QPalette.Base, QColor(t["surface"]))
    p.setColor(QPalette.AlternateBase, QColor(t["surface2"]))
    p.setColor(QPalette.Text, QColor(t["text"]))
    p.setColor(QPalette.Button, QColor(t["surface2"]))
    p.setColor(QPalette.ButtonText, QColor(t["text"]))
    p.setColor(QPalette.Highlight, QColor(t["accent"]))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ToolTipBase, QColor(t["surface"]))
    p.setColor(QPalette.ToolTipText, QColor(t["text"]))
    p.setColor(QPalette.PlaceholderText, QColor(t["muted"]))
    p.setColor(QPalette.Link, QColor(t["accent"]))
    return p


def theory_css() -> str:
    """Estilo del texto teórico (QTextBrowser admite un subconjunto de CSS)."""
    t = T
    return (f"h3 {{ color: {t['text']}; font-size: 13pt; margin-bottom: 4px; }}"
            f"p, li {{ color: {t['text']}; line-height: 140%; }}"
            f"b {{ color: {t['text']}; }}"
            f"code {{ background-color: {t['surface2']}; }}"
            f"a {{ color: {t['accent']}; }}")


set_theme("light")
