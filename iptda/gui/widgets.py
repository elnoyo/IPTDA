"""Widgets reutilizables: lienzo de matplotlib, deslizadores de valores
reales, selector de nube de puntos y la clase base de los paneles (con
ejecución de cálculos en segundo plano)."""

from __future__ import annotations

import traceback

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame,
                               QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter,
                               QTextBrowser, QVBoxLayout, QWidget)

from ..core.datasets import DATASETS, datasets_of_dim
from . import style
from .icons import make_icon


class MplCanvas(QFrame):
    """Figura de matplotlib incrustada en una tarjeta, con barra de
    herramientas opcional (zoom, desplazamiento, guardar imagen)."""

    def __init__(self, parent=None, toolbar: bool = True, figsize=(5, 4)):
        super().__init__(parent)
        self.setObjectName("plotCard")
        self.figure = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 8)
        lay.setSpacing(0)
        self.toolbar = None
        if toolbar:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setIconSize(QSize(18, 18))
            lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)
        self.apply_theme()

    def apply_theme(self):
        """Adapta el fondo de la figura y los iconos de la barra al tema."""
        self.figure.set_facecolor(style.SURFACE)
        if self.toolbar is None:
            return
        self.toolbar.setPalette(style.palette())
        try:   # API interna de Matplotlib: si cambia, se conservan los iconos
            for text, _tip, image, callback in self.toolbar.toolitems:
                act = self.toolbar._actions.get(callback) if text else None
                if act is not None:
                    act.setIcon(self.toolbar._icon(image + ".png"))
        except Exception:  # noqa: BLE001
            pass

    def draw(self):
        self.canvas.draw_idle()


class FloatSlider(QWidget):
    """Deslizador de valores reales con etiqueta del valor actual."""

    valueChanged = Signal(float)

    def __init__(self, vmin=0.0, vmax=1.0, value=0.0, steps=1000, fmt="{:.3f}", parent=None):
        super().__init__(parent)
        self._min, self._max, self._steps, self._fmt = float(vmin), float(vmax), steps, fmt
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, steps)
        self.label = QLabel()
        self.label.setObjectName("sliderValue")
        self.label.setMinimumWidth(56)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.label)
        self.slider.valueChanged.connect(self._changed)
        self.setValue(value)

    def _to_float(self, i):
        return self._min + (self._max - self._min) * i / self._steps

    def _changed(self, i):
        v = self._to_float(i)
        self.label.setText(self._fmt.format(v))
        self.valueChanged.emit(v)

    def value(self) -> float:
        return self._to_float(self.slider.value())

    def setValue(self, v: float):
        span = self._max - self._min
        i = 0 if span <= 0 else int(round((float(v) - self._min) / span * self._steps))
        i = max(0, min(self._steps, i))
        if i == self.slider.value():
            self._changed(i)
        else:
            self.slider.setValue(i)

    def setRange(self, vmin: float, vmax: float, value: float | None = None):
        """Cambia el rango; si no se da ``value`` se conserva la posición
        relativa del deslizador."""
        self._min, self._max = float(vmin), float(vmax)
        if value is None:
            self._changed(self.slider.value())
        else:
            self.setValue(value)

    def maximum(self) -> float:
        return self._max


def dspin(vmin, vmax, value, step=0.01, decimals=3) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(vmin, vmax)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    return w


def ispin(vmin, vmax, value, step=1) -> QSpinBox:
    w = QSpinBox()
    w.setRange(vmin, vmax)
    w.setSingleStep(step)
    w.setValue(value)
    return w


class PointCloudBox(QGroupBox):
    """Selector de nube de puntos: conjunto, número de puntos, ruido y
    semilla. Emite ``changed`` cuando cambia cualquier parámetro."""

    changed = Signal()

    def __init__(self, dims=(2, 3), default="Circunferencia", n=60, noise=0.03,
                 max_n=600, title="Datos", outliers: float | None = None, parent=None):
        super().__init__(title, parent)
        form = QFormLayout(self)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        self.combo = QComboBox()
        names = [nm for d in dims for nm in datasets_of_dim(d)]
        for nm in names:
            self.combo.addItem(f"{nm}  ({DATASETS[nm].dim}D)", nm)
        if default in names:
            self.combo.setCurrentIndex(names.index(default))
        self.n = ispin(3, max_n, n, 5)
        self.noise = dspin(0.0, 0.5, noise, 0.01, 3)
        self.seed = ispin(0, 9999, 0)
        self.betti = QLabel()
        self.betti.setWordWrap(True)
        self.betti.setObjectName("status")
        form.addRow("Conjunto", self.combo)
        form.addRow("Puntos", self.n)
        form.addRow("Ruido (σ)", self.noise)
        form.addRow("Semilla", self.seed)
        self.outliers = None
        if outliers is not None:
            self.outliers = ispin(0, 100, int(round(outliers * 100)), 5)
            self.outliers.setSuffix(" %")
            self.outliers.setToolTip("Puntos atípicos uniformes en la caja que contiene los datos, "
                                     "en porcentaje del número de puntos")
            form.addRow("Atípicos añadidos", self.outliers)
            self.outliers.valueChanged.connect(self.changed)
        form.addRow("Topología", self.betti)
        for w in (self.n, self.noise, self.seed):
            w.valueChanged.connect(self.changed)
        self.combo.currentIndexChanged.connect(self._update_betti)
        self.combo.currentIndexChanged.connect(self.changed)
        self._update_betti()

    def _update_betti(self):
        self.betti.setText(DATASETS[self.name()].betti)

    def name(self) -> str:
        return self.combo.currentData()

    def dim(self) -> int:
        return DATASETS[self.name()].dim

    def sample(self) -> np.ndarray:
        X = DATASETS[self.name()].sample(self.n.value(), self.noise.value(), self.seed.value())
        if self.outliers is not None and self.outliers.value() > 0:
            m = int(round(len(X) * self.outliers.value() / 100))
            rng = np.random.default_rng(10_000 + self.seed.value())
            lo, hi = X.min(axis=0), X.max(axis=0)
            pad = 0.1 * (hi - lo)
            X = np.vstack([X, rng.uniform(lo - pad, hi + pad, size=(m, X.shape[1]))])
        return X


class _JobSignals(QObject):
    done = Signal(int, object, object)    # job_id, resultado, error (texto)


class _Job(QRunnable):
    def __init__(self, job_id, fn, signals):
        super().__init__()
        self.job_id, self.fn, self.signals = job_id, fn, signals

    def run(self):
        try:
            res = self.fn()
            self.signals.done.emit(self.job_id, res, None)
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario
            msg = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self.signals.done.emit(self.job_id, None, msg)


class BasePanel(QWidget):
    """Panel con cabecera (título, subtítulo, indicador de cálculo y botón
    de teoría), una columna de controles en tarjetas, el área de gráficas y
    un cajón lateral plegable con la explicación teórica.

    ``run_job(fn, on_done)`` ejecuta ``fn`` en un hilo y llama a
    ``on_done(resultado)`` en el hilo de la interfaz; si llega un trabajo más
    reciente, los resultados antiguos se descartan.
    """

    title = ""
    subtitle = ""
    #: En los tests se ejecuta todo de forma síncrona.
    synchronous = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panelRoot")
        self._job_id = 0
        self._callbacks = {}
        self._theory_html = ""
        self.has_error = False
        self._signals = _JobSignals(self)
        self._signals.done.connect(self._job_finished)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(10)

        # -- cabecera
        header = QFrame()
        header.setObjectName("panelHeader")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(2, 0, 2, 0)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.title_label = QLabel(self.title)
        self.title_label.setObjectName("panelTitle")
        self.subtitle_label = QLabel(self.subtitle)
        self.subtitle_label.setObjectName("panelSubtitle")
        self.subtitle_label.setWordWrap(True)
        titles.addWidget(self.title_label)
        titles.addWidget(self.subtitle_label)
        hl.addLayout(titles, 1)
        self.busy = QProgressBar()
        self.busy.setObjectName("busy")
        self.busy.setRange(0, 0)
        self.busy.setTextVisible(False)
        self.busy.setFixedWidth(140)
        self.busy.setVisible(False)
        hl.addWidget(self.busy, 0, Qt.AlignVCenter)
        self.theory_button = QPushButton("  Teoría")
        self.theory_button.setCheckable(True)
        self.theory_button.setChecked(True)
        self.theory_button.setToolTip("Mostrar u ocultar la explicación teórica")
        self.theory_button.toggled.connect(self._toggle_theory)
        hl.addWidget(self.theory_button, 0, Qt.AlignVCenter)
        outer.addWidget(header)

        # -- cuerpo: controles | gráficas | teoría
        self.split = split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(10)
        outer.addWidget(split, 1)
        ctrl_widget = QWidget()
        self.controls = QVBoxLayout(ctrl_widget)
        self.controls.setContentsMargins(0, 0, 8, 0)
        self.controls.setSpacing(10)
        scroll = QScrollArea()
        scroll.setObjectName("controlsScroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(ctrl_widget)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(330)
        split.addWidget(scroll)

        self.plot_area = QWidget()
        self.plots = QVBoxLayout(self.plot_area)
        self.plots.setContentsMargins(0, 0, 0, 0)
        self.plots.setSpacing(10)
        split.addWidget(self.plot_area)

        self.theory = QTextBrowser()
        self.theory.setObjectName("theory")
        self.theory.setOpenExternalLinks(True)
        self.theory.setMinimumWidth(260)
        split.addWidget(self.theory)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([360, 1000, 340])
        split.setCollapsible(1, False)

        self.status = QLabel()
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self._apply_icons()

    # -- construcción ---------------------------------------------------------
    def add_group(self, title: str) -> QFormLayout:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        self.controls.addWidget(box)
        return form

    def add_button(self, text: str, slot, primary=True) -> QPushButton:
        b = QPushButton(text)
        if primary:
            b.setObjectName("primary")
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(slot)
        self.controls.addWidget(b)
        return b

    def add_info(self) -> QLabel:
        """Tarjeta de resultados (texto enriquecido)."""
        lab = QLabel()
        lab.setObjectName("badge")
        lab.setWordWrap(True)
        lab.setTextFormat(Qt.RichText)
        self.controls.addWidget(lab)
        return lab

    def finish_controls(self):
        self.controls.addWidget(self.status)
        self.controls.addStretch(1)

    def set_theory(self, html: str):
        self._theory_html = html
        self.theory.document().setDefaultStyleSheet(style.theory_css())
        self.theory.setHtml(html)

    def _toggle_theory(self, on: bool):
        self.theory.setVisible(on)

    def set_status(self, text: str, error: bool = False):
        self.has_error = error
        color = style.T["error"] if error else style.T["text2"]
        self.status.setStyleSheet(f"color: {color};")
        self.status.setText(text)

    # -- tema -----------------------------------------------------------------
    def _apply_icons(self):
        self.theory_button.setIcon(make_icon("book", style.T["text2"], 18))

    def apply_theme(self):
        """Se llama al cambiar de tema: adapta lienzos y textos y redibuja."""
        self._apply_icons()
        if self._theory_html:
            self.set_theory(self._theory_html)
        self.set_status(self.status.text(), self.has_error)
        for c in self.findChildren(MplCanvas):
            c.apply_theme()
        self.refresh()

    def refresh(self):
        """Redibuja las gráficas con los datos ya calculados."""
        fn = getattr(self, "redraw", None)
        if callable(fn):
            fn()

    # -- trabajos en segundo plano ---------------------------------------------
    def run_job(self, fn, on_done, busy: str = "Calculando…", on_error=None):
        self._job_id += 1
        jid = self._job_id
        self._callbacks[jid] = (on_done, on_error)
        self.set_status(busy)
        self.busy.setVisible(True)
        if self.synchronous:
            try:
                res, err = fn(), None
            except Exception as exc:  # noqa: BLE001
                res, err = None, f"{type(exc).__name__}: {exc}"
            self._job_finished(jid, res, err)
        else:
            QThreadPool.globalInstance().start(_Job(jid, fn, self._signals))

    def _job_finished(self, jid, res, err):
        on_done, on_error = self._callbacks.pop(jid, (None, None))
        if jid != self._job_id or on_done is None:
            return
        self.busy.setVisible(False)
        if err is not None:
            self.set_status(err, error=True)
            if on_error:
                on_error(err)
            return
        self.set_status("")
        on_done(res)

    def on_show(self):
        """Se llama cada vez que el panel se hace visible."""


def checkbox(text, checked=False, slot=None) -> QCheckBox:
    c = QCheckBox(text)
    c.setChecked(checked)
    if slot is not None:
        c.toggled.connect(slot)
    return c
