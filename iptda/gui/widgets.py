"""Widgets reutilizables: lienzo de matplotlib, deslizadores de valores
reales, selector de nube de puntos y la clase base de los paneles (con
ejecución de cálculos en segundo plano)."""

from __future__ import annotations

import traceback

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
                               QSlider, QSpinBox, QSplitter, QTextBrowser, QVBoxLayout, QWidget)

from ..core.datasets import DATASETS, datasets_of_dim


class MplCanvas(QWidget):
    """Figura de matplotlib incrustada, con barra de herramientas opcional
    (zoom, desplazamiento, guardar imagen)."""

    def __init__(self, parent=None, toolbar: bool = True, figsize=(5, 4)):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        if toolbar:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setIconSize(self.toolbar.iconSize() * 0.8)
            lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

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
                 max_n=600, title="Datos", parent=None):
        super().__init__(title, parent)
        form = QFormLayout(self)
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
        return DATASETS[self.name()].sample(self.n.value(), self.noise.value(), self.seed.value())


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
    """Panel con una columna izquierda (controles + texto teórico) y un
    área de gráficas a la derecha.

    ``run_job(fn, on_done)`` ejecuta ``fn`` en un hilo y llama a
    ``on_done(resultado)`` en el hilo de la interfaz; si llega un trabajo más
    reciente, los resultados antiguos se descartan.
    """

    title = ""
    #: En los tests se ejecuta todo de forma síncrona.
    synchronous = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job_id = 0
        self._callbacks = {}
        self._signals = _JobSignals(self)
        self._signals.done.connect(self._job_finished)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)

        left = QSplitter(Qt.Vertical)
        ctrl_widget = QWidget()
        self.controls = QVBoxLayout(ctrl_widget)
        self.controls.setContentsMargins(2, 2, 6, 2)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(ctrl_widget)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(380)
        scroll.setMinimumHeight(260)
        left.addWidget(scroll)
        self.theory = QTextBrowser()
        self.theory.setOpenExternalLinks(True)
        self.theory.setMinimumHeight(120)
        left.addWidget(self.theory)
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 2)
        left.setSizes([560, 340])
        split.addWidget(left)

        self.plot_area = QWidget()
        self.plots = QVBoxLayout(self.plot_area)
        self.plots.setContentsMargins(0, 0, 0, 0)
        split.addWidget(self.plot_area)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 1100])

        self.status = QLabel()
        self.status.setObjectName("status")
        self.status.setWordWrap(True)

    # -- construcción ---------------------------------------------------------
    def add_group(self, title: str) -> QFormLayout:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.controls.addWidget(box)
        return form

    def add_button(self, text: str, slot, primary=True) -> QPushButton:
        b = QPushButton(text)
        if primary:
            b.setObjectName("primary")
        b.clicked.connect(slot)
        self.controls.addWidget(b)
        return b

    def finish_controls(self):
        self.controls.addWidget(self.status)
        self.controls.addStretch(1)

    def set_theory(self, html: str):
        self.theory.setHtml(html)

    def set_status(self, text: str, error: bool = False):
        color = "#b3261e" if error else "#52514e"
        self.status.setStyleSheet(f"color: {color};")
        self.status.setText(text)

    # -- trabajos en segundo plano ---------------------------------------------
    def run_job(self, fn, on_done, busy: str = "Calculando…", on_error=None):
        self._job_id += 1
        jid = self._job_id
        self._callbacks[jid] = (on_done, on_error)
        self.set_status(busy)
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
