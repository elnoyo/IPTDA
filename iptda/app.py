"""Ventana principal de IPTDA."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QListWidget, QMainWindow, QMessageBox,
                               QStackedWidget, QWidget)

from . import __version__
from .gui import style

PANELS = [
    ("Introducción", "intro", "IntroPanel"),
    ("Complejos simpliciales", "complexes", "ComplexesPanel"),
    ("Homología", "homology", "HomologyPanel"),
    ("Persistencia", "persistence", "PersistencePanel"),
    ("Estabilidad", "stability", "StabilityPanel"),
    ("Vectorizaciones", "vectorization", "VectorizationPanel"),
    ("Series temporales", "timeseries", "TimeSeriesPanel"),
    ("Mapper", "mapper", "MapperPanel"),
    ("Persistencia biparamétrica", "multiparameter", "MultiParameterPanel"),
]


def make_panel(module: str, cls: str):
    import importlib
    mod = importlib.import_module(f"iptda.gui.panels.{module}")
    return getattr(mod, cls)()


class MainWindow(QMainWindow):
    """Barra lateral de navegación y una pila de paneles. Cada panel se crea
    la primera vez que se visita."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IPTDA · Análisis Topológico de Datos interactivo")
        self.resize(1500, 900)
        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(230)
        for name, _, _ in PANELS:
            self.nav.addItem(name)
        self.stack = QStackedWidget()
        lay.addWidget(self.nav)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.panels: dict[int, QWidget] = {}
        self._placeholders = []
        for _ in PANELS:
            w = QWidget()
            self._placeholders.append(w)
            self.stack.addWidget(w)
        self.nav.currentRowChanged.connect(self.show_panel)
        self._menus()
        self.nav.setCurrentRow(0)

    def _menus(self):
        m = self.menuBar().addMenu("&Archivo")
        quit_ = QAction("&Salir", self)
        quit_.setShortcut("Ctrl+Q")
        quit_.triggered.connect(self.close)
        m.addAction(quit_)
        v = self.menuBar().addMenu("&Paneles")
        for i, (name, _, _) in enumerate(PANELS):
            act = QAction(name, self)
            act.setShortcut(f"Ctrl+{i + 1}")
            act.triggered.connect(lambda _=False, i=i: self.nav.setCurrentRow(i))
            v.addAction(act)
        h = self.menuBar().addMenu("A&yuda")
        about = QAction("&Acerca de IPTDA", self)
        about.triggered.connect(self._about)
        h.addAction(about)

    def panel(self, i: int) -> QWidget:
        if i not in self.panels:
            name, module, cls = PANELS[i]
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                w = make_panel(module, cls)
            finally:
                QApplication.restoreOverrideCursor()
            old = self._placeholders[i]
            self.stack.insertWidget(i, w)
            self.stack.removeWidget(old)
            old.deleteLater()
            self.panels[i] = w
        return self.panels[i]

    def show_panel(self, i: int):
        if i < 0:
            return
        w = self.panel(i)
        self.stack.setCurrentWidget(w)
        if hasattr(w, "on_show"):
            w.on_show()
        self.statusBar().showMessage(PANELS[i][0])

    def _about(self):
        QMessageBox.about(self, "Acerca de IPTDA",
                          f"<b>IPTDA {__version__}</b><br>Aprender Análisis Topológico de Datos "
                          "de forma gráfica e interactiva.<br><br>Python · PySide6 · Matplotlib · "
                          "NumPy · SciPy · NetworkX<br>Licencia GPL-3.0")


def main(argv=None):
    import matplotlib
    matplotlib.use("QtAgg")
    style.apply_matplotlib_style()
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("IPTDA")
    app.setStyle("Fusion")
    app.setStyleSheet(style.QSS)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
