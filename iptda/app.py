"""Ventana principal de IPTDA."""

from __future__ import annotations

import sys

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QAction, QColor, QFont
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget)

from . import __version__
from .gui import style
from .gui.icons import make_icon

#: (nombre, módulo, clase, icono, sección)
PANELS = [
    ("Introducción", "intro", "IntroPanel", "intro", None),
    ("Complejos simpliciales", "complexes", "ComplexesPanel", "complexes", "Fundamentos"),
    ("Homología", "homology", "HomologyPanel", "homology", "Fundamentos"),
    ("Persistencia", "persistence", "PersistencePanel", "persistence", "Persistencia"),
    ("Estabilidad", "stability", "StabilityPanel", "stability", "Persistencia"),
    ("Vectorizaciones", "vectorization", "VectorizationPanel", "vectorization", "Persistencia"),
    ("Series temporales", "timeseries", "TimeSeriesPanel", "timeseries", "Aplicaciones"),
    ("Mapper", "mapper", "MapperPanel", "mapper", "Aplicaciones"),
    ("Persistencia biparamétrica", "multiparameter", "MultiParameterPanel", "multiparameter",
     "Aplicaciones"),
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
        self.resize(1600, 950)
        self.settings = QSettings("IPTDA", "IPTDA")
        central = QWidget()
        central.setObjectName("central")
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._build_sidebar())
        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.panels: dict[int, QWidget] = {}
        self._placeholders = []
        for _ in PANELS:
            w = QWidget()
            self._placeholders.append(w)
            self.stack.addWidget(w)
        self.nav.currentRowChanged.connect(self._row_changed)
        self._menus()
        self._refresh_icons()
        self.show_panel(0)

    # -- barra lateral --------------------------------------------------------
    def _build_sidebar(self) -> QFrame:
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(270)
        v = QVBoxLayout(side)
        v.setContentsMargins(10, 20, 10, 16)
        v.setSpacing(4)
        brand = QLabel("IPTDA")
        brand.setObjectName("brand")
        sub = QLabel("Análisis Topológico de Datos\ninteractivo")
        sub.setObjectName("brandSub")
        v.addWidget(brand)
        v.addWidget(sub)
        v.addSpacing(14)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setIconSize(QSize(20, 20))
        self.nav.setFocusPolicy(Qt.NoFocus)
        self._row_of_panel: dict[int, int] = {}
        self._panel_of_row: dict[int, int] = {}
        section = None
        for i, (name, _, _, _, sec) in enumerate(PANELS):
            if sec != section and sec is not None:
                head = QListWidgetItem(sec.upper())
                head.setFlags(Qt.NoItemFlags)
                f = QFont()
                f.setPointSizeF(7.8)
                f.setBold(True)
                f.setLetterSpacing(QFont.AbsoluteSpacing, 1.2)
                head.setFont(f)
                head.setSizeHint(QSize(0, 34))
                self.nav.addItem(head)
                section = sec
            it = QListWidgetItem(name)
            it.setSizeHint(QSize(0, 38))
            self._row_of_panel[i] = self.nav.count()
            self._panel_of_row[self.nav.count()] = i
            self.nav.addItem(it)
        v.addWidget(self.nav, 1)
        self.theme_button = QPushButton()
        self.theme_button.setObjectName("sideButton")
        self.theme_button.setCursor(Qt.PointingHandCursor)
        self.theme_button.clicked.connect(self.toggle_theme)
        v.addWidget(self.theme_button)
        about = QPushButton("  Acerca de")
        about.setObjectName("sideButton")
        about.setCursor(Qt.PointingHandCursor)
        about.clicked.connect(self._about)
        self.about_button = about
        v.addWidget(about)
        return side

    def _refresh_icons(self):
        dark = style.current == "dark"
        for i, (_, _, _, icon, _) in enumerate(PANELS):
            self.nav.item(self._row_of_panel[i]).setIcon(make_icon(icon, style.T["sidebar_text"], 20))
        for r in range(self.nav.count()):
            if r not in self._panel_of_row:           # cabeceras de sección
                self.nav.item(r).setForeground(QColor(style.T["sidebar_muted"]))
        self.theme_button.setText("  Modo claro" if dark else "  Modo oscuro")
        self.theme_button.setIcon(make_icon("sun" if dark else "moon", style.T["sidebar_text"], 18))
        self.about_button.setIcon(make_icon("info", style.T["sidebar_text"], 18))

    # -- navegación ------------------------------------------------------------
    def _row_changed(self, row: int):
        if row in self._panel_of_row:
            self.show_panel(self._panel_of_row[row])

    def panel(self, i: int) -> QWidget:
        if i not in self.panels:
            _, module, cls, _, _ = PANELS[i]
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                w = make_panel(module, cls)
            finally:
                QApplication.restoreOverrideCursor()
            if hasattr(w, "navigate"):
                w.navigate.connect(self.show_panel)
            old = self._placeholders[i]
            self.stack.insertWidget(i, w)
            self.stack.removeWidget(old)
            old.deleteLater()
            self.panels[i] = w
        return self.panels[i]

    def show_panel(self, i: int):
        if i < 0 or i >= len(PANELS):
            return
        row = self._row_of_panel[i]
        if self.nav.currentRow() != row:
            self.nav.blockSignals(True)
            self.nav.setCurrentRow(row)
            self.nav.blockSignals(False)
        w = self.panel(i)
        self.stack.setCurrentWidget(w)
        if hasattr(w, "on_show"):
            w.on_show()
        self.statusBar().showMessage(PANELS[i][0])

    def current_panel_index(self) -> int:
        return self._panel_of_row.get(self.nav.currentRow(), 0)

    # -- tema --------------------------------------------------------------------
    def toggle_theme(self):
        self.set_theme("light" if style.current == "dark" else "dark")

    def set_theme(self, name: str):
        apply_theme(QApplication.instance(), name)
        self.settings.setValue("theme", style.current)
        self._refresh_icons()
        for w in self.panels.values():
            if hasattr(w, "apply_theme"):
                w.apply_theme()

    # -- menús ---------------------------------------------------------------------
    def _menus(self):
        m = self.menuBar().addMenu("&Archivo")
        quit_ = QAction("&Salir", self)
        quit_.setShortcut("Ctrl+Q")
        quit_.triggered.connect(self.close)
        m.addAction(quit_)
        v = self.menuBar().addMenu("&Ver")
        theme = QAction("Cambiar tema claro / oscuro", self)
        theme.setShortcut("Ctrl+T")
        theme.triggered.connect(self.toggle_theme)
        v.addAction(theme)
        v.addSeparator()
        for i, (name, *_rest) in enumerate(PANELS):
            act = QAction(name, self)
            act.setShortcut(f"Ctrl+{i + 1}")
            act.triggered.connect(lambda _=False, i=i: self.show_panel(i))
            v.addAction(act)
        h = self.menuBar().addMenu("A&yuda")
        about = QAction("&Acerca de IPTDA", self)
        about.triggered.connect(self._about)
        h.addAction(about)

    def _about(self):
        QMessageBox.about(self, "Acerca de IPTDA",
                          f"<b>IPTDA {__version__}</b><br>Aprender Análisis Topológico de Datos "
                          "de forma gráfica e interactiva.<br><br>Python · PySide6 · Matplotlib · "
                          "NumPy · SciPy · NetworkX<br>Licencia GPL-3.0")


def apply_theme(app: QApplication, name: str):
    style.set_theme(name)
    app.setPalette(style.palette())
    app.setStyleSheet(style.qss())


def main(argv=None):
    import matplotlib
    matplotlib.use("QtAgg")
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("IPTDA")
    app.setStyle("Fusion")
    theme = QSettings("IPTDA", "IPTDA").value("theme", "light")
    apply_theme(app, theme if theme in style.THEMES else "light")
    win = MainWindow()
    win.showMaximized()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
