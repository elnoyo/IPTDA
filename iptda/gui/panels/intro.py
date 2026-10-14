"""Panel de bienvenida: presentación, tarjetas de acceso a cada panel,
convenciones y referencias."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from .. import style
from ..icons import make_icon

#: (índice del panel, icono, título, descripción)
CARDS = [
    (1, "complexes", "Complejos simpliciales",
     "Vietoris–Rips, Čech, alfa y testigos sobre nubes 2D y 3D. Mueve la escala r, compara "
     "métodos y añade puntos con el ratón."),
    (2, "homology", "Homología",
     "Matrices borde, rangos y números de Betti de esferas, toros, la botella de Klein o el plano "
     "proyectivo, sobre Z/2, Z/3, Z/5 y Q."),
    (3, "persistence", "Persistencia",
     "Filtraciones, códigos de barras y diagramas. Pulsa una barra para ver su ciclo "
     "representante; elige el cuerpo de coeficientes."),
    (4, "stability", "Estabilidad",
     "Distancias de cuello de botella y Wasserstein, emparejamiento óptimo y el teorema de "
     "estabilidad frente a perturbaciones y atípicos."),
    (5, "vectorization", "Vectorizaciones",
     "Curvas de Betti, paisajes e imágenes de persistencia y entropía: del diagrama a vectores."),
    (6, "timeseries", "Series temporales",
     "Inmersión de Takens para detectar periodicidad y cuasiperiodicidad, y filtraciones por "
     "subnivel de una señal."),
    (7, "mapper", "Mapper",
     "Lentes de una o dos componentes, recubrimientos solapados, clustering y el nervio "
     "resultante con sus números de Betti."),
    (8, "multiparameter", "Persistencia biparamétrica",
     "Escala × densidad en 2D y 3D: función de Hilbert, códigos de barras fibrados e invariante de "
     "rango."),
]

CONVENTIONS = """
<ul>
<li>El parámetro <b>r</b> es el <b>radio</b> de las bolas: en Rips un símplice entra en
r = diámetro/2; en Čech, en el radio de su bola mínima envolvente; en alfa, en su radio alfa.
Así Čech(r) ⊆ Rips(r) ⊆ Čech(√2·r).</li>
<li>La persistencia se calcula por defecto con coeficientes en <b>Z/2</b>; en el panel de
persistencia puede elegirse Z/3, Z/5 o Z/7, y en el de homología también Q.</li>
<li>Con símplices hasta dimensión D solo se informa de la homología hasta D − 1.</li>
<li>Colores fijos por dimensión: <b>H0</b> azul, <b>H1</b> naranja, <b>H2</b> verde; lo
seleccionado, en rojo. Ctrl+T cambia entre tema claro y oscuro.</li>
</ul>
"""

REFERENCES = """
<ul>
<li>H. Edelsbrunner, J. Harer. <i>Computational Topology: An Introduction</i>. AMS, 2010.</li>
<li>G. Carlsson. Topology and data. <i>Bull. AMS</i> 46 (2009).</li>
<li>S. Oudot. <i>Persistence Theory: From Quiver Representations to Data Analysis</i>. AMS, 2015.</li>
<li>G. Singh, F. Mémoli, G. Carlsson. Topological methods for the analysis of high dimensional
data sets and 3D object recognition. <i>SPBG</i> 2007.</li>
<li>J. Perea, J. Harer. Sliding windows and persistence. <i>Found. Comput. Math.</i> 15 (2015).</li>
<li>M. Lesnick, M. Wright. Interactive visualization of 2-D persistence modules (RIVET). 2015.</li>
<li>M. Botnan, M. Lesnick. An introduction to multiparameter persistence. 2022.</li>
</ul>
"""


class _Card(QFrame):
    def __init__(self, idx, icon, title, text, on_open):
        super().__init__()
        self.setObjectName("card")
        self.icon_name = icon
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(8)
        top = QHBoxLayout()
        self.icon = QLabel()
        self.icon.setFixedSize(40, 40)
        self.icon.setAlignment(Qt.AlignCenter)
        top.addWidget(self.icon)
        t = QLabel(title)
        t.setStyleSheet("font-size: 12.5pt; font-weight: 700;")
        top.addWidget(t, 1)
        v.addLayout(top)
        d = QLabel(text)
        d.setWordWrap(True)
        d.setObjectName("panelSubtitle")
        v.addWidget(d, 1)
        b = QPushButton("Abrir  →")
        b.setObjectName("link")
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(lambda: on_open(idx))
        v.addWidget(b, 0, Qt.AlignLeft)
        self.apply_theme()

    def apply_theme(self):
        self.icon.setStyleSheet(f"background: {style.T['accent_soft']}; border-radius: 10px;")
        self.icon.setPixmap(make_icon(self.icon_name, style.T["accent"], 24).pixmap(QSize(24, 24)))


class IntroPanel(QWidget):
    title = "Introducción"
    navigate = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panelRoot")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setObjectName("controlsScroll")
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        body = QWidget()
        body.setObjectName("panelRoot")
        scroll.setWidget(body)
        v = QVBoxLayout(body)
        v.setContentsMargins(36, 30, 36, 30)
        v.setSpacing(18)

        self.hero = QFrame()
        self.hero.setObjectName("hero")
        h = QVBoxLayout(self.hero)
        h.setContentsMargins(30, 26, 30, 26)
        h.setSpacing(8)
        kicker = QLabel("APRENDE TOPOLOGÍA DE DATOS JUGANDO CON ELLA")
        kicker.setStyleSheet("color: rgba(255,255,255,0.75); font-size: 8.5pt; font-weight: 700;"
                             "letter-spacing: 1.5px;")
        title = QLabel("Análisis Topológico de Datos, paso a paso")
        title.setStyleSheet("color: white; font-size: 24pt; font-weight: 800;")
        text = QLabel("El TDA estudia la <i>forma</i> de los datos —componentes, agujeros, cavidades, "
                      "ramificaciones— con herramientas de topología algebraica, de manera robusta "
                      "frente al ruido. Cada panel combina controles interactivos, gráficas enlazadas "
                      "y una explicación teórica.")
        text.setWordWrap(True)
        text.setStyleSheet("color: rgba(255,255,255,0.92); font-size: 11.5pt;")
        start = QPushButton("Empezar por los complejos simpliciales  →")
        start.setCursor(Qt.PointingHandCursor)
        start.setStyleSheet("QPushButton { background: white; color: #174a8b; border: none; "
                            "border-radius: 8px; padding: 9px 16px; font-weight: 700; }"
                            "QPushButton:hover { background: #e6effb; }")
        start.clicked.connect(lambda: self.navigate.emit(1))
        for w in (kicker, title, text):
            h.addWidget(w)
        h.addSpacing(6)
        h.addWidget(start, 0, Qt.AlignLeft)
        v.addWidget(self.hero)

        sec = QLabel("Recorrido")
        sec.setObjectName("panelTitle")
        v.addWidget(sec)
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        self.cards = []
        for k, (idx, icon, t, d) in enumerate(CARDS):
            c = _Card(idx, icon, t, d, self.navigate.emit)
            self.cards.append(c)
            grid.addWidget(c, k // 3, k % 3)
        for col in range(3):
            grid.setColumnStretch(col, 1)
        v.addLayout(grid)

        row = QHBoxLayout()
        row.setSpacing(16)
        self.notes = []
        for head, html in (("Convenciones", CONVENTIONS), ("Referencias", REFERENCES)):
            card = QFrame()
            card.setObjectName("card")
            cv = QVBoxLayout(card)
            cv.setContentsMargins(18, 14, 18, 14)
            lab = QLabel(head)
            lab.setStyleSheet("font-size: 12.5pt; font-weight: 700;")
            body_l = QLabel(html)
            body_l.setWordWrap(True)
            body_l.setTextFormat(Qt.RichText)
            body_l.setObjectName("panelSubtitle")
            cv.addWidget(lab)
            cv.addWidget(body_l)
            row.addWidget(card, 1)
        v.addLayout(row)
        v.addStretch(1)
        self.apply_theme()

    def apply_theme(self):
        a, b = ("#2a78d6", "#6250d6") if style.current == "light" else ("#1f4f8c", "#4b3fa6")
        self.hero.setStyleSheet(
            "QFrame#hero { border-radius: 16px; background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
            f"stop:0 {a}, stop:1 {b}); }}")
        for c in self.cards:
            c.apply_theme()

    def on_show(self):
        pass
