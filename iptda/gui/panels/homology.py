"""Panel: homología simplicial (matrices borde, rangos, números de Betti y
generadores)."""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QComboBox, QLabel, QPlainTextEdit, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ...core.complexes_library import LIBRARY, LibraryComplex
from ...core.homology import FIELDS, boundary_matrix, homology, parse_complex, simplices_by_dim
from .. import style
from ..plotting import setup_axes_for
from ..widgets import BasePanel, MplCanvas

CUSTOM = "Personalizado (escribe los símplices)"

THEORY = """
<h3>Homología simplicial</h3>
<p>Los <b>k-cadenas</b> C<sub>k</sub> son combinaciones formales de k-símplices con
coeficientes en un cuerpo 𝔽. El <b>operador borde</b></p>
<p align="center">∂[v<sub>0</sub>,…,v<sub>k</sub>] = Σ<sub>i</sub> (−1)<sup>i</sup>
[v<sub>0</sub>,…,v̂<sub>i</sub>,…,v<sub>k</sub>]</p>
<p>cumple ∂∘∂ = 0, por lo que los <b>bordes</b> B<sub>k</sub> = im ∂<sub>k+1</sub> están
contenidos en los <b>ciclos</b> Z<sub>k</sub> = ker ∂<sub>k</sub>. La homología es el
cociente H<sub>k</sub> = Z<sub>k</sub> / B<sub>k</sub> y su dimensión es el
<b>número de Betti</b></p>
<p align="center">β<sub>k</sub> = dim Z<sub>k</sub> − dim B<sub>k</sub> =
c<sub>k</sub> − rango ∂<sub>k</sub> − rango ∂<sub>k+1</sub>.</p>
<p>β<sub>0</sub> cuenta componentes conexas, β<sub>1</sub> «agujeros» (ciclos que no
bordean), β<sub>2</sub> cavidades…</p>
<p><b>El cuerpo importa:</b> en el plano proyectivo o la botella de Klein, la homología
entera tiene torsión Z/2, que solo se detecta con coeficientes en Z/2. Compara Z/2 con Q.</p>
<p><b>Generadores:</b> se obtienen (con coeficientes en Z/2) reduciendo la matriz borde;
cada clase esencial da un ciclo representante, que se resalta en rojo. Un
representante no es único: cualquier ciclo homólogo sirve.</p>
<p><b>Polígonos fundamentales:</b> los vértices con la misma etiqueta son el mismo
vértice (los lados del polígono se identifican).</p>
<p><b>Complejo personalizado:</b> escribe símplices maximales separados por «;», por
ejemplo <code>0 1 2; 2 3; 3 4; 4 2</code>. Se añade automáticamente todas sus caras.</p>
"""


class HomologyPanel(BasePanel):
    title = "Homología"
    subtitle = "Matrices borde, rangos y números de Betti sobre distintos cuerpos de coeficientes"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entry: LibraryComplex | None = None
        self.report = None
        self.by_dim = None

        f = self.add_group("Complejo")
        self.source = QComboBox()
        self.source.addItems(list(LIBRARY.keys()) + [CUSTOM])
        self.source.setCurrentText("Toro (polígono fundamental)")
        self.text = QPlainTextEdit("0 1 2; 0 2 3; 0 3 1; 1 2 3")
        self.text.setMaximumHeight(70)
        f.addRow("Ejemplo", self.source)
        f.addRow("Símplices", self.text)
        self.source.currentIndexChanged.connect(self.load)
        self.desc = QLabel()
        self.desc.setWordWrap(True)
        self.desc.setObjectName("status")
        f.addRow(self.desc)
        self.add_button("Construir complejo personalizado", self._custom, primary=False)

        g = self.add_group("Cálculo")
        self.field = QComboBox()
        self.field.addItems(list(FIELDS.keys()))
        self.field.currentIndexChanged.connect(self.compute)
        self.matrix_k = QComboBox()
        self.matrix_k.currentIndexChanged.connect(self._show_matrix)
        self.gen = QComboBox()
        self.gen.currentIndexChanged.connect(self.redraw)
        g.addRow("Coeficientes", self.field)
        g.addRow("Matriz borde", self.matrix_k)
        g.addRow("Generador", self.gen)
        self.finish_controls()
        self.set_theory(THEORY)

        split = QSplitter(Qt.Horizontal)
        self.canvas = MplCanvas()
        split.addWidget(self.canvas)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.summary = QLabel()
        self.summary.setObjectName("badge")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        rl.addWidget(self.summary)
        self.table = QTableWidget()
        rl.addWidget(self.table, 1)
        split.addWidget(right)
        split.setSizes([650, 520])
        self.plots.addWidget(split)
        self.load()

    def refresh(self):
        self.compute()        # recalcula (barato) para recolorear tabla y resumen

    # -- carga --------------------------------------------------------------------
    def load(self):
        name = self.source.currentText()
        if name == CUSTOM:
            self._custom()
            return
        self.entry = LIBRARY[name]
        self.desc.setText(self.entry.description)
        self.compute()

    def _custom(self):
        try:
            F = parse_complex(self.text.toPlainText())
        except ValueError as exc:
            self.set_status(str(exc), error=True)
            return
        maximal = [s for s in F.simplices if not any(
            len(t) == len(s) + 1 and set(s) <= set(t) for t in F.simplices)]
        nv = max(s[0] for s in F.simplices if len(s) == 1) + 1
        G = nx.Graph()
        G.add_nodes_from(v for (v,) in (s for s in F.simplices if len(s) == 1))
        G.add_edges_from(s for s in F.simplices if len(s) == 2)
        pos = nx.kamada_kawai_layout(G) if G.number_of_edges() else nx.circular_layout(G)
        P = np.full((nv, 2), np.nan)
        for v, p in pos.items():
            P[v] = p
        self.entry = LibraryComplex("Personalizado", P, maximal, np.arange(nv),
                                    "Complejo definido por el usuario.")
        if self.source.currentText() != CUSTOM:
            self.source.blockSignals(True)
            self.source.setCurrentText(CUSTOM)
            self.source.blockSignals(False)
        self.desc.setText(self.entry.description)
        self.compute()

    def compute(self):
        if self.entry is None:
            return
        F = self.entry.filtration()
        field = self.field.currentText()
        try:
            self.report = homology(F, field, with_generators=True)
        except Exception as exc:  # noqa: BLE001
            self.set_status(f"Error: {exc}", error=True)
            return
        self.set_status("")
        self.by_dim = simplices_by_dim(F)
        self.matrix_k.blockSignals(True)
        self.matrix_k.clear()
        for k in range(1, len(self.by_dim)):
            self.matrix_k.addItem(f"∂{k} : C{k} → C{k-1}", k)
        self.matrix_k.blockSignals(False)
        self.gen.blockSignals(True)
        self.gen.clear()
        self.gen.addItem("(ninguno)", None)
        if field == "Z/2":
            for k, gens in enumerate(self.report.generators):
                for i in range(len(gens)):
                    self.gen.addItem(f"H{k} · generador {i + 1}", (k, i))
        else:
            self.gen.addItem("(los generadores se muestran con Z/2)", None)
        self.gen.blockSignals(False)
        self._summary()
        self._show_matrix()
        self.redraw()

    # -- resumen y matriz --------------------------------------------------------
    def _summary(self):
        R = self.report
        rows = []
        top = len(R.counts) - 1
        for k in range(top + 1):
            rk = R.ranks[k] if k < len(R.ranks) else 0
            rk1 = R.ranks[k + 1] if k + 1 < len(R.ranks) else 0
            exp = ""
            if self.entry is not None and R.field in self.entry.expected:
                e = self.entry.expected[R.field]
                if k < len(e):
                    exp = "✓" if e[k] == R.betti[k] else f"✗ (esperado {e[k]})"
            rows.append(f"<tr><td>{k}</td><td>{R.counts[k]}</td><td>{rk}</td>"
                        f"<td>{R.counts[k] - rk}</td><td>{rk1}</td><td><b>{R.betti[k]}</b> {exp}</td></tr>")
        chi_b = sum((-1) ** k * b for k, b in enumerate(R.betti))
        html = (f"<b>Homología con coeficientes en {R.field}</b>"
                f"<table cellspacing='0' cellpadding='4' border='1' "
                f"style='border-color:{style.T['border']}'>"
                "<tr><th>k</th><th>c<sub>k</sub></th><th>rango ∂<sub>k</sub></th>"
                "<th>dim Z<sub>k</sub></th><th>dim B<sub>k</sub></th><th>β<sub>k</sub></th></tr>"
                + "".join(rows) + "</table>"
                f"χ = Σ(−1)<sup>k</sup>c<sub>k</sub> = {R.euler} &nbsp; = &nbsp; "
                f"Σ(−1)<sup>k</sup>β<sub>k</sub> = {chi_b}"
                f"<br>∂<sub>k</sub>∘∂<sub>k+1</sub> = 0: {'✓' if R.dd_zero else '✗'}")
        self.summary.setText(html)

    def _show_matrix(self):
        k = self.matrix_k.currentData()
        self.table.clear()
        if k is None or self.by_dim is None:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        rows, cols = self.by_dim[k - 1], self.by_dim[k]
        M = boundary_matrix(rows, cols)
        z2 = self.field.currentText() == "Z/2"
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(["[" + ",".join(map(str, s)) + "]" for s in cols])
        self.table.setVerticalHeaderLabels(["[" + ",".join(map(str, s)) + "]" for s in rows])
        pos = QBrush(QColor(style.T["accent_soft"]))
        neg = QBrush(QColor(style.dim_color(1)).lighter(170) if style.current == "light"
                     else QColor(style.dim_color(1)).darker(260))
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = int(M[i, j])
                if v:
                    txt = "1" if z2 else str(v)
                    it = QTableWidgetItem(txt)
                    it.setBackground(pos if (v > 0 or z2) else neg)
                else:
                    it = QTableWidgetItem("")
                it.setTextAlignment(Qt.AlignCenter)
                it.setFlags(Qt.ItemIsEnabled)
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

    # -- dibujo ---------------------------------------------------------------------
    def redraw(self, *_):
        E = self.entry
        if E is None:
            return
        fig = self.canvas.figure
        fig.clear()
        P = E.points
        d = P.shape[1]
        ax = fig.add_subplot(111, projection="3d" if d == 3 else None)
        disp = set()
        for s in E.display_maximal:
            for k in range(1, len(s) + 1):
                disp.update(combinations(sorted(s), k))
        tris = [s for s in disp if len(s) == 3]
        edges = [s for s in disp if len(s) == 2]
        verts = [s[0] for s in disp if len(s) == 1]

        hl = []
        sel = self.gen.currentData()
        hl_dim = None
        if sel is not None and self.report is not None:
            k, i = sel
            hl_dim = k
            for sim in self.report.generators[k][i]:
                hl.extend(E.display_occurrences(sim))

        if d == 2:
            if tris:
                ax.add_collection(PolyCollection([P[list(t)] for t in tris], facecolor=style.TRIANGLE_COLOR,
                                                 alpha=0.22, edgecolor="none"))
            if edges:
                ax.add_collection(LineCollection([P[list(e)] for e in edges], colors=style.EDGE_COLOR,
                                                 linewidths=1.0))
            if hl_dim == 2:
                ax.add_collection(PolyCollection([P[list(t)] for t in hl], facecolor=style.HIGHLIGHT,
                                                 alpha=0.45, edgecolor=style.HIGHLIGHT))
            elif hl_dim == 1:
                ax.add_collection(LineCollection([P[list(e)] for e in hl], colors=style.HIGHLIGHT,
                                                 linewidths=3.5, zorder=4))
            ax.scatter(P[verts, 0], P[verts, 1], s=40, color=style.POINT_COLOR, zorder=5,
                       edgecolors="white")
            if hl_dim == 0:
                vv = [s[0] for s in hl]
                ax.scatter(P[vv, 0], P[vv, 1], s=140, color=style.HIGHLIGHT, zorder=6)
            for v in verts:
                ax.annotate(str(E.labels[v]), P[v], textcoords="offset points", xytext=(5, 5),
                            fontsize=10, color=style.TEXT, fontweight="bold")
            if E.identified:
                ax.set_title("Polígono fundamental\n(vértices con la misma etiqueta = mismo vértice)")
            ok = ~np.isnan(P[:, 0])
            setup_axes_for(ax, P[ok], pad=0.25)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        else:
            if tris:
                ax.add_collection3d(Poly3DCollection([P[list(t)] for t in tris], alpha=0.18,
                                                     facecolor=style.TRIANGLE_COLOR, edgecolor="none"))
            if edges:
                ax.add_collection3d(Line3DCollection([P[list(e)] for e in edges], colors=style.EDGE_COLOR,
                                                     linewidths=0.9))
            if hl_dim == 2:
                ax.add_collection3d(Poly3DCollection([P[list(t)] for t in hl], alpha=0.5,
                                                     facecolor=style.HIGHLIGHT, edgecolor=style.HIGHLIGHT))
            elif hl_dim == 1:
                ax.add_collection3d(Line3DCollection([P[list(e)] for e in hl], colors=style.HIGHLIGHT,
                                                     linewidths=3.5))
            ax.scatter(P[verts, 0], P[verts, 1], P[verts, 2], s=25, color=style.POINT_COLOR)
            if hl_dim == 0:
                vv = [s[0] for s in hl]
                ax.scatter(P[vv, 0], P[vv, 1], P[vv, 2], s=120, color=style.HIGHLIGHT)
            for v in verts:
                ax.text(*P[v], f" {E.labels[v]}", fontsize=9, color=style.TEXT)
            setup_axes_for(ax, P)
        if not E.identified:
            ax.set_title(E.name)
        self.canvas.draw()
