"""Panel: complejos simpliciales a partir de nubes de puntos 2D y 3D."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel
from scipy.spatial import Voronoi, voronoi_plot_2d

from ...core.filtration import METHODS, build_filtration
from ...core.geometry import pairwise_distances
from ...core.persistence import compute_persistence
from .. import style
from ..plotting import draw_complex, setup_axes_for
from ..widgets import BasePanel, FloatSlider, MplCanvas, PointCloudBox, checkbox, dspin, ispin

THEORY = """
<h3>Complejos simpliciales desde nubes de puntos</h3>
<p>Un <b>complejo simplicial</b> es una colección de símplices (vértices, aristas,
triángulos, tetraedros…) cerrada por caras. A partir de una nube de puntos
<i>X</i> y una escala <i>r</i> se construyen varios:</p>
<ul>
<li><b>Vietoris–Rips</b> <i>VR(X, r)</i>: un símplice por cada conjunto de puntos
a distancia ≤ 2r <i>dos a dos</i>. Es un complejo «bandera»: queda determinado por
su grafo. Barato de definir, pero puede tener muchos símplices.</li>
<li><b>Čech</b> <i>Č(X, r)</i>: un símplice por cada conjunto de puntos cuyas bolas
de radio r tienen un punto <i>común</i> (bola mínima envolvente de radio ≤ r). Por
el <b>teorema del nervio</b>, Č(X, r) tiene el tipo de homotopía de la unión de las
bolas.</li>
<li><b>Alfa</b>: nervio de las bolas recortadas por las celdas de Voronoi. Es un
subcomplejo de la triangulación de Delaunay, tiene el mismo tipo de homotopía que
la unión de bolas (como Čech) y muchos menos símplices.</li>
<li><b>Testigos (perezoso)</b>: se eligen <i>puntos de referencia</i> (por maxmin) y el resto
de puntos actúan como <i>testigos</i> de las aristas. Útil con muchos puntos.</li>
</ul>
<p><b>Convención de escala:</b> en toda la aplicación <i>r</i> es el <b>radio</b> de las
bolas. Se cumple <i>Č(X, r) ⊆ VR(X, r) ⊆ Č(X, √2·r)</i>. Compara Rips y Čech con el
mismo <i>r</i>: aparecen triángulos en Rips que no están en Čech.</p>
<p><b>Característica de Euler:</b> χ = Σ (−1)<sup>k</sup> c<sub>k</sub> = Σ (−1)<sup>k</sup> β<sub>k</sub>,
con c<sub>k</sub> el número de k-símplices y β<sub>k</sub> los números de Betti.</p>
<p><b>Prueba:</b> en 2D activa «Editar puntos con el ratón» (clic izquierdo añade un
punto, clic derecho elimina el más cercano).</p>
"""


class ComplexesPanel(BasePanel):
    title = "Complejos simpliciales"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.X = None
        self.custom = False
        self.results = {}     # vista -> (filtración, persistencia, landmarks)

        self.data = PointCloudBox(default="Circunferencia", n=30, noise=0.05, max_n=400)
        self.controls.addWidget(self.data)
        self.data.changed.connect(self.regenerate)

        f = self.add_group("Construcción")
        self.method_a = QComboBox()
        self.method_a.addItems(METHODS)
        self.method_b = QComboBox()
        self.method_b.addItems(["(ninguno)"] + METHODS)
        self.method_b.setCurrentText("Čech")
        self.max_dim = ispin(1, 3, 2)
        self.rmax = dspin(0.01, 100, 1.0, 0.05)
        self.auto_rmax = checkbox("automático (½ del diámetro)", True)
        self.landmarks = ispin(3, 200, 15)
        self.nu = ispin(0, 2, 1)
        f.addRow("Método (izquierda)", self.method_a)
        f.addRow("Comparar con", self.method_b)
        f.addRow("Dimensión máx. de símplices", self.max_dim)
        f.addRow("Radio máximo", self.rmax)
        f.addRow("", self.auto_rmax)
        f.addRow("Puntos de referencia", self.landmarks)
        f.addRow("ν (testigos)", self.nu)
        for w in (self.method_a, self.method_b):
            w.currentIndexChanged.connect(self.rebuild)
        for w in (self.max_dim, self.rmax, self.landmarks, self.nu):
            w.valueChanged.connect(self.rebuild)
        self.auto_rmax.toggled.connect(self.rebuild)

        g = self.add_group("Escala y vista")
        self.r = FloatSlider(0, 1, 0.25)
        self.r.valueChanged.connect(self.redraw)
        g.addRow("r", self.r)
        self.show_balls = checkbox("Mostrar bolas de radio r (2D)", True, self.redraw)
        self.show_voronoi = checkbox("Mostrar diagrama de Voronoi (2D)", False, self.redraw)
        self.edit = checkbox("Editar puntos con el ratón (2D)", False)
        self.show_labels = checkbox("Numerar los vértices", False, self.redraw)
        for w in (self.show_balls, self.show_voronoi, self.edit, self.show_labels):
            g.addRow(w)
        self.add_button("Regenerar datos", self.regenerate, primary=False)
        self.finish_controls()
        self.set_theory(THEORY)

        self.canvas = MplCanvas()
        self.plots.addWidget(self.canvas, 1)
        info = QHBoxLayout()
        self.info_a = QLabel()
        self.info_b = QLabel()
        for lab in (self.info_a, self.info_b):
            lab.setObjectName("badge")
            lab.setWordWrap(True)
            info.addWidget(lab, 1)
        self.plots.addLayout(info)
        self.canvas.canvas.mpl_connect("button_press_event", self._on_click)
        self.regenerate()

    # -- datos ------------------------------------------------------------------
    def regenerate(self):
        self.custom = False
        self.X = self.data.sample()
        self.rebuild()

    def _on_click(self, ev):
        if not self.edit.isChecked() or self.X is None or self.X.shape[1] != 2:
            return
        if ev.inaxes is None or ev.xdata is None or self.canvas.toolbar.mode:
            return
        p = np.array([ev.xdata, ev.ydata])
        if ev.button == 1:
            self.X = np.vstack([self.X, p])
        elif ev.button == 3 and len(self.X) > 1:
            i = int(np.argmin(np.linalg.norm(self.X - p, axis=1)))
            self.X = np.delete(self.X, i, axis=0)
        self.custom = True
        self.rebuild()

    def _rmax(self):
        if self.auto_rmax.isChecked():
            D = pairwise_distances(self.X)
            return float(max(D.max() / 2, 1e-6))
        return float(self.rmax.value())

    def rebuild(self):
        if self.X is None:
            return
        X = self.X.copy()
        rmax = self._rmax()
        methods = [self.method_a.currentText()]
        if self.method_b.currentText() != "(ninguno)":
            methods.append(self.method_b.currentText())
        max_dim = self.max_dim.value()
        nl, nu = self.landmarks.value(), self.nu.value()

        def job():
            out = {}
            for m in methods:
                F, L = build_filtration(m, X, max_dim, rmax, nl, nu)
                P = compute_persistence(F, cycles=False)
                out[m] = (F, P, L)
            return X, rmax, out

        self.run_job(job, self._built, "Construyendo complejos…")

    def _built(self, res):
        X, rmax, out = res
        if X is not self.X and not np.array_equal(X, self.X):
            return
        self.results = out
        self.r.setRange(0, rmax)
        self.redraw()

    # -- dibujo -------------------------------------------------------------------
    def redraw(self, *_):
        if not self.results or self.X is None:
            return
        fig = self.canvas.figure
        fig.clear()
        r = self.r.value()
        X = self.X
        d = X.shape[1]
        items = list(self.results.items())
        labels = [self.info_a, self.info_b]
        self.info_b.setVisible(len(items) > 1)
        for k, (m, (F, P, L)) in enumerate(items):
            ax = fig.add_subplot(1, len(items), k + 1, projection="3d" if d == 3 else None)
            S = F.simplices_at(r)
            show_b = self.show_balls.isChecked() and d == 2 and m != "Testigos (perezoso)"
            draw_complex(ax, X, S, balls=r if show_b else None,
                         active_vertices=L, labels=self.show_labels.isChecked())
            if d == 2 and self.show_voronoi.isChecked() and len(X) >= 4:
                try:
                    vor = Voronoi(X)
                    voronoi_plot_2d(vor, ax, show_points=False, show_vertices=False,
                                    line_colors=style.TEXT2, line_width=0.7, line_alpha=0.6)
                except Exception:   # puntos degenerados
                    pass
            setup_axes_for(ax, X, pad=0.12 + (2 * r / max(np.ptp(X), 1e-9) if show_b else 0))
            unit = "α" if m == "Testigos (perezoso)" else "r"
            ax.set_title(f"{m}   ({unit} = {r:.3f})")
            labels[k].setText(self._info_text(m, F, P, r))
        self.canvas.draw()

    def _info_text(self, m, F, P, r):
        counts = F.counts_at(r)
        names = ["vértices", "aristas", "triángulos", "tetraedros"]
        cs = ", ".join(f"{c} {names[i]}" for i, c in enumerate(counts) if i < len(names))
        chi = F.euler_at(r)
        betti = P.betti_numbers_at(r)
        bs = ", ".join(f"β<sub>{k}</sub> = {b}" for k, b in enumerate(betti))
        note = ""
        if F.reliable_dim < F.max_dim:
            note = (f"<br><small>β<sub>{F.max_dim}</sub> no se muestra: faltan los "
                    f"{F.max_dim + 1}-símplices que podrían matar esas clases.</small>")
        return f"<b>{m}</b>: {cs}.<br>χ = {chi}. &nbsp; {bs}{note}"
