"""Panel: filtraciones, códigos de barras y diagramas de persistencia."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QComboBox

from ...core.filtration import METHODS, build_filtration
from ...core.geometry import pairwise_distances
from ...core.persistence import compute_persistence
from ..plotting import (draw_barcode, draw_complex, draw_diagram, pick_barcode_row,
                        pick_diagram_point, setup_axes_for, visible_pairs)
from ..widgets import BasePanel, FloatSlider, MplCanvas, PointCloudBox, checkbox, dspin, ispin

THEORY = """
<h3>Homología persistente</h3>
<p>Una <b>filtración</b> es una familia creciente de complejos
K<sub>r</sub> ⊆ K<sub>s</sub> (r ≤ s). Las inclusiones inducen aplicaciones
H<sub>k</sub>(K<sub>r</sub>) → H<sub>k</sub>(K<sub>s</sub>): una clase <b>nace</b> en
<i>b</i> cuando aparece y <b>muere</b> en <i>d</i> cuando se vuelve trivial o se
fusiona con una clase más antigua (<i>regla del más viejo</i>).</p>
<p>El <b>teorema de estructura</b> asegura que el módulo de persistencia se
descompone de forma única en intervalos [b, d): es el <b>código de barras</b>. El
<b>diagrama de persistencia</b> dibuja cada intervalo como el punto (b, d).</p>
<ul>
<li>Barras largas / puntos lejos de la diagonal: rasgos topológicos robustos.</li>
<li>Barras cortas / puntos cerca de la diagonal: típicamente ruido.</li>
<li>β<sub>k</sub>(K<sub>r</sub>) = número de barras de dimensión k que contienen a r
(región sombreada del diagrama: b ≤ r &lt; d).</li>
</ul>
<p><b>Algoritmo:</b> reducción por columnas de la matriz borde con coeficientes en un
cuerpo Z/p (por defecto Z/2), con la optimización de «limpieza». Si la columna reducida del símplice j tiene pivote
i, se forma el par (i, j). La columna reducida es además un <b>ciclo
representante</b> de la clase.</p>
<p><b>Interacción:</b> haz clic en una barra o en un punto del diagrama para ver su
ciclo representante (en rojo); el parámetro r se sitúa en mitad de su vida. En H0,
se resalta la arista que fusiona las dos componentes.</p>
<p><b>Coeficientes:</b> el diagrama puede depender del cuerpo cuando la homología entera
tiene torsión (p. ej. el plano proyectivo: H<sub>1</sub>(RP²; Z) = Z/2). En las filtraciones
alfa y de Čech de puntos de R² o R³, cada K<sub>r</sub> equivale a una unión de bolas de R³,
cuya homología entera no tiene torsión, así que sus números de Betti no dependen del cuerpo.</p>
<p><b>Truncamiento:</b> con símplices hasta dimensión D solo es fiable la homología
hasta D−1, y con un radio máximo las barras que no mueren antes aparecen infinitas.</p>
"""


class PersistencePanel(BasePanel):
    title = "Persistencia"
    subtitle = "Cómo nacen y mueren los rasgos topológicos al crecer la escala"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.X = None
        self.F = self.P = self.L = None
        self.rmax_used = 1.0
        self.selected = None
        self.pairs = []
        self.inf_y = 1.0

        self.data = PointCloudBox(default="Ocho (dos círculos tangentes)", n=60, noise=0.04,
                                  max_n=500)
        self.controls.addWidget(self.data)
        self.data.changed.connect(self.compute)

        f = self.add_group("Filtración")
        self.method = QComboBox()
        self.method.addItems(METHODS)
        self.hom_dim = ispin(0, 2, 1)
        self.rmax = dspin(0.01, 100, 1.0, 0.05)
        self.auto_rmax = checkbox("automático (½ del diámetro)", True)
        self.landmarks = ispin(3, 200, 30)
        self.nu = ispin(0, 2, 1)
        self.field = QComboBox()
        for p in (2, 3, 5, 7):
            self.field.addItem(f"Z/{p}", p)
        f.addRow("Método", self.method)
        f.addRow("Coeficientes", self.field)
        f.addRow("Homología hasta dim.", self.hom_dim)
        f.addRow("Radio máximo", self.rmax)
        f.addRow("", self.auto_rmax)
        f.addRow("Puntos de referencia", self.landmarks)
        f.addRow("ν (testigos)", self.nu)
        self.method.currentIndexChanged.connect(self.compute)
        self.field.currentIndexChanged.connect(self.compute)
        for w in (self.hom_dim, self.rmax, self.landmarks, self.nu):
            w.valueChanged.connect(self.compute)
        self.auto_rmax.toggled.connect(self.compute)

        g = self.add_group("Vista")
        self.t = FloatSlider(0, 1, 0.2)
        self.t.valueChanged.connect(self.redraw)
        g.addRow("r", self.t)
        self.show_balls = checkbox("Mostrar bolas (2D)", False, self.redraw)
        self.show_complex = checkbox("Mostrar el complejo", True, self.redraw)
        g.addRow(self.show_balls)
        g.addRow(self.show_complex)
        self.betti = self.add_info()
        self.finish_controls()
        self.set_theory(THEORY)

        self.canvas = MplCanvas()
        self.plots.addWidget(self.canvas)
        self.canvas.canvas.mpl_connect("button_press_event", self._on_click)
        self.compute()

    def compute(self):
        X = self.data.sample()
        method = self.method.currentText()
        hd = self.hom_dim.value()
        max_dim = hd + 1
        if method == "Alfa":
            max_dim = X.shape[1]
        if self.auto_rmax.isChecked():
            rmax = float(pairwise_distances(X).max() / 2)
        else:
            rmax = float(self.rmax.value())
        nl, nu = self.landmarks.value(), self.nu.value()
        field = self.field.currentData()

        def job():
            F, L = build_filtration(method, X, max_dim, rmax, nl, nu)
            P = compute_persistence(F, max_dim=min(hd, F.reliable_dim), essential_cycles=True,
                                    field=field)
            return X, F, P, L, rmax

        self.run_job(job, self._done, "Calculando la persistencia…")

    def _done(self, res):
        self.X, self.F, self.P, self.L, rmax = res
        vmax = min(rmax, float(self.F.values.max())) if self.method.currentText() == "Alfa" else rmax
        self.rmax_used = max(vmax, 1e-9)
        self.selected = None
        self.t.setRange(0, self.rmax_used)
        self.redraw()

    def _on_click(self, ev):
        if self.P is None or self.canvas.toolbar.mode:
            return
        p = pick_barcode_row(ev, self.ax_bar, self.pairs) if ev.inaxes is self.ax_bar else \
            pick_diagram_point(ev, self.ax_dgm, self.pairs, self.inf_y)
        if p is None:
            return
        self.selected = p
        d = p.death if np.isfinite(p.death) else self.rmax_used
        mid = p.birth + 0.5 * (d - p.birth) if p.dim > 0 else p.death
        self.t.setValue(min(mid, self.rmax_used))
        self.redraw()

    def redraw(self, *_):
        if self.P is None:
            return
        t = self.t.value()
        fig = self.canvas.figure
        fig.clear()
        X = self.X
        d = X.shape[1]
        gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
        ax = fig.add_subplot(gs[:, 0], projection="3d" if d == 3 else None)
        self.ax_bar = fig.add_subplot(gs[0, 1])
        self.ax_dgm = fig.add_subplot(gs[1, 1])

        S = self.F.simplices_at(t) if self.show_complex.isChecked() else []
        hl, hl_dim = None, 1
        sel = self.selected
        if sel is not None:
            if sel.dim == 0 and sel.death_index is not None:
                hl = [self.F.simplices[sel.death_index]]
            elif sel.cycle:
                hl, hl_dim = sel.cycle, sel.dim
        balls = t if (self.show_balls.isChecked() and d == 2 and self.L is None) else None
        draw_complex(ax, X, S, balls=balls, highlight=hl, highlight_dim=hl_dim,
                     active_vertices=self.L)
        setup_axes_for(ax, X)
        unit = "α" if self.L is not None else "r"
        ax.set_title(f"{self.F.name}: complejo en {unit} = {t:.3f}")

        self.pairs = visible_pairs(self.P, range(self.P.max_dim + 1))
        draw_barcode(self.ax_bar, self.pairs, self.rmax_used, t, sel, xlabel=unit)
        self.inf_y = draw_diagram(self.ax_dgm, self.pairs, self.rmax_used, t, sel)
        self.canvas.draw()

        betti = self.P.betti_numbers_at(t)
        txt = " &nbsp; ".join(f"β<sub>{k}</sub>({unit}) = <b>{b}</b>" for k, b in enumerate(betti))
        txt += f"<br>{len(self.F)} símplices en la filtración; coeficientes en Z/{self.P.field}."
        top = sorted([p for p in self.pairs if p.dim >= 1], key=lambda p: -p.persistence)[:3]
        if top:
            txt += "<br>Rasgos más persistentes: " + "; ".join(
                f"H{p.dim} [{p.birth:.3f}, {'∞' if not np.isfinite(p.death) else f'{p.death:.3f}'})"
                for p in top)
        if sel is not None:
            dd = "∞" if not np.isfinite(sel.death) else f"{sel.death:.4f}"
            txt += (f"<br><b>Seleccionado:</b> H{sel.dim}, nace {sel.birth:.4f}, muere {dd}, "
                    f"persistencia {sel.persistence:.4f}")
        self.betti.setText(txt)
