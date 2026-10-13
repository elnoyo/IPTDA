"""Panel: representaciones vectoriales de diagramas (curvas de Betti,
paisajes, imágenes de persistencia y entropía)."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QComboBox, QLabel

from ...core.filtration import alpha_filtration, rips_filtration
from ...core.geometry import pairwise_distances
from ...core.persistence import compute_persistence
from ...core.vectorization import (betti_curve, finite_part, landscapes, persistence_image,
                                   persistent_entropy)
from .. import style
from ..plotting import draw_diagram, visible_pairs
from ..widgets import BasePanel, MplCanvas, PointCloudBox, dspin, ispin

THEORY = """
<h3>Vectorizaciones de diagramas</h3>
<p>Los diagramas de persistencia son multiconjuntos de puntos: no se pueden sumar
ni promediar directamente. Para usarlos en estadística o aprendizaje automático
se transforman en vectores o funciones:</p>
<ul>
<li><b>Curva de Betti</b>: β(t) = #{(b, d) : b ≤ t &lt; d}.</li>
<li><b>Paisajes de persistencia</b> (Bubenik, 2015): para cada punto, la «tienda»
Λ<sub>(b,d)</sub>(t) = max(0, min(t − b, d − t)); λ<sub>k</sub>(t) es el k-ésimo mayor
valor de las tiendas en t. Viven en un espacio de Banach: tienen media y son estables.</li>
<li><b>Imagen de persistencia</b> (Adams et al., 2017): en coordenadas (nacimiento,
persistencia), cada punto se sustituye por una gaussiana de anchura σ ponderada por
su persistencia; se integra sobre una rejilla de píxeles.</li>
<li><b>Entropía de persistencia</b>: E = −Σ p<sub>i</sub> log p<sub>i</sub>, con
p<sub>i</sub> = ℓ<sub>i</sub> / Σ ℓ<sub>j</sub> la proporción de la longitud total que aporta
cada barra.</li>
</ul>
<p>Las barras infinitas se truncan en el valor máximo de la filtración para las
curvas y paisajes, y se omiten en la imagen y la entropía.</p>
"""


class VectorizationPanel(BasePanel):
    title = "Vectorizaciones"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.P = None
        self.data = PointCloudBox(default="Dos circunferencias", n=70, noise=0.04, max_n=400)
        self.controls.addWidget(self.data)
        self.data.changed.connect(self.compute)
        f = self.add_group("Filtración")
        self.method = QComboBox()
        self.method.addItems(["Alfa", "Vietoris–Rips"])
        f.addRow("Método", self.method)
        self.method.currentIndexChanged.connect(self.compute)
        g = self.add_group("Representación")
        self.dim = QComboBox()
        self.dim.addItems(["H0", "H1", "H2"])
        self.dim.setCurrentIndex(1)
        self.k = ispin(1, 10, 3)
        self.sigma = dspin(0.001, 1.0, 0.03, 0.005)
        self.res = ispin(5, 100, 30)
        self.weight = QComboBox()
        self.weight.addItems(["lineal", "constante"])
        g.addRow("Diagrama", self.dim)
        g.addRow("Número de paisajes", self.k)
        g.addRow("σ de la imagen", self.sigma)
        g.addRow("Resolución (píxeles)", self.res)
        g.addRow("Peso", self.weight)
        self.dim.currentIndexChanged.connect(self.redraw)
        for w in (self.k, self.sigma, self.res):
            w.valueChanged.connect(self.redraw)
        self.weight.currentIndexChanged.connect(self.redraw)
        self.info = QLabel()
        self.info.setObjectName("badge")
        self.info.setWordWrap(True)
        self.controls.addWidget(self.info)
        self.finish_controls()
        self.set_theory(THEORY)
        self.canvas = MplCanvas()
        self.plots.addWidget(self.canvas)
        self.compute()

    def compute(self):
        X = self.data.sample()
        method = self.method.currentText()

        def job():
            if method == "Alfa":
                F = alpha_filtration(X)
                P = compute_persistence(F, cycles=False)
                vmax = float(F.values.max())
            else:
                rmax = float(pairwise_distances(X).max() / 2)
                F = rips_filtration(X, 2, rmax)      # H0 y H1
                P = compute_persistence(F, cycles=False)
                vmax = rmax
            return P, vmax

        self.run_job(job, self._done)

    def _done(self, res):
        self.P, self.vmax = res
        self.redraw()

    def redraw(self, *_):
        if self.P is None:
            return
        k = self.dim.currentIndex()
        self.set_status("")
        if k > self.P.max_dim:
            self.set_status(f"H{k} no está disponible para esta filtración/dimensión.", error=True)
            k = self.P.max_dim
        D = self.P.diagram(k)
        fin = finite_part(D)
        vmax = self.vmax
        # escala de los ejes: hasta la mayor muerte finita (o vmax)
        top = float(fin[:, 1].max()) if len(fin) else vmax
        top = max(top, 1e-9) * 1.1
        ts = np.linspace(0, top, 400)
        Dcap = finite_part(D, cap=top)

        fig = self.canvas.figure
        fig.clear()
        axs = fig.subplots(2, 2)
        draw_diagram(axs[0, 0], visible_pairs(self.P, [k]), top, None,
                     title=f"Diagrama H{k}")
        ax = axs[0, 1]
        ax.step(ts, betti_curve(Dcap, ts), where="post", color=style.dim_color(k))
        ax.set_title(f"Curva de Betti β{k}(t)")
        ax.set_xlabel("t")
        ax = axs[1, 0]
        L = landscapes(Dcap, ts, self.k.value())
        ramp = style.SEQ_CMAP(np.linspace(1.0, 0.35, self.k.value()))
        for j in range(self.k.value()):
            ax.plot(ts, L[j], color=ramp[j], lw=2, label=f"λ{j + 1}")
        ax.set_title("Paisajes de persistencia")
        ax.set_xlabel("t")
        ax.legend(loc="upper right")
        ax = axs[1, 1]
        pers_max = float((fin[:, 1] - fin[:, 0]).max()) * 1.1 if len(fin) else 1.0
        bmin = float(fin[:, 0].min()) if len(fin) else 0.0
        bmax = float(fin[:, 0].max()) if len(fin) else 1.0
        pad = max(3 * self.sigma.value(), 0.05 * max(bmax - bmin, 1e-9))
        br = (bmin - pad, bmax + pad)
        pr = (0.0, max(pers_max, 1e-9) + pad)
        img = persistence_image(D, br, pr, self.res.value(), self.sigma.value(), self.weight.currentText())
        im = ax.imshow(img, origin="lower", extent=(*br, *pr), aspect="auto", cmap=style.SEQ_CMAP)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Imagen de persistencia")
        ax.set_xlabel("nacimiento")
        ax.set_ylabel("persistencia")
        self.canvas.draw()
        E = persistent_entropy(D)
        self.info.setText(f"H{k}: {len(D)} puntos ({len(fin)} finitos).<br>"
                          f"Entropía de persistencia E = <b>{E:.4f}</b> "
                          f"(máximo posible log n = {np.log(max(len(fin), 1)):.4f}).<br>"
                          f"Vector de la imagen: {img.size} componentes.")
