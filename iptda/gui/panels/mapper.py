"""Panel: algoritmo Mapper."""

from __future__ import annotations

import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from PySide6.QtWidgets import QComboBox, QLabel

from ...core.mapper import LENSES, lens_values, mapper
from .. import style
from ..plotting import setup_axes_for
from ..widgets import BasePanel, FloatSlider, MplCanvas, PointCloudBox, dspin, ispin

THEORY = """
<h3>Mapper</h3>
<p>Mapper (Singh, Mémoli, Carlsson, 2007) resume la forma de unos datos en un
<b>grafo</b>:</p>
<ol>
<li>Se elige una <b>función filtro</b> (lente) f : X → ℝ (una coordenada, la
excentricidad, la densidad, la primera componente principal…).</li>
<li>Se recubre la imagen f(X) con <b>intervalos solapados</b> (número de intervalos
y porcentaje de solapamiento).</li>
<li>En cada preimagen f<sup>−1</sup>(I) se agrupan los puntos (<b>clustering</b>):
enlace simple cortado a una distancia o DBSCAN.</li>
<li>Cada grupo es un <b>nodo</b>; dos nodos se unen si comparten puntos.</li>
</ol>
<p>El grafo es el (1-esqueleto del) <b>nervio</b> del recubrimiento de X formado
por los grupos. Es una versión discreta del <b>grafo de Reeb</b> de f. Sus números
de Betti: β<sub>0</sub> = componentes y β<sub>1</sub> = aristas − nodos + β<sub>0</sub> (ciclos
independientes).</p>
<p><b>Parámetros:</b> pocos intervalos o mucho solapamiento → grafo grueso; muchos
intervalos con poco solapamiento → grafo fragmentado. Un umbral de agrupamiento
pequeño parte los grupos; uno grande los fusiona.</p>
<p><b>Interacción:</b> haz clic en un nodo para ver sus puntos y en la franja de un
intervalo (gráfica de la lente) para ver su preimagen.</p>
"""


class MapperPanel(BasePanel):
    title = "Mapper"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.X = None
        self.G = None
        self.pos = None
        self.sel_node = None
        self.sel_interval = None

        self.data = PointCloudBox(default="Ocho (dos círculos tangentes)", n=400, noise=0.03,
                                  max_n=3000)
        self.controls.addWidget(self.data)
        self.data.changed.connect(self.compute)
        f = self.add_group("Lente y recubrimiento")
        self.lens = QComboBox()
        self.lens.addItems(LENSES)
        self.sigma = dspin(0.01, 5, 0.3, 0.05)
        self.nint = ispin(1, 60, 10)
        self.overlap = FloatSlider(0, 0.9, 0.3, steps=90, fmt="{:.0%}")
        f.addRow("Función filtro", self.lens)
        f.addRow("σ (densidad)", self.sigma)
        f.addRow("Intervalos", self.nint)
        f.addRow("Solapamiento", self.overlap)
        g = self.add_group("Agrupamiento")
        self.clust = QComboBox()
        self.clust.addItems(["Enlace simple", "DBSCAN"])
        self.eps = dspin(0.001, 10, 0.25, 0.01)
        self.minpts = ispin(1, 50, 3)
        g.addRow("Método", self.clust)
        g.addRow("Distancia (umbral / ε)", self.eps)
        g.addRow("Mín. vecinos (DBSCAN)", self.minpts)
        self.lens.currentIndexChanged.connect(self.compute)
        self.clust.currentIndexChanged.connect(self.compute)
        for w in (self.sigma, self.nint, self.eps, self.minpts):
            w.valueChanged.connect(self.compute)
        self.overlap.valueChanged.connect(self.compute)
        self.info = QLabel()
        self.info.setObjectName("badge")
        self.info.setWordWrap(True)
        self.controls.addWidget(self.info)
        self.finish_controls()
        self.set_theory(THEORY)
        self.canvas = MplCanvas()
        self.plots.addWidget(self.canvas)
        self.canvas.canvas.mpl_connect("button_press_event", self._on_click)
        self.compute()

    def compute(self):
        X = self.data.sample()
        lens, sigma = self.lens.currentText(), self.sigma.value()
        args = (self.nint.value(), self.overlap.value(), self.clust.currentText(),
                self.eps.value(), self.minpts.value())

        def job():
            f = lens_values(X, lens, sigma)
            G = mapper(X, f, *args)
            H = nx.Graph()
            H.add_nodes_from(range(len(G.nodes)))
            H.add_weighted_edges_from(G.edges)
            pos = nx.spring_layout(H, seed=1, iterations=200) if len(G.nodes) else {}
            return X, G, pos

        self.run_job(job, self._done, "Calculando Mapper…")

    def _done(self, res):
        self.X, self.G, self.pos = res
        self.sel_node = None
        self.sel_interval = None
        self.redraw()

    def _on_click(self, ev):
        if self.G is None or self.canvas.toolbar.mode or ev.inaxes is None:
            return
        if ev.inaxes is self.ax_graph and self.pos:
            P = np.array([self.pos[i] for i in range(len(self.G.nodes))])
            disp = self.ax_graph.transData.transform(P)
            dist = np.hypot(disp[:, 0] - ev.x, disp[:, 1] - ev.y)
            i = int(np.argmin(dist))
            self.sel_node = i if dist[i] < 15 else None
            self.sel_interval = None
            self.redraw()
        elif ev.inaxes is self.ax_lens and ev.ydata is not None:
            k = int(round(ev.ydata))
            self.sel_interval = k if 0 <= k < len(self.G.intervals) else None
            self.sel_node = None
            self.redraw()

    def redraw(self):
        G, X = self.G, self.X
        if G is None:
            return
        fig = self.canvas.figure
        fig.clear()
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1.6, 1])
        d = X.shape[1]
        ax = fig.add_subplot(gs[0, 0], projection="3d" if d == 3 else None)
        f = G.lens
        vmin, vmax = float(f.min()), float(f.max())
        hl = None
        if self.sel_node is not None:
            hl = G.nodes[self.sel_node]
        elif self.sel_interval is not None:
            a, b = G.intervals[self.sel_interval]
            hl = np.nonzero((f >= a) & (f <= b))[0]
        alpha = 0.25 if hl is not None else 0.9
        if d == 2:
            ax.scatter(X[:, 0], X[:, 1], c=f, cmap=style.SEQ_CMAP, vmin=vmin, vmax=vmax, s=10, alpha=alpha)
            if hl is not None:
                ax.scatter(X[hl, 0], X[hl, 1], color=style.HIGHLIGHT, s=16)
        else:
            ax.scatter(*X.T, c=f, cmap=style.SEQ_CMAP, vmin=vmin, vmax=vmax, s=6, alpha=alpha)
            if hl is not None:
                ax.scatter(*X[hl].T, color=style.HIGHLIGHT, s=12)
        setup_axes_for(ax, X)
        ax.set_title("Datos coloreados por la lente")

        # lente y recubrimiento
        self.ax_lens = axl = fig.add_subplot(gs[1, 0])
        for k, (a, b) in enumerate(G.intervals):
            sel = k == self.sel_interval
            axl.add_patch(Rectangle((a, k - 0.35), b - a, 0.7,
                                    facecolor=style.HIGHLIGHT if sel else style.SEQ_CMAP(0.15 + 0.7 * k / max(len(G.intervals) - 1, 1)),
                                    alpha=0.8 if sel else 0.6, edgecolor="none"))
        jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(f))
        axl.scatter(f, -1 + jitter * 0.6, s=3, color=style.TEXT2)
        axl.set_xlim(vmin - 0.02 * (vmax - vmin + 1e-9), vmax + 0.02 * (vmax - vmin + 1e-9))
        axl.set_ylim(-1.6, len(G.intervals) - 0.4)
        axl.set_yticks([])
        axl.set_xlabel(self.lens.currentText())
        axl.set_title("Valores de la lente (abajo) y recubrimiento por intervalos")
        axl.grid(axis="y", visible=False)

        # grafo
        self.ax_graph = axg = fig.add_subplot(gs[:, 1])
        axg.grid(False)
        axg.set_xticks([])
        axg.set_yticks([])
        if self.pos:
            P = np.array([self.pos[i] for i in range(len(G.nodes))])
            segs = [[P[u], P[v]] for u, v, _ in G.edges]
            axg.add_collection(LineCollection(segs, colors=style.TEXT2, linewidths=1.2, zorder=1))
            sizes = np.array([len(m) for m in G.nodes])
            s = 30 + 400 * np.sqrt(sizes / sizes.max())
            axg.scatter(P[:, 0], P[:, 1], s=s, c=G.node_lens_mean(), cmap=style.SEQ_CMAP,
                        vmin=vmin, vmax=vmax, edgecolors=style.TEXT2, linewidths=0.6, zorder=2)
            if self.sel_node is not None:
                axg.scatter(*P[self.sel_node], s=s[self.sel_node] * 1.3, facecolors="none",
                            edgecolors=style.HIGHLIGHT, linewidths=2.5, zorder=3)
            if self.sel_interval is not None:
                ks = [i for i, k in enumerate(G.node_interval) if k == self.sel_interval]
                axg.scatter(P[ks, 0], P[ks, 1], s=s[ks] * 1.3, facecolors="none",
                            edgecolors=style.HIGHLIGHT, linewidths=2.5, zorder=3)
            pad = 0.15
            axg.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
            axg.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)
        axg.set_title("Grafo Mapper (color = media de la lente, tamaño = nº de puntos)")
        self.canvas.draw()

        b0, b1 = G.betti
        covered = len(set(np.concatenate(G.nodes).tolist())) if G.nodes else 0
        txt = (f"Nodos: <b>{len(G.nodes)}</b>, aristas: <b>{len(G.edges)}</b><br>"
               f"β<sub>0</sub> = <b>{b0}</b>, β<sub>1</sub> = <b>{b1}</b> (del grafo)<br>"
               f"Puntos en algún nodo: {covered} de {len(X)}")
        if self.sel_node is not None:
            txt += (f"<br><b>Nodo {self.sel_node}</b>: {len(G.nodes[self.sel_node])} puntos, "
                    f"intervalo {G.node_interval[self.sel_node] + 1}")
        self.info.setText(txt)
