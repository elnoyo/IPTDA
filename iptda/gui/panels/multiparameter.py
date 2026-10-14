"""Panel: persistencia biparamétrica (bifiltración Rips × codensidad)."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import BoundaryNorm
from PySide6.QtWidgets import QComboBox

from ...core.geometry import pairwise_distances
from ...core.multipers import (CODENSITIES, codensity, fibered_barcode, hilbert_function,
                               rips_codensity_bifiltration)
from .. import style
from ..plotting import draw_barcode, draw_complex, draw_diagram, setup_axes_for, visible_pairs
from ..widgets import BasePanel, FloatSlider, MplCanvas, PointCloudBox, checkbox, dspin, ispin

THEORY = """
<h3>Persistencia biparamétrica</h3>
<p>La persistencia de un parámetro <b>no es robusta frente a atípicos</b>: unos pocos
puntos aislados pueden destruir la barra del círculo. Una solución es filtrar
también por <b>densidad</b>. Cada símplice σ recibe un bigrado
(r<sub>σ</sub>, c<sub>σ</sub>): su valor de Rips y la mayor <b>codensidad</b> de sus vértices
(distancia al k-ésimo vecino: grande en zonas poco densas). El complejo</p>
<p align="center">K(r, c) = {σ : r<sub>σ</sub> ≤ r, c<sub>σ</sub> ≤ c}</p>
<p>crece en ambas direcciones: es una <b>bifiltración</b>.</p>
<p><b>No existe un invariante discreto completo</b> para módulos de 2 parámetros
(Carlsson–Zomorodian, 2009): no hay «código de barras». Se usan invariantes
parciales:</p>
<ul>
<li><b>Función de Hilbert</b>: HF(r, c) = dim H<sub>k</sub>(K(r, c)) (mapa de calor).</li>
<li><b>Invariante de rango</b>: rk(a → b) = rango de H<sub>k</sub>(K(a)) → H<sub>k</sub>(K(b)),
para a ≤ b. Cuenta las clases de K(a) que siguen vivas en K(b).</li>
<li><b>Código de barras fibrado</b> (RIVET): restringido a una recta de pendiente
no negativa, el módulo es de un parámetro y tiene código de barras. Las barras de
la recta que pasa por a y b que contienen [t<sub>a</sub>, t<sub>b</sub>] dan rk(a → b).
El conjunto de todos los códigos fibrados equivale al invariante de rango.</li>
</ul>
<p><b>Interacción:</b> haz clic en el mapa de calor para fijar el punto a (se dibuja
K(a) a la izquierda). El ángulo θ (en coordenadas normalizadas del rectángulo) fija
la recta y «s» el punto b sobre ella. Funciona con nubes 2D y 3D; con «Atípicos añadidos»
puedes contaminar cualquier conjunto y ver cómo la densidad los separa. θ = 0° es la filtración de Rips de los puntos
con codensidad ≤ c<sub>a</sub>; θ = 90° es la filtración por codensidad con radio fijo.</p>
"""


class MultiParameterPanel(BasePanel):
    title = "Persistencia biparamétrica"
    subtitle = "Escala y densidad a la vez: invariantes de un módulo de dos parámetros"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.B = None
        self.HF = None
        self.a = None

        self.data = PointCloudBox(dims=(2, 3), default="Circunferencia con atípicos", n=70,
                                  noise=0.03, max_n=150, outliers=0.0)
        self.controls.addWidget(self.data)
        self.data.changed.connect(self.compute)
        f = self.add_group("Bifiltración")
        self.cod = QComboBox()
        self.cod.addItems(CODENSITIES)
        self.k = ispin(1, 50, 5)
        self.sigma = dspin(0.01, 5, 0.3, 0.05)
        self.rmax = dspin(0.01, 10, 0.8, 0.05)
        self.auto_rmax = checkbox("automático (40 % del diámetro)", True)
        self.grid = ispin(5, 60, 30)
        f.addRow("Codensidad", self.cod)
        f.addRow("k (vecino)", self.k)
        f.addRow("σ (densidad)", self.sigma)
        f.addRow("Radio máximo", self.rmax)
        f.addRow("", self.auto_rmax)
        f.addRow("Resolución de la rejilla", self.grid)
        self.cod.currentIndexChanged.connect(self.compute)
        for w in (self.k, self.sigma, self.rmax, self.grid):
            w.valueChanged.connect(self.compute)
        self.auto_rmax.toggled.connect(self.compute)

        g = self.add_group("Invariantes")
        self.dim = QComboBox()
        self.dim.addItems(["H0", "H1"])
        self.dim.setCurrentIndex(1)
        self.theta = FloatSlider(0, 90, 0, steps=90, fmt="{:.0f}°")
        self.s = FloatSlider(0, 1, 0.3, fmt="{:.2f}")
        g.addRow("Homología", self.dim)
        g.addRow("Ángulo θ de la recta", self.theta)
        g.addRow("Posición s de b", self.s)
        self.dim.currentIndexChanged.connect(self.redraw)
        self.theta.valueChanged.connect(self.redraw)
        self.s.valueChanged.connect(self.redraw)
        self.info = self.add_info()
        self.finish_controls()
        self.set_theory(THEORY)
        self.canvas = MplCanvas()
        self.plots.addWidget(self.canvas)
        self.canvas.canvas.mpl_connect("button_press_event", self._on_click)
        self.compute()

    # -- cálculo ------------------------------------------------------------------
    def compute(self):
        X = self.data.sample()
        kind, k, sigma, G = self.cod.currentText(), self.k.value(), self.sigma.value(), self.grid.value()
        if self.auto_rmax.isChecked():
            rmax = 0.4 * float(pairwise_distances(X).max())
        else:
            rmax = float(self.rmax.value())

        def job():
            cod = codensity(X, kind, k, sigma)
            B = rips_codensity_bifiltration(X, cod, 2, rmax)
            rg = np.linspace(0, rmax, G)
            cg = np.linspace(float(cod.min()), float(cod.max()), G)
            HF = hilbert_function(B, rg, cg, dims=(0, 1))
            return X, cod, B, rg, cg, HF, rmax

        self.run_job(job, self._done, "Calculando la función de Hilbert…")

    def _done(self, res):
        self.X, self.cod_v, self.B, self.rg, self.cg, self.HF, self.rmax_used = res
        self.a = (0.35 * self.rmax_used, float(np.quantile(self.cod_v, 0.8)))
        self.redraw()

    def _on_click(self, ev):
        if self.B is None or self.canvas.toolbar.mode or ev.inaxes is not getattr(self, "ax_hf", None):
            return
        if ev.xdata is None:
            return
        r = float(np.clip(ev.xdata, self.rg[0], self.rg[-1]))
        c = float(np.clip(ev.ydata, self.cg[0], self.cg[-1]))
        self.a = (r, c)
        self.redraw()

    def _direction(self):
        th = np.deg2rad(self.theta.value())
        dr = self.rg[-1] - self.rg[0]
        dc = self.cg[-1] - self.cg[0]
        ur = np.cos(th) * dr
        uc = np.sin(th) * dc
        # evita componentes residuales por redondeo en 0° y 90°
        if abs(ur) < 1e-12 * max(dr, 1):
            ur = 0.0
        if abs(uc) < 1e-12 * max(dc, 1):
            uc = 0.0
        return ur, uc

    def _t_range(self, a, u):
        lo, hi = -np.inf, np.inf
        for ai, ui, mn, mx in ((a[0], u[0], self.rg[0], self.rg[-1]), (a[1], u[1], self.cg[0], self.cg[-1])):
            if ui > 0:
                lo = max(lo, (mn - ai) / ui)
                hi = min(hi, (mx - ai) / ui)
        return lo, hi

    # -- dibujo -------------------------------------------------------------------
    def redraw(self, *_):
        if self.B is None or self.a is None:
            return
        k = self.dim.currentIndex()
        a = self.a
        u = self._direction()
        t_lo, t_hi = self._t_range(a, u)
        s = self.s.value() * max(t_hi, 0.0)
        b = (a[0] + s * u[0], a[1] + s * u[1])
        P = fibered_barcode(self.B, a, u, max_dim=1)

        fig = self.canvas.figure
        fig.clear()
        gs = fig.add_gridspec(2, 2)
        ax0 = fig.add_subplot(gs[0, 0], projection="3d" if self.X.shape[1] == 3 else None)
        S = self.B.complex_at(*a)
        active = np.nonzero(self.cod_v <= a[1])[0]
        draw_complex(ax0, self.X, S, active_vertices=active)
        setup_axes_for(ax0, self.X)
        ax0.set_title(f"K(a): r ≤ {a[0]:.3f}, codensidad ≤ {a[1]:.3f}")

        self.ax_hf = ax1 = fig.add_subplot(gs[0, 1])
        H = self.HF[k]
        dr = (self.rg[1] - self.rg[0]) / 2 if len(self.rg) > 1 else 0.5
        dc = (self.cg[1] - self.cg[0]) / 2 if len(self.cg) > 1 else 0.5
        top = max(int(H.max()), 1)
        cmap = style.SEQ_CMAP.resampled(top + 1)
        norm = BoundaryNorm(np.arange(-0.5, top + 1.5), cmap.N)
        im = ax1.imshow(H, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                        interpolation="nearest",
                        extent=(self.rg[0] - dr, self.rg[-1] + dr, self.cg[0] - dc, self.cg[-1] + dc))
        ticks = np.arange(0, top + 1, max(1, (top + 1) // 8))
        cb = fig.colorbar(im, ax=ax1, shrink=0.85, ticks=ticks)
        cb.set_label(f"dim H{k}")
        ax1.grid(False)
        if np.isfinite(t_lo) and np.isfinite(t_hi) and t_hi > t_lo:
            ts = np.array([t_lo, t_hi])
            ax1.plot(a[0] + ts * u[0], a[1] + ts * u[1], color=style.HIGHLIGHT, lw=1.5)
        ax1.scatter([a[0]], [a[1]], color=style.HIGHLIGHT, s=50, zorder=4)
        ax1.scatter([b[0]], [b[1]], color=style.HIGHLIGHT, marker="s", s=40, zorder=4)
        ax1.annotate("a", a, xytext=(6, 6), textcoords="offset points", color=style.HIGHLIGHT,
                     fontweight="bold")
        ax1.annotate("b", b, xytext=(6, -12), textcoords="offset points", color=style.HIGHLIGHT,
                     fontweight="bold")
        ax1.set_xlim(self.rg[0] - dr, self.rg[-1] + dr)
        ax1.set_ylim(self.cg[0] - dc, self.cg[-1] + dc)
        ax1.set_xlabel("r (Rips)")
        ax1.set_ylabel("codensidad c")
        ax1.set_title(f"Función de Hilbert de H{k} (clic: fija a)")

        ax2 = fig.add_subplot(gs[1, 0])
        pairs = visible_pairs(P, [k])
        t_hi_f = t_hi if np.isfinite(t_hi) else 1.0
        t_lo_f = t_lo if np.isfinite(t_lo) else 0.0
        draw_barcode(ax2, pairs, t_hi_f, None, xmin=min(t_lo_f, 0.0),
                     title=f"Código de barras fibrado H{k} (recta θ = {self.theta.value():.0f}°)",
                     xlabel="t  (a = L(0), b = L(s))")
        ax2.axvline(0, color=style.HIGHLIGHT, lw=1.2, ls="--")
        ax2.axvline(s, color=style.HIGHLIGHT, lw=1.2, ls=":")

        ax3 = fig.add_subplot(gs[1, 1])
        draw_diagram(ax3, pairs, t_hi_f, None, vmin=min(t_lo_f, 0.0),
                     title=f"Diagrama de la recta (H{k})")
        self.canvas.draw()

        rank = sum(1 for p in P.pairs if p.dim == k and p.birth <= 0.0 and s < p.death)
        hf_a = sum(1 for p in P.pairs if p.dim == k and p.birth <= 0.0 < p.death)
        hf_b = sum(1 for p in P.pairs if p.dim == k and p.birth <= s < p.death)
        n_out = int(np.sum(self.cod_v > a[1]))
        self.info.setText(
            f"a = ({a[0]:.3f}, {a[1]:.3f}), &nbsp; b = ({b[0]:.3f}, {b[1]:.3f})<br>"
            f"dim H{k}(K(a)) = <b>{hf_a}</b>, &nbsp; dim H{k}(K(b)) = <b>{hf_b}</b><br>"
            f"Invariante de rango rk(a → b) = <b>{rank}</b><br>"
            f"Puntos excluidos en a (codensidad > c<sub>a</sub>): {n_out} de {len(self.X)}")
