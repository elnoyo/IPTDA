"""Panel: series temporales (inmersión de Takens y filtración por subnivel)."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QComboBox, QTabWidget

from ...core.filtration import alpha_filtration, lower_star_path, rips_filtration
from ...core.geometry import maxmin_subsample, pairwise_distances
from ...core.persistence import compute_persistence
from ...core.timeseries import (SIGNALS, autocorrelation, make_signal, pca_project, suggest_tau,
                                takens_embedding)
from .. import style
from ..plotting import draw_diagram, setup_axes_for, visible_pairs
from ..widgets import BasePanel, FloatSlider, MplCanvas, checkbox, dspin, ispin

THEORY = """
<h3>Series temporales</h3>
<p><b>Inmersión por retardos (Takens).</b> De una serie x<sub>0</sub>, x<sub>1</sub>, … se
forma la nube de puntos</p>
<p align="center">v<sub>i</sub> = (x<sub>i</sub>, x<sub>i+τ</sub>, …, x<sub>i+(d−1)τ</sub>) ∈ ℝ<sup>d</sup>.</p>
<p>El <b>teorema de Takens</b> asegura que, para un sistema dinámico determinista y
d suficientemente grande (d &gt; 2·dimensión del atractor), esta nube reconstruye el
atractor salvo difeomorfismo. Así:</p>
<ul>
<li>una señal <b>periódica</b> da una curva cerrada → una barra larga en H1;</li>
<li>una señal <b>cuasiperiódica</b> (dos frecuencias inconmensurables) llena un
<b>toro</b> → dos barras largas en H1 (y una en H2) si d ≥ 3;</li>
<li>el <b>ruido</b> no produce rasgos persistentes.</li>
</ul>
<p>La máxima persistencia en H1 sirve como <b>medida de periodicidad</b> (Perea–Harer,
SW1PerS). El retardo τ suele elegirse con el primer cero de la autocorrelación (un
cuarto del periodo en una sinusoide). Con d &gt; 3 se dibuja la proyección sobre las
tres primeras componentes principales.</p>
<p><b>Filtración por subnivel.</b> Para la función f(i) = x<sub>i</sub> sobre el camino
0–1–…–n, los subniveles {f ≤ t} forman una filtración. En H0, cada <b>mínimo local</b>
crea una componente y cada <b>máximo local</b> fusiona dos (muere la más joven): los
pares (mínimo, máximo) miden la «prominencia» de cada valle.</p>
"""


class TimeSeriesPanel(BasePanel):
    title = "Series temporales"
    subtitle = "Periodicidad y forma de una señal mediante ventanas deslizantes y subniveles"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.x = None
        self.E = None
        self.sub_idx = None
        self.P = None

        f = self.add_group("Señal")
        self.signal = QComboBox()
        self.signal.addItems(SIGNALS)
        self.n = ispin(100, 5000, 600, 50)
        self.period = dspin(4, 1000, 50, 1, 1)
        self.noise = dspin(0, 2, 0.0, 0.02)
        self.seed = ispin(0, 9999, 0)
        f.addRow("Tipo", self.signal)
        f.addRow("Muestras", self.n)
        f.addRow("Periodo (muestras)", self.period)
        f.addRow("Ruido (σ)", self.noise)
        f.addRow("Semilla", self.seed)
        self.signal.currentIndexChanged.connect(self.compute)
        for w in (self.n, self.period, self.noise, self.seed):
            w.valueChanged.connect(self.compute)

        g = self.add_group("Inmersión de Takens")
        self.dim = ispin(2, 10, 3)
        self.tau = ispin(1, 500, 12)
        self.step = ispin(1, 50, 2)
        self.npts = ispin(20, 400, 150, 10)
        self.method = QComboBox()
        self.method.addItems(["Alfa (si d ≤ 3)", "Vietoris–Rips"])
        g.addRow("Dimensión d", self.dim)
        g.addRow("Retardo τ", self.tau)
        g.addRow("Paso entre ventanas", self.step)
        g.addRow("Puntos para la persistencia", self.npts)
        g.addRow("Filtración", self.method)
        for w in (self.dim, self.tau, self.step, self.npts):
            w.valueChanged.connect(self.compute)
        self.method.currentIndexChanged.connect(self.compute)
        self.add_button("Usar τ sugerido (autocorrelación)", self._suggest, primary=False)
        self.window = FloatSlider(0, 1, 0.1, fmt="{:.2f}")
        self.window.valueChanged.connect(self.redraw_takens)
        g.addRow("Ventana resaltada", self.window)

        h = self.add_group("Filtración por subnivel")
        self.level = FloatSlider(-1, 1, 0.0)
        self.level.valueChanged.connect(self.redraw_sublevel)
        self.superlevel = checkbox("Usar superniveles (−f)", False, self.compute_sublevel)
        h.addRow("Nivel t", self.level)
        h.addRow(self.superlevel)

        self.info = self.add_info()
        self.finish_controls()
        self.set_theory(THEORY)

        self.tabs = QTabWidget()
        self.c_takens = MplCanvas()
        self.c_sub = MplCanvas()
        self.tabs.addTab(self.c_takens, "Inmersión de Takens")
        self.tabs.addTab(self.c_sub, "Filtración por subnivel")
        self.plots.addWidget(self.tabs)
        self.compute()

    def refresh(self):
        self.redraw_takens()
        self.redraw_sublevel()

    # -- señal ----------------------------------------------------------------------
    def _signal(self):
        return make_signal(self.signal.currentText(), self.n.value(), self.noise.value(),
                           self.seed.value(), self.period.value())

    def _suggest(self):
        _, x = self._signal()
        self.tau.setValue(suggest_tau(x, max_lag=len(x) // 2))

    def compute(self):
        t, x = self._signal()
        self.x = x
        d, tau, step, npts = self.dim.value(), self.tau.value(), self.step.value(), self.npts.value()
        use_alpha = self.method.currentIndex() == 0 and d <= 3

        def job():
            E = takens_embedding(x, d, tau, step)
            idx = maxmin_subsample(E, min(npts, len(E)))
            Es = E[idx]
            if use_alpha:
                F = alpha_filtration(Es)
                P = compute_persistence(F, max_dim=min(2, d - 1), cycles=False)
                vmax = float(F.values.max())
            else:
                rmax = float(pairwise_distances(Es).max() / 2)
                F = rips_filtration(Es, 2, rmax)
                P = compute_persistence(F, max_dim=1, cycles=False)
                vmax = rmax
            return x, E, idx, P, vmax, "Alfa" if use_alpha else "Vietoris–Rips"

        self.run_job(job, self._done, "Calculando la inmersión y su persistencia…")
        self.compute_sublevel()

    def _done(self, res):
        x, self.E, self.sub_idx, self.P, self.vmax, self.mname = res
        if x is not self.x:
            return
        self.redraw_takens()

    # -- Takens -----------------------------------------------------------------------
    def redraw_takens(self, *_):
        if self.P is None or self.E is None:
            return
        x, E = self.x, self.E
        d, tau, step = self.dim.value(), self.tau.value(), self.step.value()
        if E.shape[1] != d:
            return
        fig = self.c_takens.figure
        fig.clear()
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.6])
        ax = fig.add_subplot(gs[0, :2])
        ax.plot(x, color=style.TEXT2, lw=1.0)
        i_win = int(round(self.window.value() * (len(E) - 1)))
        i0 = i_win * step
        idx = i0 + tau * np.arange(d)
        ax.scatter(idx, x[idx], color=style.HIGHLIGHT, s=40, zorder=3)
        comps = ", ".join(f"x{j}" for j in idx[:4]) + (", …" if d > 4 else "")
        ax.set_title(f"Señal: {self.signal.currentText()}\n"
                     f"en rojo, la ventana v{i_win} = ({comps})")
        ax.set_xlabel("i")
        ax2 = fig.add_subplot(gs[0, 2])
        ac = autocorrelation(x, min(len(x) // 2, max(4 * tau, 50)))
        ax2.plot(ac, color=style.dim_color(0), lw=1.5)
        ax2.axvline(tau, color=style.HIGHLIGHT, lw=1, ls="--")
        ax2.axhline(0, color=style.MUTED, lw=0.8)
        ax2.set_title("Autocorrelación\n(τ en rojo)")
        ax2.set_xlabel("retardo")

        Y = E if d <= 3 else pca_project(E, 3)
        Ys = Y[self.sub_idx]
        proj = "3d" if Y.shape[1] == 3 else None
        ax3 = fig.add_subplot(gs[1, :2], projection=proj)
        if proj is None:
            ax3.plot(Y[:, 0], Y[:, 1], color=style.MUTED, lw=0.6)
            ax3.scatter(Ys[:, 0], Ys[:, 1], s=10, color=style.dim_color(0))
            ax3.scatter([Y[i_win, 0]], [Y[i_win, 1]], s=80, color=style.HIGHLIGHT, zorder=4)
            ax3.set_xlabel("x(i)")
            ax3.set_ylabel("x(i+τ)")
        else:
            ax3.plot(*Y.T, color=style.MUTED, lw=0.5)
            ax3.scatter(*Ys.T, s=8, color=style.dim_color(0))
            ax3.scatter(*Y[i_win], s=80, color=style.HIGHLIGHT)
        setup_axes_for(ax3, Y)
        ax3.set_title(f"Inmersión d = {d}, τ = {tau}" + (" (proyección PCA)" if d > 3 else ""))

        ax4 = fig.add_subplot(gs[1, 2])
        pairs = visible_pairs(self.P, range(self.P.max_dim + 1))
        draw_diagram(ax4, pairs, self.vmax, None, title=f"Diagrama ({self.mname})")
        self.c_takens.draw()

        D1 = self.P.diagram(1)
        pers = np.sort(np.diff(D1, axis=1).ravel())[::-1] if len(D1) else np.array([])
        txt = (f"Nube: {len(E)} ventanas; persistencia sobre {len(self.sub_idx)} puntos (maxmin).<br>"
               f"Máx. persistencia H1 (periodicidad): <b>{pers[0] if len(pers) else 0:.4f}</b>")
        if len(pers) > 1:
            txt += f"; segunda: {pers[1]:.4f}"
        if self.P.max_dim >= 2:
            D2 = self.P.diagram(2)
            if len(D2):
                txt += f"<br>Máx. persistencia H2: {np.max(np.diff(D2, axis=1)):.4f}"
        self.info.setText(txt)

    # -- subnivel -----------------------------------------------------------------------
    def compute_sublevel(self, *_):
        if self.x is None:
            return
        f = -self.x if self.superlevel.isChecked() else self.x
        self.f = f
        self.Fs = lower_star_path(f)
        self.Ps = compute_persistence(self.Fs, max_dim=0)
        self.level.setRange(float(f.min()), float(f.max()))
        self.redraw_sublevel()

    def redraw_sublevel(self, *_):
        if getattr(self, "Ps", None) is None:
            return
        f, t = self.f, self.level.value()
        fig = self.c_sub.figure
        fig.clear()
        gs = fig.add_gridspec(1, 3)
        ax = fig.add_subplot(gs[0, :2])
        i = np.arange(len(f))
        ax.plot(i, f, color=style.MUTED, lw=1.0)
        under = np.where(f <= t, f, np.nan)
        ax.plot(i, under, color=style.dim_color(0), lw=2.0, label="{f ≤ t}")
        ax.axhline(t, color=style.TEXT, lw=1, ls="--")
        for p in self.Ps.by_dim(0):
            bi = self.Fs.simplices[p.birth_index][0]
            ax.scatter([bi], [p.birth], color=style.dim_color(0), s=25, zorder=3)
            if p.death_index is not None:
                e = self.Fs.simplices[p.death_index]
                di = e[int(np.argmax(f[list(e)]))]
                ax.plot([bi, bi], [p.birth, p.death], color=style.dim_color(1), lw=1.2, alpha=0.8)
                ax.scatter([di], [p.death], color=style.dim_color(1), marker="^", s=25, zorder=3)
        ax.set_title(("−f" if self.superlevel.isChecked() else "f") +
                     ": subnivel en azul; pares (mínimo ●, máximo ▲)")
        ax.set_xlabel("i")
        ax.legend(loc="upper right")
        ax2 = fig.add_subplot(gs[0, 2])
        pairs = visible_pairs(self.Ps, [0])
        lo, hi = float(f.min()), float(f.max())
        draw_diagram(ax2, pairs, hi, t, vmin=lo, title="Diagrama H0 (subnivel)")
        self.c_sub.draw()
        b0 = self.Ps.betti_at(t, 0)
        self.c_sub.setToolTip(f"β0 del subnivel en t = {t:.3f}: {b0}")
        self.tabs.setTabText(1, f"Filtración por subnivel (β0 = {b0})")
