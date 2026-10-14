"""Panel: estabilidad de los diagramas de persistencia y distancias entre
diagramas."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from PySide6.QtWidgets import QComboBox

from ...core.distances import bottleneck, wasserstein
from ...core.filtration import alpha_filtration, cech_filtration, rips_filtration
from ...core.geometry import hausdorff_distance
from ...core.persistence import compute_persistence
from .. import style
from ..plotting import setup_axes_for
from ..widgets import BasePanel, FloatSlider, MplCanvas, PointCloudBox, ispin

THEORY = """
<h3>Estabilidad</h3>
<p>Los diagramas de persistencia se comparan con la <b>distancia de cuello de
botella</b></p>
<p align="center">d<sub>B</sub>(D, D′) = ínf<sub>γ</sub> sup<sub>x</sub> ‖x − γ(x)‖<sub>∞</sub>,</p>
<p>donde γ recorre las biyecciones entre D y D′ en las que cualquier punto puede
emparejarse con la diagonal (a distancia (d − b)/2). La <b>distancia de
Wasserstein</b> W<sub>p</sub> sustituye el supremo por (Σ ‖x − γ(x)‖<sup>p</sup>)<sup>1/p</sup>.</p>
<p><b>Teorema de estabilidad</b> (Cohen-Steiner, Edelsbrunner, Harer, 2007; Chazal
et al., 2009): con la convención de radio,</p>
<p align="center">d<sub>B</sub>(Dgm X, Dgm Y) ≤ d<sub>H</sub>(X, Y)</p>
<p>para las filtraciones de Čech, alfa y Rips (para Rips incluso
≤ d<sub>GH</sub>(X, Y)). Además, si cada punto se mueve como mucho δ, los valores de
filtración cambian como mucho δ, y d<sub>B</sub> ≤ ‖f − g‖<sub>∞</sub> ≤ δ.</p>
<p><b>Pequeñas perturbaciones de los datos → pequeños cambios en el diagrama.</b>
Pero los <b>valores atípicos</b> aumentan mucho d<sub>H</sub>, y el diagrama puede
cambiar mucho: la persistencia de un solo parámetro no es robusta frente a
atípicos (ver el panel de persistencia biparamétrica).</p>
<p>En el diagrama central, los segmentos muestran el emparejamiento óptimo; el rojo
es el que realiza la distancia de cuello de botella.</p>
<p>Las filtraciones se calculan completas (sin radio máximo), para que el teorema
se aplique exactamente.</p>
"""

PERTURB = ["Uniforme en la bola de radio δ", "Gaussiana de desviación δ"]


class StabilityPanel(BasePanel):
    title = "Estabilidad"
    subtitle = "Pequeños cambios en los datos producen pequeños cambios en el diagrama"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.res = None
        self.curve = None
        self.data = PointCloudBox(default="Circunferencia", n=40, noise=0.0, max_n=80)
        self.controls.addWidget(self.data)
        self.data.changed.connect(self.compute)

        f = self.add_group("Perturbación")
        self.kind = QComboBox()
        self.kind.addItems(PERTURB)
        self.delta = FloatSlider(0, 0.3, 0.05)
        self.outliers = ispin(0, 20, 0)
        self.pseed = ispin(0, 9999, 1)
        f.addRow("Tipo", self.kind)
        f.addRow("δ", self.delta)
        f.addRow("Atípicos añadidos", self.outliers)
        f.addRow("Semilla", self.pseed)
        g = self.add_group("Comparación")
        self.method = QComboBox()
        self.method.addItems(["Vietoris–Rips", "Čech", "Alfa"])
        self.dim = QComboBox()
        self.dim.addItems(["H0", "H1"])
        self.dim.setCurrentIndex(1)
        g.addRow("Filtración", self.method)
        g.addRow("Diagrama", self.dim)
        self.kind.currentIndexChanged.connect(self.compute)
        self.delta.valueChanged.connect(self.compute)
        self.outliers.valueChanged.connect(self.compute)
        self.pseed.valueChanged.connect(self.compute)
        self.method.currentIndexChanged.connect(self.compute)
        self.dim.currentIndexChanged.connect(self.redraw)
        self.add_button("Calcular curva d_B(δ)", self.compute_curve, primary=False)
        self.info = self.add_info()
        self.finish_controls()
        self.set_theory(THEORY)
        self.canvas = MplCanvas()
        self.plots.addWidget(self.canvas)
        self.compute()

    # -- cálculo ------------------------------------------------------------------
    def _perturbed(self, X, delta):
        rng = np.random.default_rng(self.pseed.value())
        n, d = X.shape
        if self.kind.currentText() == PERTURB[0]:
            U = rng.normal(size=(n, d))
            U /= np.linalg.norm(U, axis=1, keepdims=True)
            U *= rng.uniform(0, 1, (n, 1)) ** (1.0 / d)
        else:
            U = rng.normal(size=(n, d))
        Y = X + delta * U
        m = self.outliers.value()
        if m:
            lo, hi = X.min(axis=0), X.max(axis=0)
            Y = np.vstack([Y, rng.uniform(lo, hi, size=(m, d))])
        return Y

    def _filt(self, X, method):
        if method == "Vietoris–Rips":
            return rips_filtration(X, 2)
        if method == "Čech":
            return cech_filtration(X, 2)
        return alpha_filtration(X)

    def compute(self):
        X = self.data.sample()
        delta = self.delta.value()
        Y = self._perturbed(X, delta)
        method = self.method.currentText()

        def job():
            PX = compute_persistence(self._filt(X, method), max_dim=1, cycles=False)
            PY = compute_persistence(self._filt(Y, method), max_dim=1, cycles=False)
            return X, Y, PX, PY

        self.curve = None
        self.run_job(job, self._done)

    def _done(self, res):
        self.res = res
        self.redraw()

    def compute_curve(self):
        X = self.data.sample()
        method = self.method.currentText()
        deltas = np.linspace(0, self.delta.maximum(), 11)
        Ys = [self._perturbed(X, d) for d in deltas]
        k = self.dim.currentIndex()

        def job():
            PX = compute_persistence(self._filt(X, method), max_dim=1, cycles=False)
            dB, dH = [], []
            for Y in Ys:
                PY = compute_persistence(self._filt(Y, method), max_dim=1, cycles=False)
                dB.append(bottleneck(PX.diagram(k), PY.diagram(k)).value)
                dH.append(hausdorff_distance(X, Y))
            return deltas, np.array(dB), np.array(dH), k

        self.run_job(job, self._curve_done, "Calculando la curva de estabilidad…")

    def _curve_done(self, res):
        self.curve = res
        self.redraw()

    # -- dibujo ---------------------------------------------------------------------
    def redraw(self, *_):
        if self.res is None:
            return
        X, Y, PX, PY = self.res
        k = self.dim.currentIndex()
        DX, DY = PX.diagram(k), PY.diagram(k)
        mB = bottleneck(DX, DY)
        mW1 = wasserstein(DX, DY, 1)
        mW2 = wasserstein(DX, DY, 2)
        dH = hausdorff_distance(X, Y)
        n = len(X)
        disp = float(np.max(np.linalg.norm(Y[:n] - X, axis=1)))

        fig = self.canvas.figure
        fig.clear()
        ncols = 3 if self.curve is not None else 2
        d = X.shape[1]
        ax0 = fig.add_subplot(1, ncols, 1, projection="3d" if d == 3 else None)
        if d == 2:
            ax0.add_collection(LineCollection(np.stack([X, Y[:n]], axis=1), colors=style.MUTED, linewidths=0.8))
            ax0.scatter(X[:, 0], X[:, 1], s=18, color=style.dim_color(0), label="X")
            ax0.scatter(Y[:, 0], Y[:, 1], s=18, color=style.dim_color(1), marker="x", label="Y")
        else:
            ax0.scatter(*X.T, s=12, color=style.dim_color(0), label="X")
            ax0.scatter(*Y.T, s=12, color=style.dim_color(1), marker="x", label="Y")
        setup_axes_for(ax0, np.vstack([X, Y]))
        ax0.legend(loc="upper right")
        ax0.set_title("Datos X y perturbación Y")

        ax1 = fig.add_subplot(1, ncols, 2)
        allD = np.vstack([DX, DY]) if len(DX) + len(DY) else np.zeros((1, 2))
        fin = allD[np.isfinite(allD)]
        vmax = float(fin.max()) if len(fin) else 1.0
        vmax = vmax * 1.1 if vmax > 0 else 1.0
        ax1.plot([0, vmax], [0, vmax], color=style.MUTED, lw=1)
        segs, worst = [], None
        for i, j in mB.pairs:
            a = DX[i] if i >= 0 else None
            b = DY[j] if j >= 0 else None
            if a is None:
                m = (b[0] + b[1]) / 2
                a = np.array([m, m])
            if b is None:
                m = (a[0] + a[1]) / 2
                b = np.array([m, m])
            if not (np.isfinite(a[1]) and np.isfinite(b[1])):
                continue
            c = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
            segs.append([a, b])
            if worst is None or c > worst[0]:
                worst = (c, [a, b])
        ax1.add_collection(LineCollection(segs, colors=style.TEXT2, linewidths=0.8))
        if worst is not None and worst[0] > 0:
            ax1.add_collection(LineCollection([worst[1]], colors=style.HIGHLIGHT, linewidths=2.5))
        fx, fy = DX[np.isfinite(DX[:, 1])], DY[np.isfinite(DY[:, 1])]
        ax1.scatter(fx[:, 0], fx[:, 1], s=30, color=style.dim_color(0), label="Dgm X", zorder=3)
        ax1.scatter(fy[:, 0], fy[:, 1], s=40, color=style.dim_color(1), marker="x", label="Dgm Y", zorder=3)
        ax1.set_xlim(-0.02 * vmax, vmax)
        ax1.set_ylim(-0.02 * vmax, vmax)
        ax1.set_aspect("equal")
        ax1.set_xlabel("nacimiento")
        ax1.set_ylabel("muerte")
        ax1.legend(loc="lower right")
        ax1.set_title(f"Diagramas H{k} y emparejamiento óptimo")

        if self.curve is not None:
            deltas, dB, dHs, kc = self.curve
            ax2 = fig.add_subplot(1, ncols, 3)
            ax2.plot(deltas, dHs, color=style.TEXT2, lw=2, label="d_H(X, Y)  (cota)")
            ax2.plot(deltas, dB, color=style.dim_color(kc), lw=2, marker="o", markersize=5,
                     label=f"d_B(Dgm H{kc})")
            ax2.set_xlabel("δ")
            ax2.set_ylabel("distancia")
            ax2.legend(loc="upper left")
            ax2.set_title("Estabilidad: d_B ≤ d_H")
        self.canvas.draw()

        ok = mB.value <= dH + 1e-12
        txt = (f"d<sub>H</sub>(X, Y) = <b>{dH:.4f}</b> &nbsp; (desplazamiento máx. de los puntos "
               f"originales = {disp:.4f})<br>"
               f"d<sub>B</sub>(H{k}) = <b>{mB.value:.4f}</b> &nbsp; {'≤' if ok else '>'} d<sub>H</sub> "
               f"{'✓' if ok else '✗'}<br>"
               f"W<sub>1</sub> = {mW1.value:.4f}, &nbsp; W<sub>2</sub> = {mW2.value:.4f}<br>"
               f"Puntos en el diagrama: X → {len(DX)}, Y → {len(DY)}")
        self.info.setText(txt)
