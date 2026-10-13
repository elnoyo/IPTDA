"""Pruebas de humo de la interfaz: se construyen todos los paneles y se
ejercitan sus controles e interacciones de ratón, sin pantalla
(``QT_QPA_PLATFORM=offscreen``) y con los cálculos en modo síncrono."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backend_bases import MouseEvent
    from iptda.gui import style
    from iptda.gui.widgets import BasePanel
except ImportError as exc:   # pragma: no cover - p. ej. faltan bibliotecas del sistema
    pytest.skip(f"No se puede cargar la interfaz: {exc}", allow_module_level=True)


@pytest.fixture(scope="module")
def app():
    style.apply_matplotlib_style()
    BasePanel.synchronous = True
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def click(canvas, ax, x, y, button=1):
    """Simula un clic en las coordenadas de datos (x, y) del eje ``ax``."""
    canvas.draw()
    px, py = ax.transData.transform((x, y))
    ev = MouseEvent("button_press_event", canvas, px, py, button=button)
    canvas.callbacks.process("button_press_event", ev)


def no_error(panel):
    assert panel.status.styleSheet() != "color: #b3261e;", panel.status.text()


def test_main_window_visits_all_panels(app):
    from iptda.app import PANELS, MainWindow
    w = MainWindow()
    for i in range(len(PANELS)):
        w.nav.setCurrentRow(i)
        assert w.stack.currentWidget() is w.panels[i]
    w.close()


def test_complexes_panel(app):
    from iptda.core.filtration import METHODS
    from iptda.gui.panels.complexes import ComplexesPanel
    p = ComplexesPanel()
    for m in METHODS:
        p.method_a.setCurrentText(m)
        no_error(p)
    p.method_b.setCurrentText("(ninguno)")
    p.show_voronoi.setChecked(True)
    p.show_labels.setChecked(True)
    n0 = len(p.X)
    p.edit.setChecked(True)
    click(p.canvas.canvas, p.canvas.figure.axes[0], 0.0, 0.0, button=1)
    assert len(p.X) == n0 + 1
    click(p.canvas.canvas, p.canvas.figure.axes[0], 0.0, 0.0, button=3)
    assert len(p.X) == n0
    p.data.combo.setCurrentIndex(p.data.combo.findData("Esfera"))
    p.method_a.setCurrentText("Alfa")
    no_error(p)
    assert "vértices" in p.info_a.text()


def test_homology_panel(app):
    from iptda.core.complexes_library import LIBRARY
    from iptda.core.homology import FIELDS
    from iptda.gui.panels.homology import CUSTOM, HomologyPanel
    p = HomologyPanel()
    for name, entry in LIBRARY.items():
        p.source.setCurrentText(name)
        for field in FIELDS:
            p.field.setCurrentText(field)
            if field in entry.expected:
                assert p.report.betti == entry.expected[field], (name, field)
        p.field.setCurrentText("Z/2")
        for i in range(p.gen.count()):
            p.gen.setCurrentIndex(i)
        for i in range(p.matrix_k.count()):
            p.matrix_k.setCurrentIndex(i)
        no_error(p)
    p.text.setPlainText("0 1 2; 2 3; 3 4; 4 2")
    p._custom()
    assert p.source.currentText() == CUSTOM
    assert p.report.betti == [1, 1, 0]
    p.text.setPlainText("0 0 1")
    p._custom()
    assert "repetido" in p.status.text()


def test_persistence_panel(app):
    from iptda.core.filtration import METHODS
    from iptda.gui.panels.persistence import PersistencePanel
    p = PersistencePanel()
    for m in METHODS:
        p.method.setCurrentText(m)
        no_error(p)
    p.method.setCurrentText("Vietoris–Rips")
    h1 = [q for q in p.pairs if q.dim == 1]
    assert len(h1) >= 2
    best = max(h1, key=lambda q: q.persistence)
    click(p.canvas.canvas, p.ax_dgm, best.birth, best.death)
    assert p.selected is best
    assert p.selected.cycle
    row = p.pairs.index(best)
    click(p.canvas.canvas, p.ax_bar, best.birth + 0.01, row)
    assert p.selected is best
    p.data.combo.setCurrentIndex(p.data.combo.findData("Esfera"))
    p.method.setCurrentText("Alfa")
    p.hom_dim.setValue(2)
    no_error(p)
    assert any(q.dim == 2 and q.persistence > 0.3 for q in p.pairs)


def test_stability_panel(app):
    from iptda.gui.panels.stability import StabilityPanel
    p = StabilityPanel()
    for m in ("Vietoris–Rips", "Čech", "Alfa"):
        p.method.setCurrentText(m)
        assert "✓" in p.info.text()
    p.outliers.setValue(3)
    p.kind.setCurrentIndex(1)
    p.compute_curve()
    deltas, dB, dH, k = p.curve
    assert np.all(dB <= dH + 1e-12)
    no_error(p)


def test_vectorization_panel(app):
    from iptda.gui.panels.vectorization import VectorizationPanel
    p = VectorizationPanel()
    for k in range(3):
        p.dim.setCurrentIndex(k)
    p.method.setCurrentText("Vietoris–Rips")
    p.dim.setCurrentIndex(1)
    p.weight.setCurrentText("constante")
    no_error(p)
    assert "Entropía" in p.info.text()


def test_timeseries_panel(app):
    from iptda.core.timeseries import SIGNALS
    from iptda.gui.panels.timeseries import TimeSeriesPanel
    p = TimeSeriesPanel()
    for sig in SIGNALS:
        p.signal.setCurrentText(sig)
        no_error(p)
    p.signal.setCurrentText("Seno")
    p._suggest()
    assert p.tau.value() in (12, 13)
    p.dim.setValue(5)
    p.window.setValue(0.5)
    no_error(p)
    p.superlevel.setChecked(True)
    p.level.setValue(0.0)
    assert "β0" in p.tabs.tabText(1)


def test_mapper_panel(app):
    from iptda.core.mapper import LENSES
    from iptda.gui.panels.mapper import MapperPanel
    p = MapperPanel()
    for lens in LENSES:
        p.lens.setCurrentText(lens)
        if lens == "Coordenada z":
            assert "coordenada z" in p.status.text()      # nube 2D: error controlado
        else:
            no_error(p)
    p.lens.setCurrentText("Coordenada x")
    p.clust.setCurrentText("DBSCAN")
    no_error(p)
    p.clust.setCurrentText("Enlace simple")
    assert p.G.betti == (1, 2)
    pos = p.pos[0]
    click(p.canvas.canvas, p.ax_graph, pos[0], pos[1])
    assert p.sel_node == 0
    click(p.canvas.canvas, p.ax_lens, p.G.intervals[2][0] + 0.01, 2)
    assert p.sel_interval == 2


def test_multiparameter_panel(app):
    from iptda.gui.panels.multiparameter import MultiParameterPanel
    p = MultiParameterPanel()
    no_error(p)
    click(p.canvas.canvas, p.ax_hf, p.rg[10], p.cg[20])
    assert np.isclose(p.a[0], p.rg[10]) and np.isclose(p.a[1], p.cg[20])
    for th in (0, 45, 90):
        p.theta.setValue(th)
        p.s.setValue(0.5)
        assert "rk(a → b)" in p.info.text()
    p.cod.setCurrentIndex(1)
    p.dim.setCurrentIndex(0)
    no_error(p)
