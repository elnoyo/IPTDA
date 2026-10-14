"""Pruebas de humo de la interfaz: se construyen todos los paneles y se
ejercitan sus controles e interacciones de ratón, sin pantalla
(``QT_QPA_PLATFORM=offscreen``) y con los cálculos en modo síncrono."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QtCore = pytest.importorskip("PySide6.QtCore")

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
    assert not panel.has_error, panel.status.text()


def test_main_window_visits_all_panels_and_themes(app):
    from iptda.app import PANELS, MainWindow
    from iptda.gui import style
    w = MainWindow()
    for i in range(len(PANELS)):
        w.show_panel(i)
        assert w.stack.currentWidget() is w.panels[i]
        assert w.current_panel_index() == i
    # la barra lateral contiene cabeceras de sección no seleccionables
    rows = [r for r in range(w.nav.count()) if r not in w._panel_of_row]
    assert rows and all(not (w.nav.item(r).flags() & QtCore.Qt.ItemIsSelectable) for r in rows)
    w.nav.setCurrentRow(w._row_of_panel[2])
    assert w.stack.currentWidget() is w.panels[2]
    # cambio de tema con todos los paneles creados
    before = style.current
    w.toggle_theme()
    assert style.current != before
    for p in w.panels.values():
        assert not getattr(p, "has_error", False)
    w.toggle_theme()
    assert style.current == before
    # la portada navega a los paneles
    w.panels[0].navigate.emit(7)
    assert w.stack.currentWidget() is w.panels[7]
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


@pytest.mark.skipif(__import__("sys").version_info >= (3, 12),
                    reason="desde Python 3.12 None es inmortal y su contador no cambia")
def test_qt_calls_keep_none_refcount(app):
    """Regresión: PySide6 6.12.0 con Python < 3.12 pierde una referencia a
    None en cada llamada a un método que no devuelve nada (setText…), y el
    intérprete acaba abortando («none_dealloc»). Ver requirements.txt."""
    import gc
    import sys
    from iptda.gui.panels.homology import HomologyPanel
    p = HomologyPanel()
    p.source.setCurrentText("Toro (polígono fundamental)")
    gc.collect()
    before = sys.getrefcount(None)
    for _ in range(20):
        for i in range(p.gen.count()):
            p.gen.setCurrentIndex(i)
        p._show_matrix()
        p.set_status("…")
    gc.collect()
    assert sys.getrefcount(None) >= before - 50


def test_persistence_panel(app):
    from iptda.core.filtration import METHODS
    from iptda.gui.panels.persistence import PersistencePanel
    p = PersistencePanel()
    for m in METHODS:
        p.method.setCurrentText(m)
        no_error(p)
    p.method.setCurrentText("Vietoris–Rips")
    for k in range(p.field.count()):          # coeficientes Z/2, Z/3, Z/5, Z/7
        p.field.setCurrentIndex(k)
        no_error(p)
        assert p.P.field == p.field.currentData()
    p.field.setCurrentIndex(0)
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
    assert p.sel_cell == 2
    # lente 2D: recubrimiento por rectángulos y nervio con triángulos
    p.lens2.setCurrentText("Coordenada y")
    p.nint.setValue(5)
    p.overlap.setValue(0.4)
    no_error(p)
    assert p.G.lens_dim == 2 and len(p.G.cells) == 25
    (a, b), (c, d) = p.G.cells[12]
    click(p.canvas.canvas, p.ax_lens, (a + b) / 2, (c + d) / 2)
    assert p.sel_cell == 12
    assert "Nervio" in p.info.text() or not p.G.triangles


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
    # nube 3D con atípicos añadidos
    p.cod.setCurrentIndex(0)
    p.data.n.setValue(50)
    p.data.combo.setCurrentIndex(p.data.combo.findData("Nudo de trébol"))
    p.data.outliers.setValue(20)
    no_error(p)
    assert p.X.shape == (60, 3)
    assert p.canvas.figure.axes[0].name == "3d"
