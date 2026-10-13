"""Panel de bienvenida."""

from __future__ import annotations

from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

HTML = """
<h1 style="color:#174a8b">IPTDA · Análisis Topológico de Datos interactivo</h1>
<p style="font-size:12pt">El <b>Análisis Topológico de Datos</b> (TDA) estudia la <i>forma</i> de los
datos —componentes, agujeros, cavidades, ramificaciones— con herramientas de la
topología algebraica, de manera robusta frente al ruido.</p>

<h3>Recorrido sugerido</h3>
<ol style="font-size:11pt">
<li><b>Complejos simpliciales</b>: de una nube de puntos a un complejo (Vietoris–Rips,
Čech, alfa, testigos) a una escala r.</li>
<li><b>Homología</b>: matrices borde, rangos y números de Betti de complejos clásicos
(esfera, toro, botella de Klein, plano proyectivo…), con distintos cuerpos de
coeficientes y generadores.</li>
<li><b>Persistencia</b>: haciendo variar r se obtiene una filtración; la homología
persistente la resume en un código de barras / diagrama.</li>
<li><b>Estabilidad</b>: distancias de cuello de botella y Wasserstein; los diagramas
cambian poco si los datos cambian poco.</li>
<li><b>Vectorizaciones</b>: curvas de Betti, paisajes, imágenes de persistencia y
entropía, para estadística y aprendizaje automático.</li>
<li><b>Series temporales</b>: inmersión de Takens (periodicidad, cuasiperiodicidad) y
filtraciones por subnivel.</li>
<li><b>Mapper</b>: un grafo que resume la forma de los datos a través de una función
filtro.</li>
<li><b>Persistencia biparamétrica</b>: escala × densidad; función de Hilbert, invariante
de rango y códigos de barras fibrados.</li>
</ol>

<h3>Convenciones</h3>
<ul style="font-size:11pt">
<li>El parámetro <b>r</b> es el <b>radio</b> de las bolas. En Rips un símplice entra en
r = diámetro/2; en Čech, en el radio de su bola mínima envolvente; en alfa, en su
radio alfa. Así Čech(r) ⊆ Rips(r) ⊆ Čech(√2·r).</li>
<li>La persistencia se calcula con coeficientes en <b>Z/2</b>. En el panel de homología
puede elegirse Z/2, Z/3, Z/5 o Q.</li>
<li>Con símplices hasta dimensión D solo se informa de la homología hasta D−1.</li>
<li>Colores fijos por dimensión: <span style="color:#2a78d6"><b>H0 azul</b></span>,
<span style="color:#eb6834"><b>H1 naranja</b></span>,
<span style="color:#1baf7a"><b>H2 verde</b></span>; lo seleccionado, en
<span style="color:#e34948"><b>rojo</b></span>.</li>
</ul>

<h3>Cómo usarla</h3>
<p style="font-size:11pt">Elige un panel en la barra lateral. Cada panel tiene, a la izquierda,
los controles y una explicación teórica; a la derecha, las gráficas. Las gráficas
admiten zoom, desplazamiento y exportación con la barra de herramientas; las 3D se
giran arrastrando con el ratón. Los cálculos pesados se hacen en segundo plano.</p>
<p style="font-size:11pt">Todos los algoritmos (filtraciones, reducción de la matriz borde,
distancias, Mapper, bifiltraciones) están implementados en el paquete
<code>iptda.core</code> con NumPy/SciPy, para que puedan leerse y estudiarse.</p>

<h3>Referencias</h3>
<ul>
<li>H. Edelsbrunner, J. Harer. <i>Computational Topology: An Introduction</i>. AMS, 2010.</li>
<li>G. Carlsson. Topology and data. <i>Bull. AMS</i> 46 (2009).</li>
<li>S. Oudot. <i>Persistence Theory: From Quiver Representations to Data Analysis</i>. AMS, 2015.</li>
<li>G. Singh, F. Mémoli, G. Carlsson. Topological methods for the analysis of high dimensional
data sets and 3D object recognition. <i>SPBG</i> 2007.</li>
<li>J. Perea, J. Harer. Sliding windows and persistence. <i>Found. Comput. Math.</i> 15 (2015).</li>
<li>M. Lesnick, M. Wright. Interactive visualization of 2-D persistence modules (RIVET). 2015.</li>
<li>M. Botnan, M. Lesnick. An introduction to multiparameter persistence. 2022.</li>
</ul>
"""


class IntroPanel(QWidget):
    title = "Introducción"

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        tb = QTextBrowser()
        tb.setHtml(HTML)
        tb.setOpenExternalLinks(True)
        lay.addWidget(tb)

    def on_show(self):
        pass
