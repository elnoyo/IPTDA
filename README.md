# IPTDA

Aprender **Análisis Topológico de Datos (TDA)** de forma gráfica e interactiva.

IPTDA es una aplicación de escritorio en Python (PySide6 + Matplotlib) con
paneles interactivos sobre los conceptos centrales del TDA. Todos los
algoritmos (complejos simpliciales, reducción de la matriz borde, distancias
entre diagramas, Mapper, bifiltraciones…) están implementados en el propio
paquete con NumPy/SciPy, de forma legible, y validados con tests.

## Instalación y ejecución

Requiere Python ≥ 3.10.

```bash
pip install -r requirements.txt
python -m iptda
```

o bien, instalándola como paquete:

```bash
pip install .
iptda
```

En Linux, Qt necesita algunas bibliotecas del sistema (por ejemplo
`libegl1`, `libxkbcommon0`, `libfontconfig1`), que suelen estar ya instaladas
en un escritorio.

## Paneles

| Panel | Qué se puede hacer |
|---|---|
| **Introducción** | Recorrido sugerido, convenciones y referencias. |
| **Complejos simpliciales** | Construir complejos de **Vietoris–Rips**, **Čech**, **alfa** y **testigos** sobre nubes 2D y 3D; mover la escala *r*; comparar dos métodos lado a lado; ver bolas y diagrama de Voronoi; recuento de símplices, característica de Euler y números de Betti; **añadir/quitar puntos con el ratón**. |
| **Homología** | Complejos clásicos (círculo, esfera, toro, botella de Klein, plano proyectivo, banda de Möbius, cilindro…) o uno escrito por el usuario. Matrices borde ∂ₖ, rangos, dim Zₖ, dim Bₖ, números de Betti sobre **Z/2, Z/3, Z/5 y Q** (la torsión de RP² y de la botella de Klein solo se ve en Z/2), comprobación ∂∘∂ = 0 y χ, y **generadores** resaltados. |
| **Persistencia** | Filtraciones, **código de barras** y **diagrama de persistencia** enlazados con el complejo; clic en una barra o punto para ver su **ciclo representante**. |
| **Estabilidad** | Distancias de **cuello de botella** y **Wasserstein** con el emparejamiento óptimo dibujado; perturbaciones y atípicos; curva d_B(δ) frente a la cota d_H. |
| **Vectorizaciones** | Curvas de Betti, **paisajes de persistencia**, **imágenes de persistencia** y entropía de persistencia. |
| **Series temporales** | **Inmersión de Takens** (ventanas deslizantes) con autocorrelación y τ sugerido, detección de periodicidad y cuasiperiodicidad; **filtración por subnivel** de la señal con los pares (mínimo, máximo). |
| **Mapper** | Lentes (coordenadas, excentricidad, densidad, PCA), recubrimiento por intervalos solapados, agrupamiento (enlace simple, DBSCAN), grafo Mapper con β₀ y β₁; clic en nodos e intervalos. |
| **Persistencia biparamétrica** | Bifiltración **Rips × codensidad**: **función de Hilbert**, **códigos de barras fibrados** sobre rectas, **invariante de rango** rk(a → b) y el complejo K(a). |

Cada panel incluye a la izquierda una explicación teórica del concepto.

## Convenciones

* El parámetro **r** es el **radio** de las bolas: en Rips un símplice entra en
  r = diámetro/2, en Čech en el radio de su bola mínima envolvente y en alfa en
  su radio alfa (no al cuadrado). Así `Čech(r) ⊆ Rips(r) ⊆ Čech(√2·r)`.
* La persistencia se calcula con coeficientes en **Z/2**.
* Si se construyen símplices hasta dimensión D, solo se informa de la
  homología hasta dimensión D − 1.
* Colores fijos por dimensión: H0 azul, H1 naranja, H2 verde; selección en rojo.

## Estructura

```
iptda/
├── core/                    # matemáticas, sin dependencias gráficas
│   ├── datasets.py          # nubes de puntos 2D/3D de topología conocida
│   ├── geometry.py          # esferas circunscritas, bola mínima, maxmin, Hausdorff
│   ├── filtration.py        # Rips, Čech, alfa, testigos, subnivel, complejos estáticos
│   ├── persistence.py       # reducción de la matriz borde / cofrontera (Z/2)
│   ├── homology.py          # matrices borde, rangos sobre Z/p y Q, Betti, generadores
│   ├── complexes_library.py # triangulaciones clásicas y polígonos fundamentales
│   ├── distances.py         # cuello de botella y Wasserstein (exactas)
│   ├── vectorization.py     # curvas de Betti, paisajes, imágenes, entropía
│   ├── timeseries.py        # señales, Takens, autocorrelación
│   ├── mapper.py            # lentes, recubrimientos, clustering, grafo Mapper
│   └── multipers.py         # bifiltración, función de Hilbert, códigos fibrados, rango
├── gui/                     # interfaz PySide6 + Matplotlib
│   ├── widgets.py, plotting.py, style.py
│   └── panels/              # un módulo por panel
└── app.py                   # ventana principal
tests/                       # tests del núcleo y de la interfaz
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Los tests comprueban, entre otras cosas, que los diagramas de Rips y alfa
coinciden con los de [GUDHI](https://gudhi.inria.fr/) (si está instalado), que
alfa y Čech dan los mismos diagramas, que β(r) coincide con la homología
calculada directamente, los números de Betti de cada superficie en cada
cuerpo, la distancia de cuello de botella frente a fuerza bruta y GUDHI, el
teorema de estabilidad y los invariantes biparamétricos frente a cálculos
directos. Los tests de la interfaz se ejecutan sin pantalla
(`QT_QPA_PLATFORM=offscreen`).

## Licencia

GPL-3.0 (ver `LICENSE`).
