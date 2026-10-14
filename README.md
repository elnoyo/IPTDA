# IPTDA

[![Tests](https://github.com/elnoyo/IPTDA/actions/workflows/tests.yml/badge.svg)](https://github.com/elnoyo/IPTDA/actions/workflows/tests.yml)

Aprender **Análisis Topológico de Datos (TDA)** de forma gráfica e interactiva.

IPTDA es una aplicación de escritorio en Python (PySide6 + Matplotlib) con
paneles interactivos sobre los conceptos centrales del TDA. Todos los
algoritmos (complejos simpliciales, reducción de la matriz borde, distancias
entre diagramas, Mapper, bifiltraciones…) están implementados en el propio
paquete con NumPy/SciPy, de forma legible, y validados con tests.

## Instalación y ejecución

Requiere Python ≥ 3.10. Con Python 3.10 y 3.11 se instala PySide6 < 6.12: la versión
6.12.0 pierde una referencia a `None` en cada llamada y el intérprete acaba abortando
(«none_dealloc»); con Python ≥ 3.12 el problema no existe.

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
| **Persistencia** | Filtraciones, **código de barras** y **diagrama de persistencia** enlazados con el complejo; clic en una barra o punto para ver su **ciclo representante**; coeficientes en **Z/2, Z/3, Z/5 o Z/7**. |
| **Estabilidad** | Distancias de **cuello de botella** y **Wasserstein** con el emparejamiento óptimo dibujado; perturbaciones y atípicos; curva d_B(δ) frente a la cota d_H. |
| **Vectorizaciones** | Curvas de Betti, **paisajes de persistencia**, **imágenes de persistencia** y entropía de persistencia. |
| **Series temporales** | **Inmersión de Takens** (ventanas deslizantes) con autocorrelación y τ sugerido, detección de periodicidad y cuasiperiodicidad; **filtración por subnivel** de la señal con los pares (mínimo, máximo). |
| **Mapper** | Lentes de **una o dos componentes** (coordenadas, excentricidad, densidad, PCA), recubrimiento por intervalos o rectángulos solapados, agrupamiento (enlace simple, DBSCAN), grafo Mapper y **nervio con triángulos**, con sus números de Betti; clic en nodos y celdas. |
| **Persistencia biparamétrica** | Bifiltración **Rips × codensidad** sobre nubes **2D o 3D**, con atípicos añadidos a voluntad: **función de Hilbert**, **códigos de barras fibrados** sobre rectas, **invariante de rango** rk(a → b) y el complejo K(a). |

## Interfaz

* Barra lateral con los paneles agrupados (Fundamentos, Persistencia, Aplicaciones) y una
  portada con tarjetas de acceso.
* Cada panel tiene una cabecera con su título, un indicador mientras se calcula en
  segundo plano, los controles en tarjetas, las gráficas y un **cajón de teoría** plegable.
* **Tema claro y oscuro** (botón de la barra lateral o `Ctrl+T`); la elección se recuerda.
  Las gráficas usan una paleta categórica validada para daltonismo, con colores fijos por
  dimensión homológica.

## Convenciones

* El parámetro **r** es el **radio** de las bolas: en Rips un símplice entra en
  r = diámetro/2, en Čech en el radio de su bola mínima envolvente y en alfa en
  su radio alfa (no al cuadrado). Así `Čech(r) ⊆ Rips(r) ⊆ Čech(√2·r)`.
* La persistencia se calcula por defecto con coeficientes en **Z/2** (también Z/3, Z/5 y
  Z/7 en el panel de persistencia; en el de homología, además, Q).
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
│   ├── persistence.py       # reducción de la matriz borde / cofrontera (Z/p)
│   ├── homology.py          # matrices borde, rangos sobre Z/p y Q, Betti, generadores
│   ├── complexes_library.py # triangulaciones clásicas y polígonos fundamentales
│   ├── distances.py         # cuello de botella y Wasserstein (exactas)
│   ├── vectorization.py     # curvas de Betti, paisajes, imágenes, entropía
│   ├── timeseries.py        # señales, Takens, autocorrelación
│   ├── mapper.py            # lentes, recubrimientos, clustering, grafo Mapper
│   └── multipers.py         # bifiltración, función de Hilbert, códigos fibrados, rango
├── gui/                     # interfaz PySide6 + Matplotlib
│   ├── style.py             # temas claro/oscuro, hoja de estilo y Matplotlib
│   ├── icons.py             # iconos vectoriales dibujados con QPainter
│   ├── widgets.py, plotting.py
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
directos, y la persistencia sobre Z/p (RP² y la botella de Klein frente a la
homología directa y a GUDHI). Los tests de la interfaz se ejecutan sin pantalla
(`QT_QPA_PLATFORM=offscreen`) y recorren todos los paneles en ambos temas.

La integración continua (GitHub Actions) ejecuta `pyflakes` y la batería completa
con Python 3.10 y 3.12 en cada *push* a `main` y en cada *pull request*.

## Licencia

GPL-3.0 (ver `LICENSE`).
