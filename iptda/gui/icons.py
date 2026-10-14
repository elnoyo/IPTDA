"""Iconos vectoriales dibujados con QPainter (sin ficheros externos)."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def _pen(color, w=1.7):
    p = QPen(QColor(color))
    p.setWidthF(w)
    p.setCapStyle(Qt.RoundCap)
    p.setJoinStyle(Qt.RoundJoin)
    return p


def _dot(p: QPainter, x, y, r=2.0):
    p.drawEllipse(QPointF(x, y), r, r)


def _draw(name: str, p: QPainter, c: str):
    """Dibuja el icono ``name`` en un lienzo lógico de 24×24."""
    p.setPen(_pen(c))
    p.setBrush(Qt.NoBrush)
    if name == "intro":            # casa
        path = QPainterPath()
        path.moveTo(4, 11)
        path.lineTo(12, 4)
        path.lineTo(20, 11)
        p.drawPath(path)
        p.drawRoundedRect(QRectF(6.5, 10.5, 11, 9), 1.5, 1.5)
    elif name == "complexes":      # triángulo relleno con vértices
        tri = QPainterPath()
        tri.moveTo(12, 4)
        tri.lineTo(20, 19)
        tri.lineTo(4, 19)
        tri.closeSubpath()
        fill = QColor(c)
        fill.setAlphaF(0.30)
        p.setBrush(fill)
        p.drawPath(tri)
        p.setBrush(QColor(c))
        for x, y in ((12, 4), (20, 19), (4, 19)):
            _dot(p, x, y, 2.2)
    elif name == "homology":       # anillo (agujero)
        p.drawEllipse(QPointF(12, 12), 8, 8)
        p.drawEllipse(QPointF(12, 12), 3.2, 3.2)
    elif name == "persistence":    # código de barras
        for y, x0, x1 in ((6, 4, 20), (10, 4, 11), (14, 7, 16), (18, 10, 14)):
            p.setPen(_pen(c, 2.4))
            p.drawLine(QPointF(x0, y), QPointF(x1, y))
    elif name == "stability":      # dos diagramas emparejados
        p.drawLine(QPointF(4, 20), QPointF(20, 4))
        p.setBrush(QColor(c))
        _dot(p, 7, 11, 2.0)
        _dot(p, 10, 8, 2.0)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(13, 13.5), 2.2, 2.2)
        p.setPen(_pen(c, 1.2))
        p.drawLine(QPointF(7, 11), QPointF(10, 8))
    elif name == "vectorization":  # paisaje (tiendas)
        path = QPainterPath()
        path.moveTo(3, 19)
        path.lineTo(9, 7)
        path.lineTo(13, 15)
        path.lineTo(16, 10)
        path.lineTo(21, 19)
        p.drawPath(path)
        p.drawLine(QPointF(3, 20.5), QPointF(21, 20.5))
    elif name == "timeseries":     # onda
        path = QPainterPath()
        path.moveTo(3, 12)
        for i in range(1, 37):
            x = 3 + i * 0.5
            path.lineTo(x, 12 - 6 * math.sin((x - 3) / 18 * 2 * math.pi))
        p.drawPath(path)
    elif name == "mapper":         # grafo
        pts = [(5, 6), (12, 11), (19, 6), (12, 19)]
        for a, b in ((0, 1), (1, 2), (1, 3)):
            p.drawLine(QPointF(*pts[a]), QPointF(*pts[b]))
        p.setBrush(QColor(c))
        for x, y in pts:
            _dot(p, x, y, 2.6)
    elif name == "multiparameter":  # escalera en rejilla
        p.setPen(_pen(c, 1.0))
        p.drawRect(QRectF(4, 4, 16, 16))
        p.setPen(_pen(c, 2.0))
        path = QPainterPath()
        path.moveTo(4, 17)
        path.lineTo(9, 17)
        path.lineTo(9, 12)
        path.lineTo(14, 12)
        path.lineTo(14, 7)
        path.lineTo(20, 7)
        p.drawPath(path)
    elif name == "sun":
        p.drawEllipse(QPointF(12, 12), 4, 4)
        for k in range(8):
            a = k * math.pi / 4
            p.drawLine(QPointF(12 + 6.5 * math.cos(a), 12 + 6.5 * math.sin(a)),
                       QPointF(12 + 8.5 * math.cos(a), 12 + 8.5 * math.sin(a)))
    elif name == "moon":
        path = QPainterPath()
        path.addEllipse(QPointF(12, 12), 7.5, 7.5)
        cut = QPainterPath()
        cut.addEllipse(QPointF(16, 9), 6.5, 6.5)
        p.setBrush(QColor(c))
        p.drawPath(path.subtracted(cut))
    elif name == "book":
        p.drawRoundedRect(QRectF(4, 5, 7.5, 14), 1.5, 1.5)
        p.drawRoundedRect(QRectF(12.5, 5, 7.5, 14), 1.5, 1.5)
    elif name == "info":
        p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
        p.drawLine(QPointF(12, 11), QPointF(12, 16.5))
        p.setBrush(QColor(c))
        _dot(p, 12, 7.8, 1.2)


def make_icon(name: str, color: str, size: int = 22) -> QIcon:
    scale = 3                          # alta resolución
    pm = QPixmap(size * scale, size * scale)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(size * scale / 24.0, size * scale / 24.0)
    _draw(name, p, color)
    p.end()
    pm.setDevicePixelRatio(scale)
    return QIcon(pm)
