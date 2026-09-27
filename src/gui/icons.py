"""Monochrome line icons drawn with QPainter.

Keeps the UI free of emoji and bitmap assets: every icon is a few vector
strokes on a 24 × 24 grid, rendered at 2× for sharp HiDPI output.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

from src.gui.theme import TEXT_PRIMARY

_GRID = 24.0


def _folder(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(3, 7)
    path.lineTo(3, 18.5)
    path.quadTo(3, 20, 4.5, 20)
    path.lineTo(19.5, 20)
    path.quadTo(21, 20, 21, 18.5)
    path.lineTo(21, 9.5)
    path.quadTo(21, 8, 19.5, 8)
    path.lineTo(12, 8)
    path.lineTo(10, 5)
    path.lineTo(4.5, 5)
    path.quadTo(3, 5, 3, 6.5)
    path.closeSubpath()
    p.drawPath(path)


def _save(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(4, 4, 16, 16), 2, 2)
    p.drawRect(QRectF(8, 4, 8, 5))
    p.drawRect(QRectF(7.5, 13, 9, 7))


def _export(p: QPainter) -> None:
    p.drawLine(QPointF(12, 4), QPointF(12, 15))
    p.drawPolyline(QPolygonF([QPointF(7.5, 8.5), QPointF(12, 4), QPointF(16.5, 8.5)]))
    p.drawPolyline(QPolygonF([QPointF(4, 13), QPointF(4, 20), QPointF(20, 20), QPointF(20, 13)]))


def _gear(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 12), 3, 3)
    path = QPainterPath()
    teeth = 8
    for i in range(teeth * 2):
        a = math.pi * i / teeth
        r = 8.5 if i % 2 == 0 else 6.5
        # widen each tooth by splitting it into two points
        for da in (-0.17, 0.17):
            x = 12 + r * math.cos(a + da)
            y = 12 + r * math.sin(a + da)
            if i == 0 and da < 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
    path.closeSubpath()
    p.drawPath(path)


def _play(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(8, 5)
    path.lineTo(19, 12)
    path.lineTo(8, 19)
    path.closeSubpath()
    p.drawPath(path)


def _bolt(p: QPainter) -> None:
    p.drawPolygon(QPolygonF([
        QPointF(13.5, 3), QPointF(5.5, 13.5), QPointF(11.5, 13.5),
        QPointF(10.5, 21), QPointF(18.5, 10.5), QPointF(12.5, 10.5),
    ]))


def _wave(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(2.5, 12)
    for i in range(1, 41):
        x = 2.5 + i * 19 / 40
        path.lineTo(x, 12 - 6 * math.sin(i / 40 * 4 * math.pi) * math.exp(-((i - 20) / 16) ** 2))
    p.drawPath(path)


def _layers(p: QPainter) -> None:
    p.drawPolygon(QPolygonF([QPointF(12, 4), QPointF(21, 9), QPointF(12, 14), QPointF(3, 9)]))
    p.drawPolyline(QPolygonF([QPointF(3, 13), QPointF(12, 18), QPointF(21, 13)]))
    p.drawPolyline(QPolygonF([QPointF(3, 16.5), QPointF(12, 21.5), QPointF(21, 16.5)]))


def _target(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 12), 8, 8)
    p.drawEllipse(QPointF(12, 12), 3.5, 3.5)
    for a, b in (((12, 1.5), (12, 5)), ((12, 19), (12, 22.5)),
                 ((1.5, 12), (5, 12)), ((19, 12), (22.5, 12))):
        p.drawLine(QPointF(*a), QPointF(*b))


def _list(p: QPainter) -> None:
    for y in (6.5, 12, 17.5):
        p.drawLine(QPointF(9, y), QPointF(20, y))
        p.drawEllipse(QPointF(5, y), 1, 1)


def _clock(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
    p.drawPolyline(QPolygonF([QPointF(12, 7), QPointF(12, 12), QPointF(15.5, 14)]))


def _cpu(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(6, 6, 12, 12), 2, 2)
    p.drawRect(QRectF(9.5, 9.5, 5, 5))
    for t in (9.5, 14.5):
        for a, b in (((t, 2.5), (t, 6)), ((t, 18), (t, 21.5)),
                     ((2.5, t), (6, t)), ((18, t), (21.5, t))):
            p.drawLine(QPointF(*a), QPointF(*b))


def _info(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
    p.drawLine(QPointF(12, 11), QPointF(12, 16.5))
    p.drawPoint(QPointF(12, 7.8))


def _plus(p: QPainter) -> None:
    p.drawLine(QPointF(12, 5), QPointF(12, 19))
    p.drawLine(QPointF(5, 12), QPointF(19, 12))


def _selection(p: QPainter) -> None:
    pen = p.pen()
    dashed = QPen(pen)
    dashed.setDashPattern([2.0, 2.0])
    p.setPen(dashed)
    p.drawRect(QRectF(4, 5, 16, 14))
    p.setPen(pen)
    p.drawLine(QPointF(8, 12), QPointF(16, 12))


_ICONS: dict[str, Callable[[QPainter], None]] = {
    "open": _folder,
    "folder": _folder,
    "save": _save,
    "export": _export,
    "settings": _gear,
    "play": _play,
    "bolt": _bolt,
    "wave": _wave,
    "layers": _layers,
    "target": _target,
    "list": _list,
    "clock": _clock,
    "cpu": _cpu,
    "info": _info,
    "plus": _plus,
    "selection": _selection,
}

_FILLED = {"play", "bolt"}


def pixmap(name: str, color: str = TEXT_PRIMARY, size: int = 18, scale: float = 2.0) -> QPixmap:
    """Render icon *name* as a transparent pixmap of *size* logical pixels."""
    px = QPixmap(int(size * scale), int(size * scale))
    px.setDevicePixelRatio(scale)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(size / _GRID, size / _GRID)
    pen = QPen(QColor(color), 1.7)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(QColor(color) if name in _FILLED else Qt.NoBrush)
    _ICONS[name](p)
    p.end()
    return px


@cache
def icon(name: str, color: str = TEXT_PRIMARY, size: int = 18) -> QIcon:
    """QIcon for *name* in *color* (cached)."""
    return QIcon(pixmap(name, color, size))
