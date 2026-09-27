"""Reusable dashboard widgets: space backdrop, glass cards, key/value rows,
striped meter bars and the segmented view switcher."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QLinearGradient,
    QPainter,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gui.icons import icon, pixmap
from src.gui.theme import (
    ACCENT_PRIMARY,
    BG_DARKEST,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

# ---------------------------------------------------------------------------
# Backdrop
# ---------------------------------------------------------------------------


class SpaceBackdrop(QWidget):
    """Container that paints a deep-space backdrop: a planet-limb glow along
    the bottom, a warm nebula haze and a faint star field."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, False)
        self._cache: QPixmap | None = None
        rng = np.random.default_rng(7)
        n = 260
        self._stars = np.column_stack([rng.random(n), rng.random(n),
                                       rng.random(n) ** 3, rng.random(n)])

    def resizeEvent(self, event) -> None:  # noqa: N802 – Qt API
        self._cache = None
        super().resizeEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802 – Qt API
        if self._cache is None or self._cache.size() != self.size() * self.devicePixelRatioF():
            self._cache = self._render()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache)
        p.end()

    def _render(self) -> QPixmap:
        w, h = max(1, self.width()), max(1, self.height())
        ratio = self.devicePixelRatioF()
        px = QPixmap(int(w * ratio), int(h * ratio))
        px.setDevicePixelRatio(ratio)
        p = QPainter(px)
        p.setRenderHint(QPainter.Antialiasing)

        base = QLinearGradient(0, 0, 0, h)
        base.setColorAt(0.0, QColor("#04060b"))
        base.setColorAt(1.0, QColor(BG_DARKEST))
        p.fillRect(QRectF(0, 0, w, h), base)

        # Stars
        p.setPen(Qt.NoPen)
        for x, y, size, alpha in self._stars:
            c = QColor(255, 255, 255, int(40 + 150 * alpha))
            p.setBrush(c)
            r = 0.4 + 1.2 * size
            p.drawEllipse(QPointF(x * w, y * h * 0.85), r, r)

        # Warm haze, upper right
        haze = QRadialGradient(QPointF(w * 0.78, h * 0.18), max(w, h) * 0.55)
        haze.setColorAt(0.0, QColor(78, 168, 245, 34))
        haze.setColorAt(1.0, QColor(78, 168, 245, 0))
        p.fillRect(QRectF(0, 0, w, h), haze)

        # Cool haze, left
        cool = QRadialGradient(QPointF(w * 0.1, h * 0.55), max(w, h) * 0.5)
        cool.setColorAt(0.0, QColor(76, 201, 240, 22))
        cool.setColorAt(1.0, QColor(76, 201, 240, 0))
        p.fillRect(QRectF(0, 0, w, h), cool)

        # Planet limb: a huge circle whose top edge peeks in at the bottom
        radius = w * 1.25
        centre = QPointF(w * 0.5, h + radius - h * 0.16)
        glow = QRadialGradient(centre, radius + h * 0.12)
        edge = radius / (radius + h * 0.12)
        glow.setColorAt(edge - 0.02, QColor(8, 14, 26, 255))
        glow.setColorAt(edge - 0.004, QColor(70, 150, 220, 90))
        glow.setColorAt(edge + 0.01, QColor(76, 201, 240, 26))
        glow.setColorAt(1.0, QColor(76, 201, 240, 0))
        p.setBrush(glow)
        p.drawEllipse(centre, radius + h * 0.12, radius + h * 0.12)
        p.end()
        return px


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------


class Card(QFrame):
    """Translucent panel with a title row, hairline rule and a body.

    ``card.body`` is the content layout; ``card.header`` accepts widgets
    placed to the right of the title.
    """

    def __init__(self, title: str = "", icon_name: str | None = None,
                 parent: QWidget | None = None, margins: int = 14) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(margins, 10, margins, margins)
        outer.setSpacing(8)

        self.header = QHBoxLayout()
        self.header.setSpacing(8)
        if icon_name:
            ic = QLabel()
            ic.setPixmap(pixmap(icon_name, TEXT_SECONDARY, 15))
            self.header.addWidget(ic)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        self.header.addWidget(self.title_label)
        self.header.addStretch()
        outer.addLayout(self.header)
        if not title:
            self.title_label.hide()

        rule = QFrame()
        rule.setObjectName("cardRule")
        rule.setVisible(bool(title))
        outer.addWidget(rule)

        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        self.body.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self.body, stretch=1)

    def set_title(self, text: str) -> None:
        self.title_label.setText(text)


class KeyValueRow(QFrame):
    """A "key  ·····  value" row with a subtle raised background."""

    def __init__(self, key: str, value: str = "—", icon_name: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("kvRow")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)
        if icon_name:
            ic = QLabel()
            ic.setPixmap(pixmap(icon_name, TEXT_SECONDARY, 14))
            lay.addWidget(ic)
        self.key_label = QLabel(key)
        self.key_label.setObjectName("kvKey")
        lay.addWidget(self.key_label)
        lay.addStretch()
        self._val = QLabel(value)
        self._val.setObjectName("kvValue")
        self._val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._val.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._val.setWordWrap(True)
        self._val.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        lay.addWidget(self._val, stretch=1)

    def set_value(self, text: str, color: str | None = None) -> None:
        self._val.setText(text)
        self._val.setStyleSheet(f"color: {color};" if color else "")

    def value(self) -> str:
        return self._val.text()


class MeterBar(QWidget):
    """Horizontal meter made of thin vertical ticks (filled ticks in the
    accent colour, the rest dimmed)."""

    def __init__(self, color: str = ACCENT_PRIMARY, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fraction = 0.0
        self._color = QColor(color)
        self.setFixedHeight(18)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_fraction(self, fraction: float, color: str | None = None) -> None:
        self._fraction = float(min(1.0, max(0.0, fraction)))
        if color:
            self._color = QColor(color)
        self.update()

    def fraction(self) -> float:
        return self._fraction

    def sizeHint(self) -> QSize:  # noqa: N802 – Qt API
        return QSize(160, 18)

    def paintEvent(self, _event) -> None:  # noqa: N802 – Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        pitch = 4
        n = max(1, w // pitch)
        filled = round(n * self._fraction)
        dim = QColor(255, 255, 255, 30)
        for i in range(n):
            if i < filled:
                c = QColor(self._color)
                # fade the leading edge a little, like a lit bar
                c.setAlpha(150 + int(105 * (i + 1) / max(1, filled)))
            else:
                c = dim
            p.fillRect(i * pitch, 0, 2, h, c)
        p.end()


class MeterRow(QWidget):
    """Label and value above a :class:`MeterBar`."""

    def __init__(self, label: str, color: str = ACCENT_PRIMARY,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.label = QLabel(label)
        self.label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        top.addWidget(self.label)
        top.addStretch()
        self.value_label = QLabel("—")
        self.value_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; "
                                       "font-weight: 700;")
        top.addWidget(self.value_label)
        lay.addLayout(top)
        self.bar = MeterBar(color)
        lay.addWidget(self.bar)

    def set(self, label: str | None, value_text: str, fraction: float,
            color: str | None = None) -> None:
        if label is not None:
            self.label.setText(label)
        self.value_label.setText(value_text)
        self.bar.set_fraction(fraction, color)

    def reset(self, label: str | None = None) -> None:
        self.set(label, "—", 0.0)


# ---------------------------------------------------------------------------
# Header controls
# ---------------------------------------------------------------------------


class SegmentedNav(QFrame):
    """Pill-style segmented control; emits the index of the clicked segment."""

    index_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("segmented")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(3, 3, 3, 3)
        self._layout.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.idClicked.connect(self.index_selected.emit)

    def set_items(self, labels: list[str], current: int = 0) -> None:
        for btn in self._group.buttons():
            self._group.removeButton(btn)
            btn.deleteLater()
        for i, text in enumerate(labels):
            btn = QPushButton(text)
            btn.setObjectName("segment")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFocusPolicy(Qt.NoFocus)
            if i < 9:
                btn.setToolTip(f"{text}  (Ctrl+{i + 1})")
            self._group.addButton(btn, i)
            self._layout.addWidget(btn)
        self.set_current(current)
        self.setVisible(len(labels) > 1)

    def set_current(self, index: int) -> None:
        btn = self._group.button(index)
        if btn is not None:
            btn.setChecked(True)

    def labels(self) -> list[str]:
        return [b.text() for b in self._group.buttons()]


def icon_button(name: str, tooltip: str, slot, parent: QWidget | None = None) -> QToolButton:
    """Square header button showing a line icon."""
    btn = QToolButton(parent)
    btn.setObjectName("iconButton")
    btn.setIcon(icon(name, TEXT_PRIMARY, 18))
    btn.setIconSize(QSize(18, 18))
    btn.setToolTip(tooltip)
    btn.setCursor(Qt.PointingHandCursor)
    btn.clicked.connect(slot)
    return btn


def primary_button(text: str, icon_name: str | None, slot, primary: bool = True,
                   parent: QWidget | None = None) -> QPushButton:
    btn = QPushButton(text, parent)
    if primary:
        btn.setProperty("primary", True)
    if icon_name:
        btn.setIcon(icon(icon_name, "#ffffff" if primary else TEXT_PRIMARY, 16))
        btn.setIconSize(QSize(14, 14))
    btn.setCursor(Qt.PointingHandCursor)
    btn.clicked.connect(lambda _checked=False: slot())
    return btn


def muted_label(text: str, size: int = 11) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {size}px;")
    lbl.setWordWrap(True)
    return lbl
