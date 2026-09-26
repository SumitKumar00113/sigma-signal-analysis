"""Viewer for images decoded from a signal (NOAA APT)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget


class ImageViewer(QWidget):
    """Shows an 8-bit greyscale image, scaled to the viewer width."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._caption = QLabel("")
        self._caption.setProperty("role", "subtitle")
        layout.addWidget(self._caption)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._label)
        layout.addWidget(scroll)
        self._image: np.ndarray | None = None
        self._qimage: QImage | None = None

    @property
    def image(self) -> np.ndarray | None:
        return self._image

    def set_image(self, image: np.ndarray, caption: str = "") -> None:
        img = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        self._image = img
        h, w = img.shape
        # QImage keeps a pointer to the buffer: copy so it owns its data
        self._qimage = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
        self._caption.setText(caption)
        self._rescale()

    def clear(self) -> None:
        self._image = self._qimage = None
        self._label.clear()
        self._caption.setText("")

    def resizeEvent(self, event) -> None:  # noqa: N802 – Qt API
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._qimage is None:
            return
        width = max(200, self.width() - 30)
        self._label.setPixmap(QPixmap.fromImage(self._qimage).scaledToWidth(
            min(width, self._qimage.width()), Qt.SmoothTransformation))
