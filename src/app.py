"""Application entry point.

Launches the PySide6 GUI, applies the dark theme, and opens the main window.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow
from src.gui.theme import apply_theme


def main() -> None:
    """Launch the Sigma Signal Analysis application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Sigma Signal Analysis")
    app.setOrganizationName("Sigma")
    app.setApplicationVersion("0.1.0")

    apply_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
