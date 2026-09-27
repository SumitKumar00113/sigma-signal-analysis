"""Application entry point.

Launches the PySide6 GUI, applies the dark theme, and opens the main window.

``--self-test`` instead checks the installation (or a packaged app) without
showing a window: it loads the classifier model, analyses a synthetic QPSK
signal with the hybrid classifier, builds the main window off-screen, and
exits with status 0 on success.
"""

from __future__ import annotations

import os
import sys


def _self_test() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from src.core.models import RecordingMetadata
    from src.dsp import synth
    from src.dsp.pipeline import AnalysisPipeline, PipelineConfig
    from src.gui.main_window import MainWindow
    from src.gui.theme import APP_VERSION, apply_theme
    from src.ml.model import BUNDLED_MODEL, default_model_path, load_default_model, ml_available

    frozen = getattr(sys, "frozen", False)
    print(f"Sigma {APP_VERSION} self-test (Python {sys.version.split()[0]}, "
          f"{'packaged app' if frozen else 'source'})")
    ok = True

    print(f"  ML extras available: {ml_available()}")
    print(f"  bundled model: {BUNDLED_MODEL} "
          f"({'present' if BUNDLED_MODEL.is_file() else 'MISSING'})")
    ok &= BUNDLED_MODEL.is_file()
    model = load_default_model()
    print(f"  model in use: {default_model_path()} "
          f"({len(model.classes)} classes)" if model else "  model in use: none")
    ok &= model is not None

    np.random.seed(0)
    x, _ = synth.generate_qpsk(4000, 4800.0, 48000.0, snr_db=15)
    res = AnalysisPipeline(PipelineConfig(classifier_mode="hybrid")).run(
        x, RecordingMetadata(sample_rate_hz=48000.0))
    mod = res.analysis.modulation.value
    print(f"  analysis of synthetic QPSK: {mod} ({res.analysis.modulation_confidence:.0%}, "
          f"decided by {res.classifier_source})")
    ok &= mod == "QPSK"

    app = QApplication(sys.argv[:1])
    apply_theme(app)
    win = MainWindow()
    win.show()
    app.processEvents()
    print(f"  main window: {win.windowTitle()}")
    win.close()

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def main() -> None:
    """Launch the Sigma Signal Analysis application."""
    import multiprocessing

    multiprocessing.freeze_support()     # classifier training uses worker processes
    if "--self-test" in sys.argv[1:]:
        sys.exit(_self_test())

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from src.gui.main_window import MainWindow
    from src.gui.theme import APP_ICON, APP_NAME, APP_VERSION, apply_theme

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Sigma")
    app.setApplicationVersion(APP_VERSION)
    if APP_ICON.is_file():
        app.setWindowIcon(QIcon(str(APP_ICON)))

    apply_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
