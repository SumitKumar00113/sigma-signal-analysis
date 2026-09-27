# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: builds the standalone macOS app "Sigma Signal Analysis.app".

    pip install -e '.[gui,ml]' pyinstaller
    pyinstaller --noconfirm --clean sigma.spec

Result: dist/Sigma Signal Analysis.app
Check it:  "dist/Sigma Signal Analysis.app/Contents/MacOS/Sigma" --self-test
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH)
APP_NAME = "Sigma Signal Analysis"

sys.path.insert(0, str(ROOT))          # so collect_submodules("src") can import the package

from src import __version__ as VERSION  # noqa: E402


def _no_tests(name: str) -> bool:
    return ".tests" not in name and ".testing" not in name


hiddenimports = (
    # The app's own modules: several are imported lazily inside functions
    collect_submodules("src")
    # scikit-learn: the classifier is a pickled HistGradientBoostingClassifier;
    # unpickling needs its (compiled) submodules, which static analysis misses
    + collect_submodules("sklearn.ensemble._hist_gradient_boosting", filter=_no_tests)
    + collect_submodules("sklearn.utils", filter=_no_tests)
    + collect_submodules("sklearn.tree", filter=_no_tests)
    + collect_submodules("sklearn.neighbors", filter=_no_tests)
    + ["sklearn.ensemble", "sklearn.metrics", "sklearn.model_selection",
       "sklearn.preprocessing", "joblib"]
    # SciPy subpackages used by the DSP code
    + collect_submodules("scipy.signal", filter=_no_tests)
    + collect_submodules("scipy.stats", filter=_no_tests)
    + collect_submodules("scipy.ndimage", filter=_no_tests)
    + collect_submodules("scipy.special", filter=_no_tests)
    + collect_submodules("scipy.fft", filter=_no_tests)
    + collect_submodules("scipy.linalg", filter=_no_tests)
    + collect_submodules("scipy.sparse", filter=_no_tests)
    + ["scipy.io", "scipy.io.wavfile"]
    # Qt modules pyqtgraph loads on demand (SVG export, printing, OpenGL widgets)
    + ["PySide6.QtSvg", "PySide6.QtSvgWidgets", "PySide6.QtPrintSupport",
       "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets"]
    + collect_submodules("pyqtgraph", filter=lambda n: _no_tests(n)
                         and not n.startswith(("pyqtgraph.examples", "pyqtgraph.opengl",
                                               "pyqtgraph.jupyter")))
)

a = Analysis(
    [str(ROOT / "src" / "app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        # Loaded at runtime from src/ml/models/ by src.ml.model.BUNDLED_MODEL
        (str(ROOT / "src" / "ml" / "models" / "modulation_classifier.joblib"),
         "src/ml/models"),
        # Window / Dock icon set by src.app via src.gui.theme.APP_ICON
        (str(ROOT / "src" / "gui" / "assets" / "app_icon.png"), "src/gui/assets"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter", "matplotlib", "IPython", "jupyter", "notebook", "pytest", "hypothesis",
        "PyQt5", "PyQt6", "PySide2", "pyqtgraph.opengl", "OpenGL",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Sigma",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=False,
    target_arch=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Sigma",
)

app = BUNDLE(
    coll,
    name=f"{APP_NAME}.app",
    icon=str(ROOT / "packaging" / "macos" / "Sigma.icns"),
    bundle_identifier="org.sigma.signal-analysis",
    version=VERSION,
    info_plist={
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
        "LSApplicationCategoryType": "public.app-category.developer-tools",
        "LSMinimumSystemVersion": "11.0",
    },
)
